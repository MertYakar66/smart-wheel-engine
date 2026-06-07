"""Tests for the defensive-roll surfacing on
:meth:`engine.wheel_tracker.WheelTracker.suggest_rolls` and its
covered-call sibling :meth:`suggest_call_rolls` (S47 F-S47-1).

S47 ("live wheel session trust audit") logged that the management
surface goes *silent* on a challenged position: a deep-ITM short put
whose only available rolls are debits returns an **empty frame** under
the default ``min_net_credit=0.0`` credit-only discipline. A trader
reads "0 candidates" as "no action" when the honest answer was "16
defensive (debit) rolls exist, all gated by the credit filter".

The fix is *additive* and stays inside CLAUDE.md section 2:

  * ``include_defensive=False`` (default) keeps the legacy behaviour
    byte-for-byte — credit-gate-failing rolls are pruned into the
    ``gate="credit"`` drops.
  * ``include_defensive=True`` scores those defensive rolls through
    :meth:`EVEngine.evaluate` (no side-channel) and surfaces them as
    rows with ``defensive=True``. Surfacing never rescues: every roll
    carries its own honest ``roll_ev`` / ``hold_ev`` and ``recommend``
    stays ``roll_ev > hold_ev``.
  * Independent of the flag, ``.attrs["defensive"]`` =
    ``{"available", "surfaced", "suppressed", "included"}`` always
    reports how many defensive rolls exist, so the silent-empty default
    is never silent.

Pinned here:

  1. Default path is unchanged: no ``defensive=True`` rows; the
     defensive count is reported as *suppressed*.
  2. ``include_defensive=True`` surfaces the debit rolls flagged
     ``defensive=True``, each scored through the engine.
  3. ``.attrs["defensive"]`` is present on every return shape and the
     ``available == surfaced + suppressed`` invariant holds.
  4. Section 2: EVEngine.evaluate is called once per surviving row
     (hold + each candidate) in both modes — no side-channel EV path —
     and a recommended defensive roll still satisfies
     ``roll_ev > hold_ev``.
  5. ``suggest_call_rolls`` mirrors all of the above.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from engine.ev_engine import EVEngine
from engine.wheel_tracker import PositionState, WheelTracker


def _synth_ohlcv(n: int = 800, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


class _FakeConn:
    def __init__(self, ohlcv: pd.DataFrame | None = None) -> None:
        self._oh = ohlcv if ohlcv is not None else _synth_ohlcv()

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._oh

    def get_risk_free_rate(self, as_of=None, tenor: str = "rate_3m") -> float:
        return 0.04


def _tracker_with_put(
    *,
    strike: float = 95.0,
    premium: float = 2.0,
    entry: date = date(2026, 1, 1),
    expiry: date = date(2026, 2, 5),
    iv: float = 0.25,
    capital: float = 100_000.0,
) -> WheelTracker:
    t = WheelTracker(initial_capital=capital, connector=_FakeConn())
    ok = t.open_short_put(
        ticker="TEST",
        strike=strike,
        premium=premium,
        entry_date=entry,
        expiration_date=expiry,
        iv=iv,
    )
    assert ok
    return t


def _tracker_with_call(
    *,
    strike: float = 110.0,
    premium: float = 1.5,
    entry: date = date(2026, 2, 5),
    expiry: date = date(2026, 3, 20),
    iv: float = 0.25,
    capital: float = 100_000.0,
) -> WheelTracker:
    """Drive a position into COVERED_CALL state via the wheel flow:
    short put -> assigned -> covered call. Mirrors the helper in
    test_suggest_rolls_drops.py."""
    t = WheelTracker(initial_capital=capital, connector=_FakeConn())
    put_entry = entry - timedelta(days=40)
    ok = t.open_short_put(
        ticker="TEST",
        strike=95.0,
        premium=2.0,
        entry_date=put_entry,
        expiration_date=entry,
        iv=iv,
    )
    assert ok
    ok = t.handle_put_assignment("TEST", entry, 95.0)
    assert ok
    assert t.positions["TEST"].state == PositionState.STOCK_OWNED
    ok = t.open_covered_call(
        ticker="TEST",
        strike=strike,
        premium=premium,
        entry_date=entry,
        expiration_date=expiry,
        iv=iv,
    )
    assert ok
    assert t.positions["TEST"].state == PositionState.COVERED_CALL
    return t


# A deep-ITM short put: every roll is a debit, so the default
# credit-only path returns an empty frame (the S47 silent-zero case).
_CHALLENGED_PUT = {
    "as_of": date(2026, 1, 22),  # entry + 21 days
    "current_spot": 82.0,
    "current_iv": 0.25,
    "risk_free_rate": 0.04,
}


def _challenged_put_tracker() -> WheelTracker:
    return _tracker_with_put(
        strike=95.0,
        premium=2.0,
        entry=date(2026, 1, 1),
        expiry=date(2026, 1, 1) + timedelta(days=35),
        iv=0.25,
    )


# ======================================================================
# 1. Default path unchanged — silence is now *reported*, not removed
# ======================================================================
class TestDefaultPathUnchanged:
    def test_default_has_no_defensive_rows_and_reports_suppressed(self):
        t = _challenged_put_tracker()
        df = t.suggest_rolls(ticker="TEST", **_CHALLENGED_PUT)  # include_defensive defaults False
        # Legacy behaviour: a deep-ITM put yields no credit rolls.
        assert df.empty
        # ...but the defensive rolls are now *visible* in the summary.
        d = df.attrs["defensive"]
        assert d["included"] is False
        assert d["surfaced"] == 0
        assert d["suppressed"] > 0
        assert d["available"] == d["suppressed"]

    def test_explicit_false_equals_default(self):
        t1 = _challenged_put_tracker()
        t2 = _challenged_put_tracker()
        df_default = t1.suggest_rolls(ticker="TEST", **_CHALLENGED_PUT)
        df_false = t2.suggest_rolls(ticker="TEST", include_defensive=False, **_CHALLENGED_PUT)
        pd.testing.assert_frame_equal(
            df_default.reset_index(drop=True), df_false.reset_index(drop=True)
        )

    def test_mixed_credit_and_defensive_labelled_correctly(self):
        """On an ATM-ish put (spot just above strike) the default credit
        floor splits the grid: some rolls clear it (credit,
        defensive=False) and some fail it (debit, defensive=True). Pin
        that BOTH subsets are present and labelled by the exact rule
        ``defensive == (net_credit_debit < min_net_credit)`` — a
        challenged all-defensive fixture would make the credit subset
        empty and the labelling assertion vacuous."""
        kw = {
            "ticker": "TEST",
            "as_of": date(2026, 1, 15),  # entry + 14 days
            "current_spot": 98.0,
            "current_iv": 0.25,
            "risk_free_rate": 0.04,
        }

        def _mk():
            return _tracker_with_put(
                strike=95.0,
                premium=2.0,
                entry=date(2026, 1, 1),
                expiry=date(2026, 1, 1) + timedelta(days=49),
                iv=0.25,
            )

        df = _mk().suggest_rolls(include_defensive=True, **kw)
        credit_rows = df[~df["defensive"]]
        defensive_rows = df[df["defensive"]]
        # Genuinely mixed — both labels exercised.
        assert len(credit_rows) > 0, "expected at least one genuine credit roll"
        assert len(defensive_rows) > 0, "expected at least one surfaced defensive roll"
        # Labelling invariant: defensive iff net_credit_debit < min_net_credit (0.0).
        for _, r in credit_rows.iterrows():
            assert r["net_credit_debit"] >= 0.0
        for _, r in defensive_rows.iterrows():
            assert r["net_credit_debit"] < 0.0
        # Additive on a NON-empty frame: the default (credit-only) result
        # equals the non-defensive subset of the include-mode result — both
        # sorted by roll_ev desc — so surfacing never alters the credit rows.
        df_default = _mk().suggest_rolls(**kw)  # include_defensive defaults False
        pd.testing.assert_frame_equal(
            df_default.reset_index(drop=True),
            credit_rows.reset_index(drop=True),
        )


# ======================================================================
# 2. include_defensive=True surfaces the debit rolls
# ======================================================================
class TestDefensiveSurfacing:
    def test_surfaces_debit_rolls_flagged_defensive(self):
        t = _challenged_put_tracker()
        df = t.suggest_rolls(ticker="TEST", include_defensive=True, **_CHALLENGED_PUT)
        assert not df.empty, "challenged put with include_defensive should surface rolls"
        defensive_rows = df[df["defensive"]]
        assert len(defensive_rows) > 0
        for _, r in defensive_rows.iterrows():
            # Defensive == failed the credit gate == debit vs min_net_credit(=0).
            assert r["net_credit_debit"] < 0.0
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])
            assert np.isfinite(r["new_premium"]) and r["new_premium"] > 0
            assert 0.0 <= r["prob_otm"] <= 1.0
            assert isinstance(r["recommend"], (bool, np.bool_))
        d = df.attrs["defensive"]
        assert d["included"] is True
        assert d["surfaced"] == len(defensive_rows)
        assert d["suppressed"] == 0
        assert d["available"] == d["surfaced"]

    def test_surfacing_is_additive_no_defensive_when_unbounded(self):
        """With min_net_credit=-1e9 nothing is defensive, so toggling
        include_defensive changes nothing — surfacing is purely additive
        over the disciplined set."""
        t1 = _challenged_put_tracker()
        t2 = _challenged_put_tracker()
        kw = dict(_CHALLENGED_PUT, min_net_credit=-1e9)
        df_off = t1.suggest_rolls(ticker="TEST", include_defensive=False, **kw)
        df_on = t2.suggest_rolls(ticker="TEST", include_defensive=True, **kw)
        assert df_off.attrs["defensive"]["available"] == 0
        assert df_on.attrs["defensive"]["available"] == 0
        # No row is defensive in either frame.
        assert not df_off["defensive"].any()
        assert not df_on["defensive"].any()
        pd.testing.assert_frame_equal(df_off.reset_index(drop=True), df_on.reset_index(drop=True))


# ======================================================================
# 3. .attrs["defensive"] is always present + invariant
# ======================================================================
class TestDefensiveSummaryAlwaysPresent:
    def test_summary_on_empty_frame(self):
        t = _challenged_put_tracker()
        df = t.suggest_rolls(ticker="TEST", **_CHALLENGED_PUT)
        assert df.empty
        assert "defensive" in df.attrs

    def test_summary_on_surfaced_frame(self):
        t = _challenged_put_tracker()
        df = t.suggest_rolls(ticker="TEST", include_defensive=True, **_CHALLENGED_PUT)
        assert "defensive" in df.attrs

    def test_summary_on_early_return(self):
        """Past-expiry early return still carries the defensive summary
        (available == 0, nothing enumerated)."""
        t = _tracker_with_put(expiry=date(2026, 1, 20))
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 25),  # past expiry
            current_spot=90.0,
            current_iv=0.25,
            risk_free_rate=0.04,
        )
        assert df.empty
        d = df.attrs["defensive"]
        assert d == {"available": 0, "surfaced": 0, "suppressed": 0, "included": False}

    def test_available_equals_surfaced_plus_suppressed(self):
        for inc in (False, True):
            t = _challenged_put_tracker()
            df = t.suggest_rolls(ticker="TEST", include_defensive=inc, **_CHALLENGED_PUT)
            d = df.attrs["defensive"]
            assert d["available"] == d["surfaced"] + d["suppressed"]
            assert d["included"] is inc


# ======================================================================
# 4. CLAUDE.md section 2 — no side-channel EV; no rescue
# ======================================================================
class TestSection2NoRescue:
    def test_evaluate_called_once_per_surfaced_row(self):
        """include_defensive=True must score every surfaced defensive
        roll through EVEngine.evaluate — exactly 1 (hold) + 1 per row.
        Guards against a future fast-path that surfaces a roll without
        the EV authority."""
        t = _challenged_put_tracker()
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            df = t.suggest_rolls(ticker="TEST", include_defensive=True, **_CHALLENGED_PUT)
        assert mock_eval.call_count == 1 + len(df), (
            f"evaluate called {mock_eval.call_count}; expected {1 + len(df)} "
            f"(1 hold + 1 per of {len(df)} surfaced rows)"
        )

    def test_suppressed_defensive_rolls_are_not_scored_on_default(self):
        """The mirror property: on the default path the suppressed
        defensive rolls are NOT scored (only the hold + any credit
        survivors), so surfacing adds work only when asked."""
        t = _challenged_put_tracker()
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            df = t.suggest_rolls(ticker="TEST", **_CHALLENGED_PUT)
        assert mock_eval.call_count == 1 + len(df)

    def test_recommended_defensive_roll_still_beats_hold(self):
        """A surfaced defensive roll may be recommended only when its
        honest roll_ev exceeds hold_ev — surfacing is not a rescue."""
        t = _challenged_put_tracker()
        df = t.suggest_rolls(ticker="TEST", include_defensive=True, **_CHALLENGED_PUT)
        for _, r in df[df["recommend"]].iterrows():
            assert r["roll_ev"] > r["hold_ev"]


# ======================================================================
# 5. suggest_call_rolls mirrors the defensive surface
# ======================================================================
class TestDefensiveCallRolls:
    # A covered call that the stock has rallied through (challenged):
    # buying it back is expensive, new OTM calls are cheap -> debits.
    _CHALLENGED_CALL = {
        "as_of": date(2026, 2, 25),
        "current_spot": 130.0,  # rallied well above the 110 strike
        "current_iv": 0.25,
        "risk_free_rate": 0.04,
    }

    def test_default_reports_suppressed_call_rolls(self):
        t = _tracker_with_call(strike=110.0)
        df = t.suggest_call_rolls(ticker="TEST", **self._CHALLENGED_CALL)
        d = df.attrs["defensive"]
        assert d["included"] is False
        assert d["surfaced"] == 0
        assert d["available"] == d["suppressed"]

    def test_include_defensive_surfaces_call_rolls(self):
        t = _tracker_with_call(strike=110.0)
        df = t.suggest_call_rolls(ticker="TEST", include_defensive=True, **self._CHALLENGED_CALL)
        d = df.attrs["defensive"]
        assert d["included"] is True
        # Lock the fixture's intent: a challenged covered call MUST have
        # defensive rolls (no silent vacuous pass if a future change zeroes them).
        assert d["available"] > 0, "challenged covered call should surface defensive rolls"
        assert not df.empty
        defensive_rows = df[df["defensive"]]
        assert len(defensive_rows) == d["surfaced"] > 0
        for _, r in defensive_rows.iterrows():
            assert r["net_credit_debit"] < 0.0
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])

    def test_call_evaluate_called_once_per_surfaced_row(self):
        t = _tracker_with_call(strike=110.0)
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            df = t.suggest_call_rolls(
                ticker="TEST", include_defensive=True, **self._CHALLENGED_CALL
            )
        assert mock_eval.call_count == 1 + len(df)
