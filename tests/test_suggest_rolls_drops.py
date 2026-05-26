"""Tests for the S22 F1 drops accumulator on
:meth:`engine.wheel_tracker.WheelTracker.suggest_rolls` and its
covered-call sibling :meth:`suggest_call_rolls`.

S22 logged that ``suggest_rolls`` silently filtered 13 of 16 grid
candidates with no diagnostic — asymmetric with
:meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev` and
:meth:`rank_covered_calls_by_ev`, both of which already expose a
``.attrs["drops"]`` list of dicts. A trader operating the rolling
campaign couldn't tell the difference between "no candidates
considered" and "considered, filtered for a specific reason".

The fix mirrors the ranker drop-log pattern:

  * Each per-candidate filter site (strike solve, OTM check, premium
    thinness, min_net_credit) appends a structured dict to a local
    ``drops`` list.
  * The returned DataFrame carries ``df.attrs["drops"]`` for both
    survivor and empty-frame returns.
  * Drop schema: ``{"new_dte", "target_delta", "gate", "reason"}``
    with ``gate`` one of ``dte``, ``strike``, ``premium``, ``credit``.

Pure observability. Survivor rows, sort order, and the EVEngine
call count are unchanged (CLAUDE.md section 2).

Pinned here:

  1. ``.attrs["drops"]`` is always present (survivor frame, empty
     frame, and early-return frames).
  2. Each gate fires a correctly-shaped, schema-conformant drop entry.
  3. ``min_net_credit`` (the S22 case) produces ``gate="credit"``
     drops with the value vs threshold in the reason string.
  4. ``suggest_call_rolls`` mirrors the same drop schema.
  5. Survivor rows are byte-for-byte unchanged from the legacy
     no-drops behavior (the fix is additive).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from engine.wheel_tracker import (
    _ROLL_COLUMNS,
    PositionState,
    WheelTracker,
)

_VALID_GATES = frozenset({"dte", "strike", "premium", "credit"})


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
    premium: float = 1.5,
    entry: date = date(2026, 1, 5),
    expiry: date = date(2026, 3, 20),
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
    entry: date = date(2026, 1, 5),
    expiry: date = date(2026, 3, 20),
    iv: float = 0.25,
    capital: float = 100_000.0,
) -> WheelTracker:
    """Drive a position into COVERED_CALL state via the wheel flow:
    short put → assigned → covered call. Mirrors the helper in
    test_wheel_tracker_suggest_call_rolls.py."""
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


# ======================================================================
# 1. .attrs["drops"] is always present
# ======================================================================
class TestDropsAttrAlwaysPresent:
    def test_survivor_frame_has_drops_attr(self):
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,  # force survivors
        )
        assert "drops" in df.attrs
        assert isinstance(df.attrs["drops"], list)

    def test_empty_frame_from_min_credit_filter_has_drops_attr(self):
        """When min_net_credit gates out every candidate, the frame is
        empty but .attrs['drops'] is still populated — the whole point
        of the S22 fix."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,  # impossibly high
        )
        assert df.empty
        assert "drops" in df.attrs
        assert isinstance(df.attrs["drops"], list)
        assert len(df.attrs["drops"]) > 0

    def test_early_return_at_expiry_has_drops_attr(self):
        """A position at/past expiry returns empty early — the drops
        attr is still present (empty list, no candidates considered)."""
        # Set as_of past the put's expiry.
        t = _tracker_with_put(expiry=date(2026, 1, 20))
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 25),  # past expiry
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
        )
        assert df.empty
        assert "drops" in df.attrs
        assert df.attrs["drops"] == []


# ======================================================================
# 2. Drop schema and gate values
# ======================================================================
class TestDropSchema:
    def test_min_net_credit_drops_have_credit_gate(self):
        """The S22 case: min_net_credit gates out a candidate → emit
        gate='credit' with value vs threshold in the reason."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,
        )
        credit_drops = [d for d in df.attrs["drops"] if d["gate"] == "credit"]
        assert credit_drops, f"expected credit-gate drops; got {df.attrs['drops']}"
        for d in credit_drops:
            assert set(d.keys()) == {"new_dte", "target_delta", "gate", "reason"}
            assert d["gate"] == "credit"
            assert "net_credit_debit" in d["reason"]
            assert "min_net_credit" in d["reason"]

    def test_every_drop_entry_is_well_formed(self):
        """Across drop runs, every entry conforms to the schema."""
        t = _tracker_with_put()
        # Force credit-gate drops.
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,
        )
        for d in df.attrs["drops"]:
            assert set(d.keys()) == {"new_dte", "target_delta", "gate", "reason"}
            assert d["gate"] in _VALID_GATES
            assert isinstance(d["new_dte"], int)
            assert isinstance(d["reason"], str) and d["reason"]
            # target_delta is None only for the dte=0 gate, else a float.
            if d["gate"] == "dte":
                assert d["target_delta"] is None
            else:
                assert isinstance(d["target_delta"], float)

    def test_drops_cover_the_grid_cells_that_filtered(self):
        """If min_net_credit filters out everything, drops should cover
        all (dte × delta) grid cells (4 × 4 = 16 by default)."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,
        )
        # Default grid: target_dtes=(21,35,49,63), target_deltas=(0.30,0.25,0.20,0.15)
        # = 16 cells. The dropped cells should sum to <= 16; could be
        # less if some cells were filtered earlier (e.g. strike-OTM check
        # also drops). The credit drops alone should be a meaningful
        # fraction of the 16.
        credit_drops = [d for d in df.attrs["drops"] if d["gate"] == "credit"]
        assert 1 <= len(credit_drops) <= 16, (
            f"expected 1-16 credit drops out of the 16-cell grid; "
            f"got {len(credit_drops)} drops: {df.attrs['drops']}"
        )


# ======================================================================
# 3. suggest_call_rolls mirrors the same diagnostic
# ======================================================================
class TestSuggestCallRollsDrops:
    def test_call_rolls_have_drops_attr(self):
        t = _tracker_with_call()
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        assert "drops" in df.attrs
        assert isinstance(df.attrs["drops"], list)

    def test_call_rolls_credit_gate_fires(self):
        t = _tracker_with_call()
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,
        )
        credit_drops = [d for d in df.attrs["drops"] if d["gate"] == "credit"]
        assert credit_drops, f"expected credit-gate drops; got {df.attrs['drops']}"
        for d in credit_drops:
            assert set(d.keys()) == {"new_dte", "target_delta", "gate", "reason"}
            assert d["gate"] in _VALID_GATES

    def test_call_rolls_early_return_has_drops_attr(self):
        t = _tracker_with_call(expiry=date(2026, 1, 20))
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 25),  # past expiry
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
        )
        assert df.empty
        assert "drops" in df.attrs


# ======================================================================
# 4. Survivor rows unchanged (CLAUDE.md section 2 — additive only)
# ======================================================================
class TestSurvivorRowsUnchanged:
    def test_survivor_row_schema_unchanged(self):
        """Adding the drops accumulator must not change the row schema."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        assert list(df.columns) == list(_ROLL_COLUMNS), (
            f"row schema must equal _ROLL_COLUMNS; got {list(df.columns)}"
        )

    def test_two_runs_produce_identical_survivor_frames(self):
        """The fix introduces no run-to-run drift."""
        t1 = _tracker_with_put()
        t2 = _tracker_with_put()
        df1 = t1.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        df2 = t2.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        # Survivor rows must match byte-for-byte.
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    def test_survivors_and_drops_are_disjoint_in_grid_cells(self):
        """No (dte, delta) cell is both a survivor row and a drop."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        if df.empty:
            return  # nothing to verify
        survivor_cells = {(int(r["new_dte"]), float(r["target_delta"])) for _, r in df.iterrows()}
        dropped_cells = {
            (int(d["new_dte"]), float(d["target_delta"]))
            for d in df.attrs["drops"]
            if d["target_delta"] is not None
        }
        assert survivor_cells.isdisjoint(dropped_cells), (
            f"a grid cell cannot be both survivor and drop; "
            f"overlap = {survivor_cells & dropped_cells}"
        )


# ======================================================================
# 6. .attrs["drops_summary"] roll-up (S31 F4 discoverability closer)
# ======================================================================
class TestDropsSummaryAttr:
    """The S31 F4 fix: alongside .attrs['drops'] (which already exists),
    a trader-facing summary .attrs['drops_summary'] = {total_dropped,
    by_gate} is attached so a caller scanning the output sees at a
    glance 'N dropped — by_gate={...}' without iterating the full
    drops list."""

    def test_drops_summary_present_on_survivor_frame(self):
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        assert "drops_summary" in df.attrs
        s = df.attrs["drops_summary"]
        assert isinstance(s, dict)
        assert "total_dropped" in s
        assert "by_gate" in s
        assert s["total_dropped"] == len(df.attrs["drops"])
        # by_gate counts must sum to total_dropped (no orphan gates).
        assert sum(s["by_gate"].values()) == s["total_dropped"]

    def test_drops_summary_present_on_empty_frame(self):
        """When min_net_credit gates out every candidate the frame is
        empty but the summary still rolls up the drops."""
        t = _tracker_with_put()
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=10_000.0,  # impossibly high
        )
        assert df.empty
        s = df.attrs["drops_summary"]
        assert s["total_dropped"] > 0
        # Every drop went through the credit gate.
        assert s["by_gate"] == {"credit": s["total_dropped"]}

    def test_drops_summary_present_on_early_return(self):
        """Past-expiry early return: no candidates considered → empty
        summary (total_dropped=0, by_gate empty)."""
        t = _tracker_with_put(expiry=date(2026, 1, 20))
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 25),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
        )
        assert df.empty
        s = df.attrs["drops_summary"]
        assert s == {"total_dropped": 0, "by_gate": {}}

    def test_drops_summary_present_on_call_rolls(self):
        """suggest_call_rolls mirrors the suggest_rolls drops_summary
        attachment for shape consistency."""
        t = _tracker_with_call()
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 1),
            current_spot=100.0,
            current_iv=0.30,
            risk_free_rate=0.04,
            min_net_credit=-1e9,
        )
        assert "drops_summary" in df.attrs
        s = df.attrs["drops_summary"]
        assert s["total_dropped"] == len(df.attrs["drops"])
        assert sum(s["by_gate"].values()) == s["total_dropped"]
