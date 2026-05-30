"""Tests for :meth:`engine.wheel_runner.WheelRunner.rank_covered_calls_by_ev`.

The covered-call **entry** ranker — issue #118 P1, the S8 follow-up.
``open_covered_call`` takes a raw strike/premium with no EV evaluation;
this ranker is the entry parallel of ``suggest_call_rolls`` (the roll):
given a held stock position it enumerates a ``(DTE x delta)`` grid of
candidate covered calls and ranks them by the forward EV of the short-call
leg, every candidate scored through :meth:`EVEngine.evaluate`.

Coverage:
  * schema / empty-frame shape / diagnostic toggle
  * edge cases (shares < 100, bad rate, missing / short OHLCV)
  * happy path (positive-EV fixture: rows, sorting, OTM strikes, sizing)
  * CLAUDE.md section 2 — *ranks, never rescues*: a negative-EV covered
    call never surfaces as tradeable
  * CLAUDE.md section 2 — the call-count regression: every candidate
    routes through ``EVEngine.evaluate``; no side-channel EV path
  * the event-lockout gate (downgrade-only)
  * ex-dividend early-assignment plumbing
  * the ``.attrs["drops"]`` diagnostic
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.wheel_runner import (
    _CC_RANK_CORE_COLUMNS,
    _CC_RANK_DIAGNOSTIC_COLUMNS,
    WheelRunner,
)

# The full (DTE x delta) grid is 4 x 4 = 16 candidates.
_GRID_SIZE = 16

# Every entry in .attrs["drops"] must carry one of these as its "gate".
_VALID_GATES = frozenset({"data", "history", "strike", "premium", "event", "ev_threshold"})


# ----------------------------------------------------------------------
# Synthetic data + fake connector — deterministic, no data-on-disk dep.
# ----------------------------------------------------------------------
def _synth_ohlcv(
    n: int = 800, seed: int = 7, mu: float = 0.0, sigma: float = 0.009
) -> pd.DataFrame:
    """A seeded geometric-Brownian OHLCV history starting 2022-01-03."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


def _flat_ohlcv() -> pd.DataFrame:
    """Sideways, low realized-vol stock. Priced against a richer IV
    (0.30), short covered calls collect a vol-risk premium -> positive EV."""
    return _synth_ohlcv(n=800, seed=7, mu=0.0, sigma=0.009)


def _trending_ohlcv() -> pd.DataFrame:
    """Strongly up-trending stock. Short calls get run through -> the whole
    (DTE x delta) grid is negative-EV. Used to pin 'ranks, never rescues'."""
    return _synth_ohlcv(n=800, seed=13, mu=0.004, sigma=0.011)


class _FakeConn:
    """Minimal single-ticker connector for rank_covered_calls_by_ev tests."""

    def __init__(
        self,
        ohlcv: pd.DataFrame | None = None,
        *,
        iv: float = 0.30,
        rf: float = 0.04,
        dividend_yield_pct: float = 0.0,
        earnings: date | None = None,
        ex_div: tuple[date, float] | None = None,
    ) -> None:
        self._oh = ohlcv if ohlcv is not None else _flat_ohlcv()
        self._iv = iv
        self._rf = rf
        self._dvd_pct = dividend_yield_pct
        self._earnings = earnings
        self._ex_div = ex_div

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._oh

    def get_fundamentals(self, ticker: str) -> dict:
        # dividend_yield is in PERCENT, mirroring Bloomberg's eqy_dvd_yld_12m.
        return {
            "implied_vol_atm": self._iv,
            "volatility_30d": self._iv * 0.9,
            "dividend_yield": self._dvd_pct,
        }

    def get_risk_free_rate(self, as_of=None) -> float:
        return self._rf

    def get_next_earnings(self, ticker: str, as_of=None):
        if self._earnings is None:
            return None
        return {"announcement_date": self._earnings}

    def get_next_dividend(self, ticker: str, as_of=None):
        if self._ex_div is None:
            return None
        ex_date, amount = self._ex_div
        return {"ex_date": ex_date, "dividend_amount": amount}


def _runner(conn: _FakeConn | None = None) -> WheelRunner:
    r = WheelRunner()
    r._connector = conn if conn is not None else _FakeConn()
    return r


def _rank(runner: WheelRunner, **extra) -> pd.DataFrame:
    """rank_covered_calls_by_ev with deterministic defaults: event gate off,
    EV floor wide open so the threshold gate is silent unless opted in.

    `max_as_of_staleness_days=10000` disables the S33 F3 staleness gate
    for these tests — `as_of=2026-01-15` is 12+ months after the
    synthetic OHLCV window. The staleness gate is exercised
    separately by `TestCoveredCallRankerAsOfBeyondData` in
    `tests/test_pit_leaks.py`.
    """
    kw: dict = {
        "ticker": "TEST",
        "shares_held": 100,
        "as_of": "2026-01-15",
        "use_event_gate": False,
        "min_ev_dollars": -1e9,
        "top_n": 50,
        "max_as_of_staleness_days": 10000,
    }
    kw.update(extra)
    return runner.rank_covered_calls_by_ev(**kw)


def _spy_evaluate():
    """A patch context that counts EVEngine.evaluate calls and passes through."""
    original = EVEngine.evaluate

    def _pass_through(self, *a, **k):
        return original(self, *a, **k)

    return patch.object(EVEngine, "evaluate", autospec=True, side_effect=_pass_through)


# ======================================================================
# 1. Schema
# ======================================================================
class TestSchema:
    def test_columns_with_diagnostics(self):
        df = _rank(_runner())
        assert list(df.columns) == _CC_RANK_CORE_COLUMNS + _CC_RANK_DIAGNOSTIC_COLUMNS

    def test_columns_without_diagnostics(self):
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert list(df.columns) == _CC_RANK_CORE_COLUMNS

    def test_empty_result_is_correctly_shaped(self):
        """A fully gated-out run still returns the pinned column schema."""
        df = _rank(_runner(_FakeConn(ohlcv=pd.DataFrame())))
        assert df.empty
        assert list(df.columns) == _CC_RANK_CORE_COLUMNS + _CC_RANK_DIAGNOSTIC_COLUMNS

    def test_drops_attr_always_present(self):
        df = _rank(_runner())
        assert "drops" in df.attrs
        assert isinstance(df.attrs["drops"], list)

    def test_pnl_percentiles_reach_survivor_rows(self):
        # Regression for the #248 P2 fix: the row dict at
        # rank_covered_calls_by_ev writes pnl_p25/50/75 but they were
        # silently dropped by `pd.DataFrame(rows, columns=cols)` until
        # the keys were added to `_CC_RANK_CORE_COLUMNS`. The schema-
        # equality tests above pin column presence; this one pins the
        # row-dict → DataFrame link so a future change that strips the
        # writes (or the schema entries) trips here.
        df = _rank(_runner())
        assert not df.empty
        for col in ("pnl_p25", "pnl_p50", "pnl_p75"):
            assert col in df.columns
            assert df[col].notna().all(), f"{col} should be finite for survivor rows"
        # Monotone ordering: same invariant as EVResult, surfaced here on
        # the ranker output so a row-dict regression that drops one of
        # the three keys is caught even if the column survives.
        assert (df["pnl_p25"] <= df["pnl_p50"]).all()
        assert (df["pnl_p50"] <= df["pnl_p75"]).all()


# ======================================================================
# 2. Edge cases
# ======================================================================
class TestEdgeCases:
    def test_fewer_than_100_shares_raises(self):
        with pytest.raises(ValueError, match="100 shares"):
            _runner().rank_covered_calls_by_ev("TEST", shares_held=50)

    def test_zero_shares_raises(self):
        with pytest.raises(ValueError, match="100 shares"):
            _runner().rank_covered_calls_by_ev("TEST", shares_held=0)

    def test_explicit_bad_risk_free_rate_raises(self):
        with pytest.raises(ValueError, match="risk_free_rate"):
            _rank(_runner(), risk_free_rate=1.5)

    def test_missing_ohlcv_drops_with_data_gate(self):
        df = _rank(_runner(_FakeConn(ohlcv=pd.DataFrame())))
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1
        assert drops[0]["gate"] == "data"

    def test_short_history_drops_with_history_gate(self):
        df = _rank(_runner(_FakeConn(ohlcv=_synth_ohlcv(n=300))))
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1
        assert drops[0]["gate"] == "history"

    def test_short_history_passes_when_gate_disabled(self):
        df = _rank(_runner(_FakeConn(ohlcv=_synth_ohlcv(n=300))), enforce_history_gate=False)
        assert not df.empty

    def test_shares_sized_to_whole_contracts(self):
        # 250 shares -> 2 whole covered-call contracts (50 shares uncovered).
        df = _rank(_runner(), shares_held=250)
        assert (df["contracts"] == 2).all()


# ======================================================================
# 3. Happy path — positive-EV fixture
# ======================================================================
class TestHappyPath:
    def test_returns_full_grid_of_candidates(self):
        df = _rank(_runner())
        assert len(df) == _GRID_SIZE

    def test_sorted_by_ev_per_day_descending(self):
        df = _rank(_runner())
        evs = df["ev_per_day"].tolist()
        assert evs == sorted(evs, reverse=True)

    def test_all_strikes_are_otm(self):
        """A covered call is sold above spot — every strike must be OTM."""
        df = _rank(_runner())
        assert (df["strike"] > df["spot"]).all()

    def test_grid_covers_every_dte_and_delta(self):
        df = _rank(_runner())
        assert set(df["dte"]) == {21, 35, 49, 63}
        assert set(df["target_delta"]) == {0.30, 0.25, 0.20, 0.15}

    def test_core_fields_are_well_formed(self):
        df = _rank(_runner())
        as_of = date(2026, 1, 15)
        for _, r in df.iterrows():
            assert r["ticker"] == "TEST"
            assert np.isfinite(r["ev_dollars"])
            assert np.isfinite(r["ev_per_day"])
            assert np.isfinite(r["premium"]) and r["premium"] > 0
            assert 0.0 <= r["prob_profit"] <= 1.0
            assert 0.0 <= r["prob_assignment"] <= 1.0
            assert r["new_expiry"] > as_of
            assert r["contracts"] == 1

    def test_top_n_truncates(self):
        df = _rank(_runner(), top_n=3)
        assert len(df) == 3

    def test_custom_grid_is_respected(self):
        df = _rank(_runner(), target_dtes=(30,), target_deltas=(0.20,))
        assert len(df) == 1
        assert df.iloc[0]["dte"] == 30
        assert df.iloc[0]["target_delta"] == 0.20

    def test_diagnostic_fields_populated(self):
        df = _rank(_runner())
        for col in ("cvar_5", "omega_ratio", "fair_value", "breakeven_move_pct"):
            assert df[col].notna().all()


# ======================================================================
# 4. CLAUDE.md section 2 — ranks, never rescues
# ======================================================================
class TestRanksNeverRescues:
    def test_fixture_is_genuinely_negative_ev(self):
        """Sanity-check the trending fixture: with the floor wide open the
        whole grid surfaces, and every candidate is negative-EV."""
        seen = _rank(_runner(_FakeConn(ohlcv=_trending_ohlcv())))
        assert len(seen) == _GRID_SIZE
        assert (seen["ev_dollars"] < 0).all()

    def test_negative_ev_covered_calls_do_not_surface(self):
        """With the default EV floor (0.0) not one negative-EV covered call
        is presented as tradeable — every candidate is dropped, no rescue."""
        runner = _runner(_FakeConn(ohlcv=_trending_ohlcv()))
        df = runner.rank_covered_calls_by_ev(
            ticker="TEST",
            shares_held=100,
            as_of="2026-01-15",
            use_event_gate=False,
            max_as_of_staleness_days=10000,
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == _GRID_SIZE
        assert all(d["gate"] == "ev_threshold" for d in drops)

    def test_every_surviving_row_clears_the_floor(self):
        floor = 50.0
        df = _rank(_runner(), min_ev_dollars=floor)
        assert not df.empty
        assert (df["ev_dollars"] >= floor).all()

    def test_floor_above_all_candidates_returns_empty(self):
        seen = _rank(_runner())
        unreachable = float(seen["ev_dollars"].max()) + 1.0
        df = _rank(_runner(), min_ev_dollars=unreachable)
        assert df.empty
        assert len(df.attrs["drops"]) == _GRID_SIZE


# ======================================================================
# 5. CLAUDE.md section 2 — every candidate routes through EVEngine.evaluate
# ======================================================================
class TestSection2EvaluateCallCount:
    def test_evaluate_called_once_per_candidate_clean_run(self):
        """Baseline: one EVEngine.evaluate call per surviving candidate,
        nothing more. Every output row is backed by exactly one evaluate."""
        with _spy_evaluate() as spy:
            df = _rank(_runner())
        assert len(df) == _GRID_SIZE
        assert spy.call_count == _GRID_SIZE
        assert spy.call_count == len(df)

    def test_threshold_drop_reuses_the_single_evaluate(self):
        """An impossibly-high EV floor drops every candidate post-evaluation.
        Each candidate is still evaluated exactly once — recording the drop
        adds no second evaluation, and nothing rescues a dropped candidate."""
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_ev_dollars=1e12)
        assert df.empty
        assert spy.call_count == _GRID_SIZE
        assert len(df.attrs["drops"]) == _GRID_SIZE

    def test_pre_evaluate_drop_triggers_no_evaluate(self):
        """A candidate dropped at a pre-evaluation gate (an unsolvable
        0.99-delta strike) costs zero EVEngine.evaluate calls — the drop is
        logged without a side-channel evaluation."""
        with _spy_evaluate() as spy:
            df = _rank(_runner(), target_deltas=(0.99,))
        assert df.empty
        assert spy.call_count == 0
        drops = df.attrs["drops"]
        assert len(drops) == 4  # 4 DTEs x the single unsolvable delta
        assert all(d["gate"] == "strike" for d in drops)

    def test_every_candidate_is_accounted_for(self):
        """The section-2 accounting identity: every (DTE, delta) candidate
        becomes either a ranked row or a logged drop, and every
        EVEngine.evaluate call yields exactly one row or one post-evaluation
        drop. No row exists without an evaluate; no evaluate is unaccounted
        for. Holds whatever the mix of survivors and drops."""
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=30))
        )
        with _spy_evaluate() as spy:
            df = runner.rank_covered_calls_by_ev(
                ticker="TEST",
                shares_held=100,
                as_of="2026-01-15",
                use_event_gate=True,
                min_ev_dollars=-1e9,
                top_n=50,
                max_as_of_staleness_days=10000,
            )
        drops = df.attrs["drops"]
        pre_eval = [d for d in drops if d["gate"] in ("data", "history", "strike", "premium")]
        post_eval = [d for d in drops if d["gate"] in ("event", "ev_threshold")]
        # every grid candidate is exactly one of: ranked row, or drop
        assert len(df) + len(drops) == _GRID_SIZE
        # pre-evaluation drops skip evaluate entirely
        assert spy.call_count == _GRID_SIZE - len(pre_eval)
        # every evaluate call produced one row or one post-evaluation drop
        assert spy.call_count == len(df) + len(post_eval)


# ======================================================================
# 6. The event-lockout gate (downgrade-only)
# ======================================================================
class TestEventGate:
    def test_earnings_in_window_blocks_candidates(self):
        """Earnings 10 days out — inside every candidate's holding window —
        blocks the whole grid via the event lockout inside EVEngine."""
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=10))
        )
        df = runner.rank_covered_calls_by_ev(
            ticker="TEST",
            shares_held=100,
            as_of="2026-01-15",
            use_event_gate=True,
            max_as_of_staleness_days=10000,
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == _GRID_SIZE
        assert all(d["gate"] == "event" for d in drops)

    def test_event_gate_disabled_lets_candidates_through(self):
        """Same imminent earnings, gate off — the candidates rank normally."""
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=10))
        )
        df = _rank(runner)  # use_event_gate=False
        assert len(df) == _GRID_SIZE

    def test_days_to_earnings_surfaced(self):
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=40))
        )
        df = _rank(runner)  # gate off, but the column is still populated
        assert (df["days_to_earnings"] == 40).all()


# ======================================================================
# 7. Ex-dividend early-assignment plumbing
# ======================================================================
class TestExDividend:
    def test_ex_dividend_surfaced_when_in_window(self):
        """A near ex-div date is surfaced so the EVEngine can model the
        early-assignment loss on the short call leg."""
        runner = _runner(
            _FakeConn(
                ohlcv=_flat_ohlcv(),
                ex_div=(date(2026, 1, 15) + timedelta(days=20), 0.55),
            )
        )
        df = _rank(runner)
        assert (df["days_to_ex_div"] == 20).all()
        assert (df["expected_dividend"] == 0.55).all()

    def test_no_dividend_leaves_fields_null(self):
        df = _rank(_runner())
        assert df["days_to_ex_div"].isna().all()
        assert (df["expected_dividend"] == 0.0).all()

    def test_past_ex_div_is_ignored(self):
        """An ex-div date before as_of is not a future early-assignment
        risk — it must not surface."""
        runner = _runner(
            _FakeConn(
                ohlcv=_flat_ohlcv(),
                ex_div=(date(2026, 1, 15) - timedelta(days=5), 0.55),
            )
        )
        df = _rank(runner)
        assert df["days_to_ex_div"].isna().all()


# ======================================================================
# 8. Drop diagnostics
# ======================================================================
class TestDropDiagnostics:
    def test_drops_have_valid_schema(self):
        """Every drop dict carries exactly {ticker, gate, reason} with a
        gate from the known taxonomy."""
        df = _rank(_runner(_FakeConn(ohlcv=_trending_ohlcv())), min_ev_dollars=0.0)
        drops = df.attrs["drops"]
        assert drops
        for d in drops:
            assert set(d.keys()) == {"ticker", "gate", "reason"}
            assert d["gate"] in _VALID_GATES
            assert d["ticker"] == "TEST"
            assert isinstance(d["reason"], str) and d["reason"]

    def test_clean_run_has_no_drops(self):
        df = _rank(_runner())
        assert df.attrs["drops"] == []
