"""Tests for :meth:`engine.wheel_runner.WheelRunner.rank_strangles_by_ev`.

The Strangle EV layer — issue #118 P1 (last item), the S14 follow-up.
The strangle path (``engine.strangle_timing``) produced only a timing
score — never a tradeable candidate, never touching ``EVEngine``. This
ranker enumerates a ``(DTE x delta)`` grid of short strangles (short OTM
put + short OTM call) and EV-ranks them.

The non-negotiable correctness properties pinned here:

  * **Two real EVEngine.evaluate calls per candidate** — one put leg,
    one call leg, over the same forward-return path. No side channel.
  * **Composed EV is additive** — ``ev_dollars == put_ev + call_ev``.
  * **Risk metrics are NOT summed** — per-leg ``put_*`` / ``call_*``
    columns only; no blended ``cvar_5`` / ``prob_profit``.
  * **§2 — ranks, never rescues** — the floor is on the *composed* EV;
    a positive put leg cannot rescue a negative composed-EV strangle.
  * **The §4 timing gate is downgrade-only** — it drops a ticker, never
    lifts a candidate's EV.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.strangle_timing import StrangleEntryScore, StrangleTimingEngine
from engine.wheel_runner import (
    _STRANGLE_RANK_CORE_COLUMNS,
    _STRANGLE_RANK_DIAGNOSTIC_COLUMNS,
    WheelRunner,
)

_GRID_SIZE = 16  # 4 DTEs x 4 deltas
_AS_OF = "2026-01-15"
_VALID_GATES = frozenset(
    {"data", "history", "timing", "strike", "premium", "event", "ev_threshold"}
)


# ----------------------------------------------------------------------
# Synthetic data + fake connector — deterministic, no data-on-disk dep.
# ----------------------------------------------------------------------
def _synth_ohlcv(
    n: int = 800, seed: int = 7, mu: float = 0.0, sigma: float = 0.009
) -> pd.DataFrame:
    """A seeded geometric-Brownian OHLCV history with a realistic
    intraday range (the timing engine's ATR needs high/low spread)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    intraday = np.abs(rng.normal(0.0, 0.004, n))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.002, n)),
            "high": close * (1.0 + intraday),
            "low": close * (1.0 - intraday),
            "close": close,
        },
        index=idx,
    )


def _flat_ohlcv() -> pd.DataFrame:
    """Sideways, low realized-vol stock. Priced against a richer IV,
    both short legs collect a vol-risk premium -> positive composed EV."""
    return _synth_ohlcv(n=800, seed=7, mu=0.0, sigma=0.009)


def _trending_ohlcv() -> pd.DataFrame:
    """Strongly up-trending stock. The short call leg gets run through;
    the composed (put + call) EV is negative even though the put leg
    alone is healthy. Pins 'ranks, never rescues'."""
    return _synth_ohlcv(n=800, seed=13, mu=0.004, sigma=0.011)


class _FakeConn:
    """Minimal single-ticker connector for rank_strangles_by_ev tests."""

    def __init__(
        self,
        ohlcv: pd.DataFrame | None = None,
        *,
        iv: float = 0.30,
        rf: float = 0.04,
        dividend_yield_pct: float = 0.0,
        earnings: date | None = None,
    ) -> None:
        self._oh = ohlcv if ohlcv is not None else _flat_ohlcv()
        self._iv = iv
        self._rf = rf
        self._dvd_pct = dividend_yield_pct
        self._earnings = earnings

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._oh

    def get_fundamentals(self, ticker: str) -> dict:
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


def _runner(conn: _FakeConn | None = None) -> WheelRunner:
    r = WheelRunner()
    r._connector = conn if conn is not None else _FakeConn()
    return r


def _rank(runner: WheelRunner, **extra) -> pd.DataFrame:
    """rank_strangles_by_ev with deterministic defaults: both gates off,
    EV floor wide open so only the strike/premium gates can fire.

    `max_as_of_staleness_days=10000` disables the S33 F3 staleness gate
    for these tests — `_AS_OF=2026-01-15` is 12+ months after the
    `_synth_ohlcv` data window (which ends ~2025-01-28). The
    staleness gate is exercised separately by
    `TestStrangleRankerAsOfBeyondData` in `tests/test_pit_leaks.py`.
    """
    kw: dict = {
        "ticker": "TEST",
        "contracts": 1,
        "as_of": _AS_OF,
        "use_event_gate": False,
        "use_timing_gate": False,
        "min_ev_dollars": -1e9,
        "top_n": 50,
        "max_as_of_staleness_days": 10000,
    }
    kw.update(extra)
    return runner.rank_strangles_by_ev(**kw)


def _spy_evaluate():
    """A patch context that records EVEngine.evaluate calls and passes through."""
    from unittest.mock import patch

    original = EVEngine.evaluate

    def _pass_through(self, *a, **k):
        return original(self, *a, **k)

    return patch.object(EVEngine, "evaluate", autospec=True, side_effect=_pass_through)


def _fake_timing(recommendation: str, total_score: float = 65.0):
    """Build a score_entry replacement returning a fixed timing verdict."""

    def _score_entry(self, df):
        return StrangleEntryScore(total_score=total_score, recommendation=recommendation)

    return _score_entry


# ======================================================================
# 1. Schema
# ======================================================================
class TestSchema:
    def test_columns_with_diagnostics(self):
        df = _rank(_runner())
        assert list(df.columns) == _STRANGLE_RANK_CORE_COLUMNS + _STRANGLE_RANK_DIAGNOSTIC_COLUMNS

    def test_columns_without_diagnostics(self):
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert list(df.columns) == _STRANGLE_RANK_CORE_COLUMNS

    def test_empty_result_is_correctly_shaped(self):
        df = _rank(_runner(_FakeConn(ohlcv=pd.DataFrame())))
        assert df.empty
        assert list(df.columns) == _STRANGLE_RANK_CORE_COLUMNS + _STRANGLE_RANK_DIAGNOSTIC_COLUMNS

    def test_drops_attr_always_present(self):
        df = _rank(_runner())
        assert isinstance(df.attrs.get("drops"), list)


# ======================================================================
# 1b. per-leg prob_profit confidence interval (small-sample honesty)
# ======================================================================
class TestProbProfitCI:
    """The strangle ranker mirrors the put ranker's CI honesty per leg:
    shared ``n_scenarios`` (both legs walk the same forward path) plus
    each leg's Wilson 95% CI bracketing its k/N ``*_prob_profit``
    frequency. These live in ``_STRANGLE_RANK_DIAGNOSTIC_COLUMNS`` right
    after the per-leg prob_profit (where the strangle's prob_profit
    travels — there is no blended prob_profit). §2-safe: this surfaces an
    existing value's uncertainty, never recalibrating prob_profit /
    ev_dollars / verdict.
    """

    _CI_COLS = (
        "n_scenarios",
        "put_prob_profit_ci_low",
        "put_prob_profit_ci_high",
        "call_prob_profit_ci_low",
        "call_prob_profit_ci_high",
    )

    def test_ci_columns_are_in_diagnostic_schema(self):
        for col in self._CI_COLS:
            assert col in _STRANGLE_RANK_DIAGNOSTIC_COLUMNS
        # n_scenarios is pinned right after the per-leg prob_profit pair.
        pp = _STRANGLE_RANK_DIAGNOSTIC_COLUMNS.index("call_prob_profit")
        assert _STRANGLE_RANK_DIAGNOSTIC_COLUMNS[pp + 1 : pp + 6] == list(self._CI_COLS)

    def test_ci_columns_absent_without_diagnostics(self):
        # Per-leg prob_profit (and thus its CI) is a diagnostic field —
        # so the CI columns must NOT appear in the core-only frame.
        df = _rank(_runner(), include_diagnostic_fields=False)
        for col in self._CI_COLS:
            assert col not in df.columns

    def test_ci_reaches_survivor_rows_and_brackets_each_leg(self):
        # Row-dict -> DataFrame link: dropped silently by
        # `pd.DataFrame(rows, columns=cols)` unless in BOTH the row dict
        # and the diagnostic column list. On the positive-EV fixture
        # every survivor has finite CI and ci_low <= prob_profit <=
        # ci_high for each leg.
        df = _rank(_runner())
        assert not df.empty
        for col in self._CI_COLS:
            assert df[col].notna().all(), f"{col} should be finite for survivor rows"
        assert (df["put_prob_profit_ci_low"] <= df["put_prob_profit"]).all()
        assert (df["put_prob_profit"] <= df["put_prob_profit_ci_high"]).all()
        assert (df["call_prob_profit_ci_low"] <= df["call_prob_profit"]).all()
        assert (df["call_prob_profit"] <= df["call_prob_profit_ci_high"]).all()
        assert (df["n_scenarios"] > 0).all()


# ======================================================================
# 1c. per-leg prob_profit CI on a REAL survivor (Bloomberg connector)
# ======================================================================
class TestProbProfitCIRealSurvivor:
    """End-to-end against the live Bloomberg connector. Skips if no
    strangle candidate survives at the fixed as_of."""

    def test_real_survivor_ci_brackets_each_leg(self):
        runner = WheelRunner()  # real provider via SWE_DATA_PROVIDER
        df = runner.rank_strangles_by_ev(
            ticker="AAPL",
            as_of="2026-03-20",
            use_event_gate=False,
            use_timing_gate=False,
            min_ev_dollars=-1e9,
            top_n=50,
        )
        if df.empty:
            pytest.skip("no strangle survivor for AAPL at as_of=2026-03-20")
        for col in (
            "n_scenarios",
            "put_prob_profit_ci_low",
            "put_prob_profit_ci_high",
            "call_prob_profit_ci_low",
            "call_prob_profit_ci_high",
        ):
            assert col in df.columns
        row = df.iloc[0]
        assert row["n_scenarios"] is not None and row["n_scenarios"] > 0
        assert (
            row["put_prob_profit_ci_low"]
            <= row["put_prob_profit"]
            <= row["put_prob_profit_ci_high"]
        )
        assert (
            row["call_prob_profit_ci_low"]
            <= row["call_prob_profit"]
            <= row["call_prob_profit_ci_high"]
        )


# ======================================================================
# 2. Edge cases
# ======================================================================
class TestEdgeCases:
    def test_explicit_bad_risk_free_rate_raises(self):
        with pytest.raises(ValueError, match="risk_free_rate"):
            _rank(_runner(), risk_free_rate=1.5)

    def test_missing_ohlcv_drops_with_data_gate(self):
        df = _rank(_runner(_FakeConn(ohlcv=pd.DataFrame())))
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1 and drops[0]["gate"] == "data"

    def test_short_history_drops_with_history_gate(self):
        df = _rank(_runner(_FakeConn(ohlcv=_synth_ohlcv(n=300))))
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1 and drops[0]["gate"] == "history"

    def test_short_history_passes_when_gate_disabled(self):
        df = _rank(_runner(_FakeConn(ohlcv=_synth_ohlcv(n=300))), enforce_history_gate=False)
        assert not df.empty


# ======================================================================
# 3. Happy path — positive composed-EV fixture
# ======================================================================
class TestHappyPath:
    def test_returns_full_grid(self):
        df = _rank(_runner())
        assert len(df) == _GRID_SIZE

    def test_sorted_by_composed_ev_descending(self):
        df = _rank(_runner())
        evs = df["ev_dollars"].tolist()
        assert evs == sorted(evs, reverse=True)

    def test_both_strikes_straddle_spot(self):
        """A strangle is a short OTM put + short OTM call: the put strike
        sits below spot, the call strike above."""
        df = _rank(_runner())
        assert (df["put_strike"] < df["spot"]).all()
        assert (df["call_strike"] > df["spot"]).all()

    def test_grid_covers_every_dte_and_delta(self):
        df = _rank(_runner())
        assert set(df["dte"]) == {21, 35, 49, 63}
        assert set(df["target_delta"]) == {0.30, 0.25, 0.20, 0.15}

    def test_core_fields_well_formed(self):
        df = _rank(_runner())
        for _, r in df.iterrows():
            assert r["ticker"] == "TEST"
            assert np.isfinite(r["ev_dollars"])
            assert r["put_premium"] > 0 and r["call_premium"] > 0
            assert r["total_premium"] == pytest.approx(
                r["put_premium"] + r["call_premium"], abs=0.01
            )
            # the short strangle profits inside the breakeven band
            assert r["lower_breakeven"] < r["put_strike"]
            assert r["upper_breakeven"] > r["call_strike"]
            assert r["contracts"] == 1

    def test_top_n_truncates(self):
        df = _rank(_runner(), top_n=4)
        assert len(df) == 4

    def test_custom_grid_respected(self):
        df = _rank(_runner(), target_dtes=(30,), target_deltas=(0.20,))
        assert len(df) == 1
        assert df.iloc[0]["dte"] == 30 and df.iloc[0]["target_delta"] == 0.20


# ======================================================================
# 4. Composed EV is additive (the headline deliverable)
# ======================================================================
class TestComposedEvAdditive:
    def test_ev_dollars_equals_put_plus_call(self):
        """ev_dollars is exactly put_ev_dollars + call_ev_dollars
        (within display rounding) — EV is linear, so the sum is exact."""
        df = _rank(_runner())
        composed = df["put_ev_dollars"] + df["call_ev_dollars"]
        assert np.allclose(composed, df["ev_dollars"], atol=0.02)

    def test_additivity_holds_on_trending_fixture(self):
        """Additivity is path-independent — it holds even when the legs
        co-move strongly (a trending underlying)."""
        df = _rank(_runner(_FakeConn(ohlcv=_trending_ohlcv())))
        composed = df["put_ev_dollars"] + df["call_ev_dollars"]
        assert np.allclose(composed, df["ev_dollars"], atol=0.02)


# ======================================================================
# 5. §2 — ranks, never rescues (the floor is on the composed EV)
# ======================================================================
class TestRanksNeverRescues:
    def test_trending_fixture_is_negative_composed_ev(self):
        seen = _rank(_runner(_FakeConn(ohlcv=_trending_ohlcv())))
        assert len(seen) == _GRID_SIZE
        assert (seen["ev_dollars"] < 0).all()

    def test_positive_put_leg_does_not_rescue_the_strangle(self):
        """The decisive §2 property: on the trending fixture the put leg
        alone is healthy (+EV) on at least one candidate, yet the
        composed strangle is negative — and the floor is on the composed
        value, so that candidate is NOT admitted."""
        seen = _rank(_runner(_FakeConn(ohlcv=_trending_ohlcv())))
        healthy_put_but_negative_strangle = seen[
            (seen["put_ev_dollars"] > 0) & (seen["ev_dollars"] < 0)
        ]
        assert not healthy_put_but_negative_strangle.empty, (
            "expected >=1 candidate with a +EV put leg but negative composed EV"
        )
        # under the default floor, none of those surface as tradeable
        ranked = _runner(_FakeConn(ohlcv=_trending_ohlcv())).rank_strangles_by_ev(
            ticker="TEST",
            as_of=_AS_OF,
            use_event_gate=False,
            use_timing_gate=False,
            max_as_of_staleness_days=10000,
        )
        assert ranked.empty

    def test_negative_composed_ev_dropped_under_default_floor(self):
        runner = _runner(_FakeConn(ohlcv=_trending_ohlcv()))
        df = runner.rank_strangles_by_ev(
            ticker="TEST",
            as_of=_AS_OF,
            use_event_gate=False,
            use_timing_gate=False,
            max_as_of_staleness_days=10000,
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == _GRID_SIZE
        assert all(d["gate"] == "ev_threshold" for d in drops)

    def test_every_surviving_row_clears_the_floor(self):
        seen = _rank(_runner())
        assert not seen.empty
        floor = float(seen["ev_dollars"].median())
        df = _rank(_runner(), min_ev_dollars=floor)
        assert not df.empty
        assert (df["ev_dollars"] >= floor).all()


# ======================================================================
# 6. §2 — two EVEngine.evaluate calls per candidate, one per leg
# ======================================================================
class TestSection2EvaluateCallCount:
    def test_two_evaluate_calls_per_candidate(self):
        """Baseline: exactly 2 x n EVEngine.evaluate calls — one put leg,
        one call leg per candidate. No blended-strangle side channel."""
        with _spy_evaluate() as spy:
            df = _rank(_runner())
        assert len(df) == _GRID_SIZE
        assert spy.call_count == 2 * _GRID_SIZE

    def test_both_legs_one_put_one_call_same_path(self):
        """Each candidate is scored as exactly one put-leg and one
        call-leg evaluate, both over the *same* forward-return path."""
        with _spy_evaluate() as spy:
            df = _rank(_runner())
        calls = spy.call_args_list
        assert len(calls) == 2 * len(df)
        for k in range(len(df)):
            put_call, call_call = calls[2 * k], calls[2 * k + 1]
            assert put_call.args[1].option_type == "put"
            assert call_call.args[1].option_type == "call"
            # the two legs share one underlying path object
            assert put_call.kwargs["forward_log_returns"] is call_call.kwargs["forward_log_returns"]

    def test_threshold_drop_reuses_the_evaluations(self):
        """An impossibly-high composed-EV floor drops every candidate
        post-evaluation — still exactly 2 x n calls, nothing rescued."""
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_ev_dollars=1e12)
        assert df.empty
        assert spy.call_count == 2 * _GRID_SIZE
        assert len(df.attrs["drops"]) == _GRID_SIZE

    def test_pre_evaluate_drop_triggers_no_evaluate(self):
        """A candidate dropped at the strike gate (an unsolvable
        0.99-delta leg) costs zero EVEngine.evaluate calls."""
        with _spy_evaluate() as spy:
            df = _rank(_runner(), target_deltas=(0.99,))
        assert df.empty
        assert spy.call_count == 0
        drops = df.attrs["drops"]
        assert len(drops) == 4 and all(d["gate"] == "strike" for d in drops)

    def test_every_candidate_is_accounted_for(self):
        """Accounting identity: every (DTE, delta) candidate is one row
        or one drop; every evaluate pair yields one row or one
        post-evaluation drop. Holds for any survivor/drop mix."""
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=30))
        )
        with _spy_evaluate() as spy:
            df = runner.rank_strangles_by_ev(
                ticker="TEST",
                as_of=_AS_OF,
                use_event_gate=True,
                use_timing_gate=False,
                min_ev_dollars=-1e9,
                top_n=50,
                max_as_of_staleness_days=10000,
            )
        drops = df.attrs["drops"]
        pre_eval = [
            d for d in drops if d["gate"] in ("data", "history", "timing", "strike", "premium")
        ]
        post_eval = [d for d in drops if d["gate"] in ("event", "ev_threshold")]
        assert len(df) + len(drops) == _GRID_SIZE
        assert spy.call_count == 2 * (_GRID_SIZE - len(pre_eval))
        assert spy.call_count == 2 * (len(df) + len(post_eval))


# ======================================================================
# 7. Risk metrics are reported per-leg, never summed
# ======================================================================
class TestRiskMetricsNotSummed:
    def test_per_leg_risk_columns_present(self):
        df = _rank(_runner())
        for col in (
            "put_cvar_5",
            "call_cvar_5",
            "put_prob_profit",
            "call_prob_profit",
            "put_prob_assignment",
            "call_prob_assignment",
        ):
            assert col in df.columns
            assert df[col].notna().all()

    def test_no_blended_risk_column(self):
        """There is no single summed cvar_5 / prob_profit / prob_assignment
        column — those would be wrong for a nonlinear strangle payoff."""
        df = _rank(_runner())
        for forbidden in ("cvar_5", "prob_profit", "prob_assignment", "cvar_99_evt"):
            assert forbidden not in df.columns

    def test_transaction_cost_is_a_real_combined_cost(self):
        """total_transaction_cost IS summed — it is a deterministic cost,
        not a path statistic, so the sum is exact and legitimate."""
        df = _rank(_runner())
        assert (df["total_transaction_cost"] > 0).all()
        assert df["total_transaction_cost"].notna().all()

    def test_breakevens_are_exact_contract_algebra(self):
        df = _rank(_runner())
        for _, r in df.iterrows():
            assert r["lower_breakeven"] == pytest.approx(
                r["put_strike"] - r["total_premium"], abs=0.01
            )
            assert r["upper_breakeven"] == pytest.approx(
                r["call_strike"] + r["total_premium"], abs=0.01
            )


# ======================================================================
# 8. The §4 timing gate — downgrade-only
# ======================================================================
class TestTimingGate:
    def test_avoid_verdict_drops_the_ticker(self, monkeypatch):
        monkeypatch.setattr(StrangleTimingEngine, "score_entry", _fake_timing("avoid", 30.0))
        df = _runner().rank_strangles_by_ev(
            ticker="TEST",
            as_of=_AS_OF,
            use_event_gate=False,
            use_timing_gate=True,
            max_as_of_staleness_days=10000,
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1 and drops[0]["gate"] == "timing"

    def test_avoid_verdict_ignored_when_gate_disabled(self, monkeypatch):
        """The timing verdict is surfaced but does not gate when
        use_timing_gate=False — it can only remove, never silently."""
        monkeypatch.setattr(StrangleTimingEngine, "score_entry", _fake_timing("avoid", 30.0))
        df = _rank(_runner())  # use_timing_gate=False
        assert len(df) == _GRID_SIZE
        assert (df["timing_recommendation"] == "avoid").all()

    def test_strong_entry_verdict_ranks(self, monkeypatch):
        monkeypatch.setattr(StrangleTimingEngine, "score_entry", _fake_timing("strong_entry", 88.0))
        df = _runner().rank_strangles_by_ev(
            ticker="TEST",
            as_of=_AS_OF,
            use_event_gate=False,
            use_timing_gate=True,
            min_ev_dollars=-1e9,
            max_as_of_staleness_days=10000,
        )
        assert len(df) == _GRID_SIZE

    def test_timing_score_does_not_affect_ev(self, monkeypatch):
        """The timing gate can only remove a ticker — it never lifts EV.
        Two wildly different (non-avoid) timing verdicts on the same
        fixture must produce byte-identical ev_dollars."""
        monkeypatch.setattr(StrangleTimingEngine, "score_entry", _fake_timing("strong_entry", 95.0))
        strong = _rank(_runner(), use_timing_gate=True)
        monkeypatch.setattr(StrangleTimingEngine, "score_entry", _fake_timing("conditional", 61.0))
        weak = _rank(_runner(), use_timing_gate=True)
        assert strong["ev_dollars"].tolist() == weak["ev_dollars"].tolist()
        assert strong["put_ev_dollars"].tolist() == weak["put_ev_dollars"].tolist()


# ======================================================================
# 9. The event-lockout gate
# ======================================================================
class TestEventGate:
    def test_earnings_in_window_blocks_candidates(self):
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=10))
        )
        df = runner.rank_strangles_by_ev(
            ticker="TEST",
            as_of=_AS_OF,
            use_event_gate=True,
            use_timing_gate=False,
            min_ev_dollars=-1e9,
            max_as_of_staleness_days=10000,
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == _GRID_SIZE
        assert all(d["gate"] == "event" for d in drops)

    def test_event_gate_disabled_lets_candidates_through(self):
        runner = _runner(
            _FakeConn(ohlcv=_flat_ohlcv(), earnings=date(2026, 1, 15) + timedelta(days=10))
        )
        df = _rank(runner)  # use_event_gate=False
        assert len(df) == _GRID_SIZE


# ======================================================================
# 10. Drop diagnostics
# ======================================================================
class TestDropDiagnostics:
    def test_drops_have_valid_schema(self):
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
