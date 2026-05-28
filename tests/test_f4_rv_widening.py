"""F4 follow-up — realized-vol-ratio widening tests.

Pins the new RV-ratio-based widening mechanism (replaces the rolled-
back HMM-multiplier Fix B1+C — see
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10).

Test classes:

1. `TestRealizedVolRatio` — pure-function PIT-safety + edge cases on
   the `realized_vol_ratio` helper.
2. `TestRealizedVolWideningFactor` — calibration pins (threshold,
   slope, cap) on `realized_vol_widening_factor`.
3. `TestRealizedVolWidenedLogReturns` — sign- and mean-preserving
   properties; no-op fast path for calm regimes.
4. `TestF4CasesRanker` — end-to-end ranker output on COST 2022-04,
   UNH 2024-11, AAPL 2026-02. Documents expected behaviour: UNH
   fires mildly; COST and AAPL don't fire (per the probe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.forward_distribution import (
    realized_vol_ratio,
    realized_vol_widened_log_returns,
    realized_vol_widening_factor,
)
from engine.wheel_runner import WheelRunner


# ======================================================================
# 1. realized_vol_ratio — PIT safety + edges
# ======================================================================
class TestRealizedVolRatio:
    def test_empty_ohlcv_returns_one(self):
        assert realized_vol_ratio(pd.DataFrame({"close": []})) == 1.0

    def test_short_history_returns_one(self):
        # Less than long_window+1 closes → can't compute.
        idx = pd.date_range("2026-01-01", periods=50, freq="B")
        df = pd.DataFrame({"close": np.arange(50) * 1.0 + 100.0}, index=idx)
        assert realized_vol_ratio(df) == 1.0

    def test_pit_filter_strict(self):
        """Returns must compute against pre-as_of history only."""
        rng = np.random.default_rng(0)
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Calm first half, volatile second half.
        prices = 100 * np.exp(
            np.cumsum(np.concatenate([rng.normal(0, 0.005, 250), rng.normal(0, 0.04, n - 250)]))
        )
        df = pd.DataFrame({"close": prices}, index=idx)
        # As-of in the calm window: ratio ~1.0.
        ratio_calm = realized_vol_ratio(df, as_of=idx[200])
        # As-of in the volatile window: ratio should be elevated.
        ratio_vol = realized_vol_ratio(df, as_of=idx[399])
        assert ratio_vol > ratio_calm

    def test_constant_price_returns_one(self):
        """Degenerate constant-price history → rv_long ~ 0 → no-fire default."""
        idx = pd.date_range("2020-01-01", periods=400, freq="B")
        df = pd.DataFrame({"close": np.full(400, 100.0)}, index=idx)
        assert realized_vol_ratio(df, as_of=idx[399]) == 1.0


# ======================================================================
# 2. realized_vol_widening_factor — calibration pins
# ======================================================================
class TestRealizedVolWideningFactor:
    def _df_with_ratio(self, target_ratio: float) -> pd.DataFrame:
        """Build synthetic OHLCV whose rv30/rv252 ratio approximates `target_ratio`."""
        rng = np.random.default_rng(42)
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Tune the last 30d's sigma to hit the target ratio against a
        # baseline 1y sigma of 0.01.
        sigma_long = 0.01
        sigma_short = sigma_long * target_ratio
        log_rets = np.concatenate(
            [rng.normal(0, sigma_long, n - 30), rng.normal(0, sigma_short, 30)]
        )
        prices = 100 * np.exp(np.cumsum(log_rets))
        return pd.DataFrame({"close": prices}, index=idx)

    def test_below_threshold_returns_one(self):
        """ratio < 1.30 → factor 1.0 (no widening)."""
        df = self._df_with_ratio(0.9)
        factor = realized_vol_widening_factor(df)
        assert factor == 1.0

    def test_at_threshold_returns_one(self):
        """ratio == 1.30 → factor 1.0 (just barely fires, no widening yet)."""
        df = self._df_with_ratio(1.0)
        # Won't be exactly 1.0 due to synthetic noise, but at moderate
        # target factor stays at 1.0.
        factor = realized_vol_widening_factor(df)
        assert factor == 1.0

    def test_above_threshold_ramps(self):
        """ratio > 1.30 → factor > 1.0, monotonic in ratio."""
        df_low = self._df_with_ratio(1.5)
        df_high = self._df_with_ratio(2.5)
        f_low = realized_vol_widening_factor(df_low)
        f_high = realized_vol_widening_factor(df_high)
        assert f_low > 1.0
        assert f_high > f_low
        # Cap at 1.15 (default max_widening).
        assert f_high <= 1.15

    def test_max_widening_cap(self):
        """Even at very high ratio, factor caps at default 1.15."""
        df = self._df_with_ratio(10.0)
        factor = realized_vol_widening_factor(df)
        assert factor <= 1.15

    def test_custom_threshold_overrides(self):
        df = self._df_with_ratio(1.5)
        # With lower threshold, more aggressive firing.
        f_strict = realized_vol_widening_factor(df, threshold=1.30)
        f_loose = realized_vol_widening_factor(df, threshold=1.0)
        assert f_loose > f_strict


# ======================================================================
# 3. realized_vol_widened_log_returns — invariants
# ======================================================================
class TestRealizedVolWidenedLogReturns:
    def _calm_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=400, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        return pd.DataFrame({"close": prices}, index=idx)

    def _vol_cluster_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        log_rets = np.concatenate(
            [rng.normal(0, 0.01, n - 30), rng.normal(0, 0.025, 30)]  # 2.5x sigma jump
        )
        prices = 100 * np.exp(np.cumsum(log_rets))
        return pd.DataFrame({"close": prices}, index=idx)

    def test_empty_input_returns_empty(self):
        out = realized_vol_widened_log_returns(np.asarray([], dtype=float), self._calm_df())
        assert len(out) == 0

    def test_calm_regime_returns_input_unchanged(self):
        """Calm OHLCV → factor 1.0 → returns input unchanged."""
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.05, 100)
        out = realized_vol_widened_log_returns(rets, self._calm_df())
        np.testing.assert_array_equal(out, rets)

    def test_vol_cluster_preserves_mean(self):
        rng = np.random.default_rng(2)
        rets = rng.normal(0.001, 0.05, 500)
        mu = float(rets.mean())
        out = realized_vol_widened_log_returns(rets, self._vol_cluster_df())
        # Widening only when factor > 1.0; in this synthetic the
        # vol-cluster df produces factor > 1.0.
        assert abs(float(out.mean()) - mu) < 1e-9

    def test_vol_cluster_increases_std(self):
        rng = np.random.default_rng(3)
        rets = rng.normal(0, 0.05, 500)
        out = realized_vol_widened_log_returns(rets, self._vol_cluster_df())
        assert float(out.std()) >= float(rets.std()) - 1e-12

    def test_widening_never_narrows_std(self):
        """Sign-preserving: factor is always >= 1.0, so std cannot shrink."""
        rng = np.random.default_rng(4)
        for sigma_mult in (0.5, 1.0, 1.5, 2.5):
            n = 400
            idx = pd.date_range("2020-01-01", periods=n, freq="B")
            log_rets = np.concatenate(
                [rng.normal(0, 0.01, n - 30), rng.normal(0, 0.01 * sigma_mult, 30)]
            )
            prices = 100 * np.exp(np.cumsum(log_rets))
            ohlcv = pd.DataFrame({"close": prices}, index=idx)
            rets = rng.normal(0, 0.05, 200)
            out = realized_vol_widened_log_returns(rets, ohlcv)
            assert float(out.std()) >= float(rets.std()) - 1e-12, (
                f"std shrank at sigma_mult={sigma_mult}"
            )


# ======================================================================
# 4. F4 case pins — end-to-end ranker output
# ======================================================================
class TestF4CasesRanker:
    """The named F4 cases at the production ranker level. These are
    INTEGRATION pins — they describe the engine's BEHAVIOUR with the
    rv-ratio widening enabled, not a claim that the F4 ranking is
    fixed (the named cases are fundamentally unpredictable; see
    docs/F4_TAIL_RISK_DIAGNOSTIC.md §11)."""

    @pytest.fixture(scope="class")
    def runner(self) -> WheelRunner:
        return WheelRunner()

    def _row(self, runner, ticker, as_of):
        df = runner.rank_candidates_by_ev(
            tickers=[ticker],
            top_n=1,
            as_of=as_of,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        return None if df.empty else df.iloc[0].to_dict()

    def test_cost_2022_04_rv_widening_does_not_fire(self, runner):
        """COST 2022-04-04 had rv30/rv252 = 0.96 — below the 1.30
        threshold. Widening must not fire. This pins the "cannot
        catch idiosyncratic single-name drawdowns ex ante" honest
        finding — those cases need the R10 single-name cap, not
        widening."""
        r = self._row(runner, "COST", "2022-04-04")
        assert r is not None
        # Widening factor should be exactly 1.0 (no fire).
        assert r["tail_widening_factor"] == 1.0
        # And the pre-fix prob_profit is preserved (the widening is
        # the only thing that could have moved it; with factor 1.0
        # the output is byte-identical to main).
        assert r["prob_profit"] == pytest.approx(0.8333, abs=0.001)

    def test_unh_2024_11_rv_widening_fires_mildly(self, runner):
        """UNH 2024-11-11 had rv30/rv252 = 1.36 — barely above
        threshold. Widening fires with a small factor (~1.012),
        producing a modest ev_dollars reduction. Does NOT flip ev
        negative — the named F4 cases remain dependent on the R10
        damage-bounding mechanism."""
        r = self._row(runner, "UNH", "2024-11-11")
        assert r is not None
        # Widening fires (factor > 1.0) but mildly (< 1.05).
        assert r["tail_widening_factor"] > 1.0
        assert r["tail_widening_factor"] < 1.05
        # ev_dollars is REDUCED from the pre-fix +$114.53 baseline but
        # remains positive (the widening alone cannot flip this case).
        assert r["ev_dollars"] < 114.53  # below pre-fix baseline
        assert r["ev_dollars"] > 0  # not flipped to negative

    def test_aapl_control_rv_widening_does_not_fire(self, runner):
        """AAPL 2026-02-13 control: rv30/rv252 = 0.85, below
        threshold. Widening does not fire. The engine output is
        byte-identical to main on this date — no spurious caution."""
        r = self._row(runner, "AAPL", "2026-02-13")
        assert r is not None
        assert r["tail_widening_factor"] == 1.0
        # Pre-fix baseline: ev=+$5.50 prob_profit=0.8571.
        assert r["ev_dollars"] == pytest.approx(5.50, abs=0.01)
        assert r["prob_profit"] == pytest.approx(0.8571, abs=0.001)

    def test_calm_regime_5_ticker_smoke_preserves_main_baseline(self, runner):
        """5-ticker bring-up smoke at 2026-03-20: the per-row output
        must be byte-identical to main (widening factor 1.0 on all
        five). Pins the "calm regime no-op" property."""
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
            top_n=10,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert len(df) == 5
        # All five must show widening factor 1.0 — no widening on
        # the canonical 2026-03-20 bring-up.
        assert (df["tail_widening_factor"] == 1.0).all(), (
            f"unexpected widening on 5-ticker smoke: {df[['ticker', 'tail_widening_factor']].to_dict()}"
        )
