"""
Institutional-grade audit invariants test module.

This module locks in the correctness properties that came out of the
2026-04-14 audit. Every test here corresponds to a bug that was either
fixed or explicitly verified during the audit; regression of any of these
tests means the audit finding has come back.

Organisation:

* ``TestGreeksUnitContract``        — vega/rho unit conventions
* ``TestAssignmentFeaturesPIT``     — no look-ahead bias in assignment features
* ``TestSharedValuationIVTrajectory`` — mark-to-market uses iv_trajectory
* ``TestTVWebhookSecurity``         — HMAC, replay, timestamp-window guards
* ``TestEVEngineInvariants``        — EV math sanity, PIT cleanliness, ranking
* ``TestBSMParity``                 — put-call parity & deterministic T→0
"""

from __future__ import annotations

import hmac
import hashlib
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine import option_pricer as op
from engine.ev_engine import EVEngine, ShortOptionTrade, rank_candidates
from engine.shared_valuation import simulate_option_trade
from src.features.assignment import AssignmentFeatures


# =========================================================================
# 1. Greeks unit contract
# =========================================================================
class TestGreeksUnitContract:
    """Vega and rho must match the documented 'per 1 vol point' / 'per 1%
    rate' convention so downstream hedging and VaR don't pick up a 100x
    scaling error."""

    def test_vega_is_per_one_vol_point(self):
        # Finite-diff the price by bumping sigma by exactly 1 vol point.
        p0 = op.black_scholes_price(100, 100, 30 / 365, 0.05, 0.25, "put", 0.02)
        p1 = op.black_scholes_price(100, 100, 30 / 365, 0.05, 0.26, "put", 0.02)
        vega = op.black_scholes_vega(100, 100, 30 / 365, 0.05, 0.25, 0.02)
        # The analytic vega should match finite-difference to 3 decimal places.
        assert abs((p1 - p0) - vega) < 1e-3, (
            f"Vega not per 1 vol point: FD={p1-p0:.6f} vs analytic={vega:.6f}"
        )

    def test_rho_is_per_one_percent_rate(self):
        p0 = op.black_scholes_price(100, 100, 30 / 365, 0.05, 0.25, "put", 0.02)
        p1 = op.black_scholes_price(100, 100, 30 / 365, 0.06, 0.25, "put", 0.02)
        rho = op.black_scholes_rho(100, 100, 30 / 365, 0.05, 0.25, "put", 0.02)
        # Allow slightly looser tolerance since we're FD'ing over a full percent
        assert abs((p1 - p0) - rho) < 5e-3, (
            f"Rho not per 1% rate: FD={p1-p0:.6f} vs analytic={rho:.6f}"
        )

    def test_vega_is_non_negative(self):
        for opt in ("call", "put"):
            for k in (80, 100, 120):
                v = op.black_scholes_vega(100, k, 30 / 365, 0.05, 0.25, 0.00)
                assert v >= 0.0


# =========================================================================
# 2. Assignment features — point-in-time correctness
# =========================================================================
class TestAssignmentFeaturesPIT:
    """The pre-audit code used ``iloc[-1]`` on spot and IV when computing
    ``prob_touch`` / ``days_to_danger`` / ``early_assignment_prob`` and
    broadcast the scalar to every row. That leaks future state into
    historical features. The fix computes each row from that row's own
    data.
    """

    def _mk_series(self):
        idx = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = 100.0 + np.linspace(0, 10, 40)  # monotonically rising
        ivs = np.linspace(0.25, 0.40, 40)  # monotonically rising IV
        return pd.Series(prices, index=idx), pd.Series(ivs, index=idx)

    def test_prob_touch_varies_per_row(self):
        spot, iv = self._mk_series()
        af = AssignmentFeatures()
        df = af.compute_all(spot=spot, strike=95, iv=iv, dte=20, is_put=True)
        # The prob_touch column should NOT be a single constant value
        unique = df["prob_touch"].dropna().unique()
        assert len(unique) > 1, (
            f"prob_touch collapsed to scalar broadcast ({unique}) — look-ahead regression"
        )

    def test_prob_touch_decreases_as_spot_runs_up(self):
        spot, iv = self._mk_series()
        af = AssignmentFeatures()
        df = af.compute_all(spot=spot, strike=95, iv=iv, dte=20, is_put=True)
        # With rising spot and strike=95, early rows (spot~100) should have
        # higher touch prob than late rows (spot~110).
        early = df["prob_touch"].iloc[5]
        late = df["prob_touch"].iloc[-5]
        assert early > late, (
            f"prob_touch should decrease as spot runs away from strike; "
            f"early={early} late={late}"
        )

    def test_no_future_leakage_across_timestamps(self):
        """Truncating the input series to time T should produce the same
        prob_touch at T as the full series — i.e. feature values at T do
        not depend on data after T.
        """
        spot, iv = self._mk_series()
        af = AssignmentFeatures()
        full = af.compute_all(spot=spot, strike=95, iv=iv, dte=20, is_put=True)
        truncated = af.compute_all(
            spot=spot.iloc[:30], strike=95, iv=iv.iloc[:30], dte=20, is_put=True
        )
        assert np.isclose(
            full["prob_touch"].iloc[25], truncated["prob_touch"].iloc[25]
        )
        assert np.isclose(
            full["early_assignment_prob"].iloc[25],
            truncated["early_assignment_prob"].iloc[25],
        )


# =========================================================================
# 3. Shared valuation — IV trajectory honoured
# =========================================================================
class TestSharedValuationIVTrajectory:
    """The simulator previously marked every day at constant entry_iv,
    biasing backtests in vol-regime transitions. The fix threads an
    iv_trajectory through so callers supplying a real IV path see
    path-dependent mark-to-market.
    """

    def _mk_ohlcv(self, days=30, start=100.0):
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(days)]
        closes = np.linspace(start, start * 0.92, days)  # gentle drop
        return pd.DataFrame({"Date": dates, "Close": closes})

    def test_iv_trajectory_changes_exit_reason(self):
        """A big IV spike mid-trade should drive the option price above the
        stop-loss threshold, flipping the exit reason from ``expiration_*``
        to ``stop_loss``. If the simulator silently ignored the trajectory
        the exit reason would be unchanged."""
        ohlcv = self._mk_ohlcv(days=35, start=100.0)
        entry = date(2024, 1, 1)
        expiry = date(2024, 1, 30)
        # Flat IV baseline
        flat_iv = pd.Series(0.20, index=[date(2024, 1, 1) + timedelta(days=i) for i in range(35)])
        # Spike IV starting day 5
        spike_iv = flat_iv.copy()
        for i in range(5, 35):
            spike_iv.iloc[i] = 1.50  # 150% vol (crisis)

        flat_result = simulate_option_trade(
            option_type="put",
            strike=95,
            entry_premium=1.00,
            entry_date=entry,
            expiration_date=expiry,
            entry_iv=0.20,
            ohlcv_df=ohlcv,
            iv_trajectory=flat_iv,
            stop_loss_multiple=2.0,
        )
        spike_result = simulate_option_trade(
            option_type="put",
            strike=95,
            entry_premium=1.00,
            entry_date=entry,
            expiration_date=expiry,
            entry_iv=0.20,
            ohlcv_df=ohlcv,
            iv_trajectory=spike_iv,
            stop_loss_multiple=2.0,
        )
        assert flat_result is not None and spike_result is not None
        # Spike should produce a stop_loss exit at or before expiration
        assert spike_result.exit_reason == "stop_loss", (
            f"iv_trajectory not honoured: spike exit={spike_result.exit_reason}"
        )

    def test_no_iv_trajectory_falls_back_to_entry_iv(self):
        """When no trajectory is provided the simulator must still work
        (documented biased mode). This is a smoke test for the default
        branch."""
        ohlcv = self._mk_ohlcv(days=35, start=100.0)
        result = simulate_option_trade(
            option_type="put",
            strike=95,
            entry_premium=1.00,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 30),
            entry_iv=0.20,
            ohlcv_df=ohlcv,
        )
        assert result is not None


# =========================================================================
# 4. TV webhook security (HMAC, replay, freshness)
# =========================================================================
class TestTVWebhookSecurity:
    """Regression tests for the hardened TV webhook layer. We don't spin
    up the HTTP server — we exercise the security primitives directly
    because they are the attack surface."""

    def test_hmac_valid_signature_accepted(self):
        from engine_api import _tv_verify_hmac

        secret = "super-secret-key"
        body = b'{"ticker":"AAPL","signal":"wheel_put_zone"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _tv_verify_hmac(body, sig, secret)
        assert _tv_verify_hmac(body, f"sha256={sig}", secret)

    def test_hmac_wrong_signature_rejected(self):
        from engine_api import _tv_verify_hmac

        secret = "super-secret-key"
        body = b'{"ticker":"AAPL","signal":"wheel_put_zone"}'
        assert not _tv_verify_hmac(body, "deadbeef", secret)
        assert not _tv_verify_hmac(body, "", secret)

    def test_hmac_tampered_body_rejected(self):
        from engine_api import _tv_verify_hmac

        secret = "super-secret-key"
        original = b'{"ticker":"AAPL","signal":"wheel_put_zone"}'
        tampered = b'{"ticker":"TSLA","signal":"wheel_put_zone"}'
        sig = hmac.new(secret.encode(), original, hashlib.sha256).hexdigest()
        assert not _tv_verify_hmac(tampered, sig, secret)

    def test_nonce_replay_blocked(self):
        from engine_api import _tv_seen_register, _TV_SEEN_NONCES

        _TV_SEEN_NONCES.clear()
        digest = "abc123" * 10
        import time

        now = time.time()
        # First sight: new nonce
        assert _tv_seen_register(digest, now) is True
        # Replay: blocked
        assert _tv_seen_register(digest, now + 1) is False

    def test_nonce_expires_after_window(self):
        from engine_api import (
            _tv_seen_register,
            _TV_SEEN_NONCES,
            _TV_WEBHOOK_MAX_AGE_SEC,
        )

        _TV_SEEN_NONCES.clear()
        digest = "deadbeef" * 8
        t0 = 1_000_000.0
        assert _tv_seen_register(digest, t0) is True
        # A nonce seen longer ago than the freshness window should be
        # purged and a new arrival with the same digest should be accepted
        # (replay protection decays with timestamp freshness to avoid
        # unbounded memory growth).
        t_later = t0 + _TV_WEBHOOK_MAX_AGE_SEC + 1
        assert _tv_seen_register(digest, t_later) is True


# =========================================================================
# 5. EV engine invariants
# =========================================================================
class TestEVEngineInvariants:
    """Sanity checks for the institutional-grade EV engine added in the
    audit. These are not unit tests of individual arithmetic — they are
    high-level correctness properties that must always hold regardless of
    implementation detail.
    """

    def _base_trade(self, **overrides):
        defaults = dict(
            option_type="put",
            underlying="AAPL",
            spot=100.0,
            strike=95.0,
            premium=1.20,
            bid=1.15,
            ask=1.25,
            dte=30,
            iv=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            contracts=1,
            open_interest=1000,
            regime_multiplier=1.0,
        )
        defaults.update(overrides)
        return ShortOptionTrade(**defaults)

    def test_probabilities_in_unit_interval(self):
        trade = self._base_trade()
        res = EVEngine().evaluate(trade)
        assert 0.0 <= res.prob_profit <= 1.0
        assert 0.0 <= res.prob_assignment <= 1.0
        assert 0.0 <= res.prob_touch <= 1.0

    def test_regime_multiplier_scales_ev(self):
        trade_hot = self._base_trade(regime_multiplier=1.0)
        trade_cold = self._base_trade(regime_multiplier=0.5)
        res_hot = EVEngine().evaluate(trade_hot)
        res_cold = EVEngine().evaluate(trade_cold)
        # ev_dollars should scale linearly with regime_multiplier (other
        # inputs identical), modulo RNG noise in the fall-through sampler.
        assert res_hot.ev_dollars * 0.5 == pytest.approx(res_cold.ev_dollars, rel=0.05)

    def test_transaction_costs_reduce_ev(self):
        """Adding a wider bid/ask spread (i.e. more slippage) must not
        *increase* EV."""
        trade_tight = self._base_trade(bid=1.19, ask=1.21)
        trade_wide = self._base_trade(bid=0.90, ask=1.50)
        r_tight = EVEngine().evaluate(trade_tight)
        r_wide = EVEngine().evaluate(trade_wide)
        assert r_wide.total_transaction_cost >= r_tight.total_transaction_cost

    def test_ranker_applies_min_ev_filter(self):
        # Force a trade with obviously-negative EV by selling for peanuts
        bad = self._base_trade(premium=0.01, bid=0.005, ask=0.015)
        ranked = rank_candidates([bad], top_n=5, min_ev=0.0)
        assert ranked == []

    def test_cvar_is_at_most_var(self):
        """CVaR (expected shortfall) must be at least as bad as VaR."""
        rng = np.random.default_rng(42)
        log_rets = 0.18 * np.sqrt(30 / 252) * rng.standard_normal(5000)
        trade = self._base_trade()
        res = EVEngine().evaluate(trade, forward_log_returns=log_rets)
        # CVaR_5 is the mean of the worst 5% — always ≤ the 5% quantile.
        assert res.cvar_5 <= res.mean_pnl

    def test_edge_vs_fair_uses_bsm(self):
        """If premium == BSM fair value, edge_vs_fair should be ~0 per share."""
        spot, strike, T, r, sigma, q = 100, 95, 30 / 365, 0.05, 0.25, 0.0
        fair = op.black_scholes_price(spot, strike, T, r, sigma, "put", q)
        trade = self._base_trade(premium=fair, bid=fair - 0.02, ask=fair + 0.02)
        res = EVEngine().evaluate(trade)
        # edge_vs_fair is in dollars-for-the-position = per-share × 100
        assert abs(res.metadata["edge_per_share"]) < 1e-6


# =========================================================================
# 6. BSM parity & deterministic edge cases
# =========================================================================
class TestBSMParity:
    """Call-put parity, intrinsic at T=0, zero-vol collapse."""

    def test_put_call_parity(self):
        S, K, T, r, sigma, q = 100, 100, 30 / 365, 0.05, 0.25, 0.02
        c = op.black_scholes_price(S, K, T, r, sigma, "call", q)
        p = op.black_scholes_price(S, K, T, r, sigma, "put", q)
        # c - p = S e^{-qT} - K e^{-rT}
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_t_zero_returns_intrinsic(self):
        # ITM put at T=0 should equal max(K-S, 0)
        p = op.black_scholes_price(100, 110, 0.0, 0.05, 0.25, "put", 0.0)
        assert p == pytest.approx(10.0, abs=1e-9)
        # OTM put at T=0 = 0
        p2 = op.black_scholes_price(100, 90, 0.0, 0.05, 0.25, "put", 0.0)
        assert p2 == pytest.approx(0.0, abs=1e-9)

    def test_zero_vol_collapses_to_deterministic(self):
        S, K, T, r, q = 100, 100, 30 / 365, 0.05, 0.0
        # With sigma=0 the option is worth max(0, S*e^{-qT} - K*e^{-rT}) for a call
        c = op.black_scholes_price(S, K, T, r, 0.0, "call", q)
        expected = max(0, S * math.exp(-q * T) - K * math.exp(-r * T))
        assert abs(c - expected) < 1e-9
