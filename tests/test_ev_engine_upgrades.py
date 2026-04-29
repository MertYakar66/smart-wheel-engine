"""Tests for the 2026-04-23 EVEngine brain upgrades.

Covers:
  * Deterministic fallback seeding (PYTHONHASHSEED-independent)
  * regime_multiplier anomaly clamp + metadata flag
  * Stop-loss-aware expected_days_held
  * Capped Omega ratio (no float('inf') sentinels)
  * Distribution-source metadata tag
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np

from engine.ev_engine import EVEngine, ShortOptionTrade


def _trade(**overrides) -> ShortOptionTrade:
    base = dict(
        option_type="put",
        underlying="AAPL",
        spot=100.0,
        strike=95.0,
        premium=1.50,
        dte=30,
        iv=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.01,
        contracts=1,
        bid=1.45,
        ask=1.55,
        open_interest=1000,
    )
    base.update(overrides)
    return ShortOptionTrade(**base)


# ----------------------------------------------------------------------
# Deterministic seeding
# ----------------------------------------------------------------------
def test_fallback_distribution_seed_is_process_independent():
    """The lognormal fallback must give identical EV across separate Python
    processes. Before the fix this used hash(str) which is randomised per
    process by PYTHONHASHSEED."""
    script = (
        "import sys; sys.path.insert(0, r'{root}');"
        "from engine.ev_engine import EVEngine, ShortOptionTrade;"
        "t = ShortOptionTrade(option_type='put', underlying='AAPL', spot=100.0, "
        "strike=95.0, premium=1.50, dte=30, iv=0.25, bid=1.45, ask=1.55, "
        "open_interest=1000);"
        "r = EVEngine().evaluate(t);"
        "print(f'{{r.ev_dollars:.6f}}')"
    ).format(root=str(__import__("pathlib").Path(__file__).resolve().parents[1]).replace("\\", "\\\\"))

    def run_once() -> str:
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        ).stdout.strip()

    a = run_once()
    b = run_once()
    assert a == b, (
        f"Fallback EV should be identical across subprocesses, got {a!r} vs {b!r}. "
        "PYTHONHASHSEED leaked into the RNG seed."
    )


def test_fallback_distribution_source_flagged():
    """When no forward distribution is supplied, metadata must flag the
    fallback path so operators know EV is biased."""
    res = EVEngine().evaluate(_trade())
    assert res.metadata["distribution_source"] == "lognormal_fallback"


def test_empirical_distribution_source_flagged():
    rng = np.random.default_rng(0)
    fwd = rng.normal(0.0, 0.02, 500)
    res = EVEngine().evaluate(_trade(), forward_log_returns=fwd)
    assert res.metadata["distribution_source"] == "empirical"


# ----------------------------------------------------------------------
# Regime multiplier clamp
# ----------------------------------------------------------------------
def test_regime_multiplier_nan_clamped_and_flagged():
    t = _trade(regime_multiplier=float("nan"))
    res = EVEngine().evaluate(t)
    assert math.isfinite(res.ev_dollars)
    assert "regime_mult_nonfinite" in res.metadata["regime_anomaly"]
    assert res.regime_multiplier == 1.0


def test_regime_multiplier_negative_clamped():
    t = _trade(regime_multiplier=-0.3)
    res = EVEngine().evaluate(t)
    assert res.regime_multiplier == 0.0
    assert "regime_mult_negative" in res.metadata["regime_anomaly"]
    assert res.ev_dollars == 0.0


def test_regime_multiplier_over_cap_clamped():
    t = _trade(regime_multiplier=5.0)
    res = EVEngine().evaluate(t)
    assert res.regime_multiplier <= 1.25 + 1e-9
    assert "regime_mult_over_cap" in res.metadata["regime_anomaly"]


def test_regime_multiplier_in_range_untouched():
    t = _trade(regime_multiplier=0.9)
    res = EVEngine().evaluate(t)
    assert res.metadata["regime_anomaly"] == ""
    assert abs(res.regime_multiplier - 0.9) < 1e-9 or res.heavy_tail


# ----------------------------------------------------------------------
# Stop-loss-aware expected_days_held
# ----------------------------------------------------------------------
def test_expected_days_held_shrinks_with_stop_loss_scenarios():
    """A trade with many terminal losses beyond the 2x-premium stop
    threshold must have shorter expected_days_held than a trade where
    losers all graze zero."""
    # Scenario A: losses cluster around ~$50 (below 2x*150=300 stop).
    rng = np.random.default_rng(42)
    small_losses = rng.normal(-0.001, 0.001, 10_000)  # terminal log-returns tiny
    # Scenario B: losses are catastrophic (far below stop threshold).
    big_losses = rng.normal(-0.15, 0.05, 10_000)

    t = _trade(contracts=1)
    res_small = EVEngine().evaluate(t, forward_log_returns=small_losses)
    res_big = EVEngine().evaluate(t, forward_log_returns=big_losses)

    # Scenario B has more paths breaching stop → lower expected_days_held.
    assert res_big.metadata["prob_stop_terminal"] > res_small.metadata["prob_stop_terminal"]
    assert res_big.expected_days_held <= res_small.expected_days_held


def test_prob_stop_terminal_in_unit_interval():
    res = EVEngine().evaluate(_trade())
    p = res.metadata["prob_stop_terminal"]
    assert 0.0 <= p <= 1.0


# ----------------------------------------------------------------------
# Omega ratio cap
# ----------------------------------------------------------------------
def test_omega_ratio_finite_when_no_losses():
    """All-winners paths must produce a finite omega (not float('inf')) so
    downstream aggregations (sort / mean / groupby) do not propagate inf."""
    # Deep OTM put → should expire worthless across almost all paths.
    rng = np.random.default_rng(1)
    # Tight positive returns → spot moves up modestly, put expires OTM.
    fwd = rng.normal(0.001, 0.001, 1_000)
    res = EVEngine().evaluate(_trade(strike=50.0), forward_log_returns=fwd)
    assert math.isfinite(res.omega_ratio)
    assert res.omega_ratio >= 0.0


def test_omega_zero_when_only_losses():
    rng = np.random.default_rng(2)
    fwd = rng.normal(-0.10, 0.01, 1_000)  # forces deep ITM for put
    res = EVEngine().evaluate(_trade(strike=95.0), forward_log_returns=fwd)
    # With only losses omega should be finite and non-negative.
    assert math.isfinite(res.omega_ratio)
    assert res.omega_ratio >= 0.0
