"""Tests for the Phase-3 EV-calibration stats (scripts/ibkr_ev_calibration.py).

The per-trade engine evaluation needs the full data layer, so these cover the
pure, deterministic calibration math — Wilson CIs and the reliability /
Brier / ECE computation — on synthetic inputs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_ev_calibration", _REPO / "scripts" / "ibkr_ev_calibration.py"
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


def test_wilson_empty():
    lo, hi = cal.wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_brackets_point_estimate():
    lo, hi = cal.wilson(8, 10)  # p_hat = 0.8
    assert 0.0 < lo < 0.8 < hi < 1.0
    # known Wilson 95% CI for 8/10 ≈ [0.490, 0.943]
    assert lo == pytest.approx(0.490, abs=0.01)
    assert hi == pytest.approx(0.943, abs=0.01)


def test_wilson_all_success_non_degenerate_lower():
    lo, hi = cal.wilson(10, 10)
    # Wilson never gives a degenerate [1, 1]: the lower bound is well below 1
    # (≈0.72 for 10/10); the upper bound rounds to ~1.0.
    assert 0.6 < lo < 1.0
    assert hi <= 1.0 + 1e-6


def test_reliability_bins_brier_ece():
    preds = [0.05, 0.75, 0.85, 0.95, 0.95]
    outcomes = [0, 1, 1, 1, 0]
    rows, brier, ece = cal.reliability(preds, outcomes, n_bins=10)
    # Brier = mean of squared errors
    assert brier == pytest.approx(0.1985, abs=1e-4)
    # ECE = sum_bin (n/N) * |mean_pred - obs|
    assert ece == pytest.approx(0.27, abs=1e-4)
    # top bin [0.9,1.0] holds the two 0.95 preds, observed 1/2
    top = [r for r in rows if r["bin"] == "[0.9,1.0)"][0]
    assert top["n"] == 2
    assert top["mean_pred"] == pytest.approx(0.95)
    assert top["obs"] == pytest.approx(0.5)


def test_reliability_perfect_calibration_low_ece():
    # 100 preds at 0.7 with exactly 70% outcomes -> ECE ~ 0
    preds = [0.7] * 100
    outcomes = [1] * 70 + [0] * 30
    _rows, _brier, ece = cal.reliability(preds, outcomes, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_load_universe_has_sp500_names():
    u = cal.load_universe()
    assert "AAPL" in u and "NVDA" in u
    assert "CLS" not in u  # Celestica is not S&P 500 (out of mandate)
    assert len(u) > 400
