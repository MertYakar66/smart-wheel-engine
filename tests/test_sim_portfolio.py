"""Tests for :mod:`engine.sim_portfolio` — the distributional forward
simulated-portfolio reporting overlay.

Covers the four properties that make the overlay trustworthy:

* **Reconciliation** — when the MC horizon equals the history length, the
  bootstrap median terminal NAV tracks the deterministic realized final NAV.
* **Distribution well-formedness** — the equity fan and terminal / drawdown
  bands are monotone in quantile, and the whole thing is deterministic under
  the canonical seed.
* **Correlation-to-1 tail** — the copula stress spoke worsens the book's tail
  CVaR relative to its realized correlation, and reports a verdict.
* **§2 safety** — the module is off the decision path: it never imports the
  EV trio, and its outputs are explicitly flagged reporting-only /
  feeds_ev=False.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from engine.sim_portfolio import (
    CANONICAL_SEED,
    SIM_CAVEATS,
    build_sim_report,
    monte_carlo_bands,
    portfolio_correlation_tail,
    reconcile_with_backtest,
    strategy_returns_from_equity_curve,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "engine" / "sim_portfolio.py"


def _synthetic_curve(n: int = 240, mu: float = 0.0006, sigma: float = 0.010, seed: int = 7):
    """A synthetic tracker equity_curve + its realized final NAV."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    cap = 200_000.0
    vals = cap * np.cumprod(1 + rets)
    curve = [{"date": None, "portfolio_value": cap, "cash": 0.0, "num_positions": 1}]
    curve += [
        {"date": None, "portfolio_value": float(v), "cash": 0.0, "num_positions": 3} for v in vals
    ]
    return curve, cap, float(vals[-1])


# ---------------------------------------------------------------------------
# strategy-return derivation
# ---------------------------------------------------------------------------


def test_strategy_returns_length_and_values():
    curve, cap, _ = _synthetic_curve(n=100)
    rets = strategy_returns_from_equity_curve(curve)
    assert len(rets) == 100  # 101 marks (cap + 100) -> 100 diffs
    assert np.all(np.isfinite(rets))


def test_strategy_returns_empty_and_short():
    assert len(strategy_returns_from_equity_curve([])) == 0
    assert len(strategy_returns_from_equity_curve([{"portfolio_value": 1.0}])) == 0
    # Missing column -> empty, not a crash.
    assert len(strategy_returns_from_equity_curve([{"cash": 1.0}, {"cash": 2.0}])) == 0


# ---------------------------------------------------------------------------
# Monte Carlo bands
# ---------------------------------------------------------------------------


def test_bands_monotone_in_quantile():
    curve, cap, _ = _synthetic_curve()
    rets = strategy_returns_from_equity_curve(curve)
    bands = monte_carlo_bands(rets, initial_capital=cap, n_simulations=3000, seed=CANONICAL_SEED)
    # Equity fan monotone every day.
    b = bands.equity_bands
    for i in range(len(b["p5"])):
        assert b["p5"][i] <= b["p25"][i] <= b["p50"][i] <= b["p75"][i] <= b["p95"][i]
    # Terminal-return band monotone.
    tq = bands.terminal_return_quantiles
    assert tq["p5"] < tq["p50"] < tq["p95"]
    # Drawdown magnitudes monotone (p95 == worst).
    dq = bands.max_drawdown_quantiles
    assert dq["p5"] <= dq["p50"] <= dq["p95"]
    # Fan starts at capital and has n_days+1 points.
    assert b["p50"][0] == pytest.approx(cap)
    assert len(b["p50"]) == bands.n_days + 1


def test_bands_deterministic_under_canonical_seed():
    curve, cap, _ = _synthetic_curve()
    rets = strategy_returns_from_equity_curve(curve)
    a = monte_carlo_bands(rets, initial_capital=cap, n_simulations=3000, seed=42)
    b = monte_carlo_bands(rets, initial_capital=cap, n_simulations=3000, seed=42)
    assert a.median_return == b.median_return
    assert a.cvar_5 == b.cvar_5
    assert a.equity_bands["p50"] == b.equity_bands["p50"]
    assert a.terminal_return_quantiles == b.terminal_return_quantiles


def test_bands_raise_on_series_shorter_than_block():
    rets = np.full(10, 0.001)  # shorter than default block_size=21
    with pytest.raises(ValueError, match="too short for block bootstrap"):
        monte_carlo_bands(rets, initial_capital=100_000.0)


def test_bands_to_dict_is_json_safe_and_labeled_model():
    curve, cap, _ = _synthetic_curve()
    rets = strategy_returns_from_equity_curve(curve)
    bands = monte_carlo_bands(rets, initial_capital=cap, n_simulations=1000)
    d = bands.to_dict()
    assert d["kind"] == "model"
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_median_tracks_realized_nav():
    curve, cap, realized = _synthetic_curve(n=240)
    rets = strategy_returns_from_equity_curve(curve)
    bands = monte_carlo_bands(rets, initial_capital=cap, n_simulations=5000, seed=CANONICAL_SEED)
    rec = reconcile_with_backtest(bands, realized_final_nav=realized, initial_capital=cap)
    # Median NAV should track realized within tolerance and the realized path
    # should be a plausible draw (inside the p5-p95 terminal band).
    assert rec["in_band"] is True
    assert rec["median_close"] is True
    assert rec["reconciled"] is True
    assert rec["median_abs_pct_gap"] < 10.0


def test_reconcile_flags_out_of_band_realized():
    curve, cap, _ = _synthetic_curve(n=240)
    rets = strategy_returns_from_equity_curve(curve)
    bands = monte_carlo_bands(rets, initial_capital=cap, n_simulations=3000, seed=CANONICAL_SEED)
    # An absurd realized NAV (3x capital) can't be inside the band nor median-close.
    rec = reconcile_with_backtest(bands, realized_final_nav=cap * 3, initial_capital=cap)
    assert rec["reconciled"] is False


def test_reconcile_bases_on_first_mark_not_nominal_capital():
    """The real tracker only marks on position-holding days, so the first
    equity mark != nominal initial_capital generically. The reconciliation must
    base on the FIRST MARK — otherwise the gap measures the nominal->first-mark
    drift, not bootstrap fidelity, and a faithful projection false-fails.
    """
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0005, 0.009, 240)
    first_mark = 180_000.0  # deliberately far from nominal capital (200k)
    vals = first_mark * np.cumprod(1 + rets)
    curve = [{"date": None, "portfolio_value": float(first_mark), "cash": 0.0, "num_positions": 2}]
    curve += [
        {"date": None, "portfolio_value": float(v), "cash": 0.0, "num_positions": 3} for v in vals
    ]
    report = build_sim_report(
        equity_curve=curve,
        initial_capital=200_000.0,  # nominal, 11% above the first mark
        n_simulations=4000,
        seed=CANONICAL_SEED,
    )
    em = report["engine_measured"]
    assert em["book_base_nav"] == pytest.approx(first_mark)
    assert em["initial_capital"] == 200_000.0
    # Faithful projection reconciles despite the 11% nominal/first-mark gap
    # (had the base been nominal capital, the gap would be ~11% -> false-fail).
    assert report["reconciliation"]["reconciled"] is True
    assert report["reconciliation"]["median_abs_pct_gap"] < 10.0


# ---------------------------------------------------------------------------
# Correlation tail (copula) — reporting/stress overlay only
# ---------------------------------------------------------------------------


def _common_factor_returns(names, n=250, load=0.7, seed=11):
    """Correlated returns via a shared factor (positive cross-name corr)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, n)
    return {t: load * factor + np.sqrt(1 - load**2) * rng.normal(0, 0.01, n) for t in names}


def test_correlation_tail_corr_to_1_worsens_tail():
    names = ["A", "B", "C", "D"]
    pnr = _common_factor_returns(names, load=0.5)
    weights = dict.fromkeys(names, 1.0)
    ct = portfolio_correlation_tail(pnr, weights, n_samples=5000, seed=CANONICAL_SEED)
    # Corr->1 spoke must worsen the coordinated tail vs realized correlation.
    assert ct.stress_corr_to_1["t_cvar"] > ct.empirical["t_cvar"]
    assert ct.stress_vs_empirical["t_cvar_multiple"] > 1.0
    assert ct.empirical["verdict"] in {
        "negligible_tail_dependence",
        "mild_tail_dependence",
        "material_tail_dependence",
        "critical_tail_dependence",
    }


def test_correlation_tail_reports_positive_amplification():
    names = ["A", "B", "C", "D", "E"]
    pnr = _common_factor_returns(names, load=0.6)
    weights = dict.fromkeys(names, 1.0)
    ct = portfolio_correlation_tail(pnr, weights, n_samples=8000, seed=CANONICAL_SEED)
    # t-copula tail amplification (t vs Gaussian) is finite and non-degenerate.
    amp = ct.empirical["tail_amplification"]
    assert np.isfinite(amp) and amp > 0.0


def test_correlation_tail_interior_nan_uses_listwise_deletion():
    """A single interior NaN in one date-aligned series must be handled by
    LISTWISE deletion (drop that row across all names), NOT per-name compaction
    (which shifts one column's dates and spuriously halves its correlations).
    """
    names = ["A", "B", "C"]
    clean = _common_factor_returns(names, n=250, load=0.8, seed=5)
    dirty = {t: v.copy() for t, v in clean.items()}
    dirty["A"][100] = np.nan
    ct = portfolio_correlation_tail(dirty, dict.fromkeys(names, 1.0), n_samples=3000)
    # Reference: drop index 100 across ALL names (correct listwise deletion).
    ref_mat = np.column_stack([np.delete(clean[t], 100) for t in names])
    ref_corr = np.corrcoef(ref_mat, rowvar=False)
    ref_mean_abs = float(np.mean(np.abs(ref_corr[np.triu_indices(3, 1)])))
    assert ct.n_obs == 249  # exactly one joint row dropped
    assert ct.mean_abs_correlation == pytest.approx(ref_mean_abs, abs=0.02)
    # And it must NOT collapse toward the misaligned (understated) value.
    assert ct.mean_abs_correlation > 0.5


def test_correlation_tail_skips_when_no_joint_finite_rows():
    # Two names whose finite observations never overlap -> no joint rows.
    a = np.array([np.nan, 0.01, np.nan, 0.02, np.nan])
    b = np.array([0.01, np.nan, 0.02, np.nan, 0.03])
    ct = portfolio_correlation_tail({"A": a, "B": b}, {"A": 1.0, "B": 1.0}, n_samples=1000)
    assert ct.empirical.get("skipped") is True
    assert ct.empirical.get("reason") == "fewer_than_2_joint_finite_observations"


def test_correlation_tail_skips_single_name():
    ct = portfolio_correlation_tail(
        {"A": np.random.default_rng(0).normal(0, 0.01, 100)}, {"A": 1.0}
    )
    assert ct.empirical.get("skipped") is True
    assert ct.stress_corr_to_1.get("skipped") is True


def test_correlation_tail_flagged_reporting_only():
    names = ["A", "B", "C"]
    pnr = _common_factor_returns(names)
    ct = portfolio_correlation_tail(pnr, dict.fromkeys(names, 1.0), n_samples=2000)
    d = ct.to_dict()
    assert d["reporting_only"] is True
    assert d["feeds_ev"] is False
    assert d["kind"] == "model"


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


def test_build_sim_report_shape_and_labels():
    curve, cap, realized = _synthetic_curve(n=240)
    names = ["A", "B", "C"]
    pnr = _common_factor_returns(names)
    report = build_sim_report(
        equity_curve=curve,
        initial_capital=cap,
        realized_final_nav=realized,
        per_name_returns=pnr,
        weights=dict.fromkeys(names, 1.0),
        n_simulations=2000,
        seed=CANONICAL_SEED,
    )
    # model vs engine-measured split is explicit.
    assert report["model"]["kind"] == "model"
    assert report["engine_measured"]["kind"] == "engine-measured"
    # Caveats block preserves the E1/E3/E5/D19/D21 chain + bootstrap + copula.
    for key in ("E1", "E3", "E5", "D19", "D21", "bootstrap", "copula"):
        assert key in report["caveats"]
    assert report["reconciliation"]["reconciled"] is True
    assert report["correlation_tail"]["feeds_ev"] is False
    json.dumps(report)  # JSON-safe


def test_build_sim_report_omits_copula_without_inputs():
    curve, cap, realized = _synthetic_curve(n=240)
    report = build_sim_report(
        equity_curve=curve, initial_capital=cap, realized_final_nav=realized, n_simulations=1000
    )
    assert "correlation_tail" not in report


# ---------------------------------------------------------------------------
# §2 safety — the overlay is off the decision path
# ---------------------------------------------------------------------------


def test_module_does_not_import_ev_trio():
    """Structural §2 guard: sim_portfolio must never import the decision-layer
    trio (ev_engine / wheel_runner / candidate_dossier). It reuses only the
    off-path quant libraries (monte_carlo, portfolio_copula, performance_metrics).
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "engine.ev_engine",
        "engine.wheel_runner",
        "engine.candidate_dossier",
        ".ev_engine",
        ".wheel_runner",
        ".candidate_dossier",
    }
    assert not (imported & forbidden), f"sim_portfolio imports the EV trio: {imported & forbidden}"


def test_caveats_constant_is_stable():
    # The caveat chain is contractual — guard against silent removal.
    assert set(SIM_CAVEATS) >= {"E1", "E3", "E5", "D19", "D21", "bootstrap", "copula"}
