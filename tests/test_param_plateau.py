"""Tests for the V3 parameter-plateau sweep harness (backtests/param_plateau.py).

Fast lane, engine-free except the F4 patch tests (which exercise the real
`engine.forward_distribution` functions on synthetic frames — no data
files): the patch lever, the R11 offline sweep math, the section-6.3
verdict logic, the activation gate, and the F4 axis-report shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests import param_plateau as pp

# ---------------------------------------------------------------------------
# The F4 patch lever
# ---------------------------------------------------------------------------


def _spiky_ohlcv(n_days: int = 400, seed: int = 5) -> pd.DataFrame:
    """Synthetic series whose last 30 returns have sample std pinned to
    exactly 0.0125 against a 0.010 baseline, so rv30/rv252 lands ~1.2 —
    inside (1.10, 1.30): fires at a lowered threshold, not at the shipped
    one. Blocks are std-normalized because raw draws at n=30 carry ~13%
    sampling noise on the std, which drowned the premise."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 1.0, size=n_days - 30)
    hot = rng.normal(0.0, 1.0, size=30)
    calm = (calm - calm.mean()) / calm.std() * 0.010
    hot = (hot - hot.mean()) / hot.std() * 0.0125
    close = 100.0 * np.exp(np.cumsum(np.concatenate([calm, hot])))
    return pd.DataFrame({"close": close}, index=pd.bdate_range("2023-01-02", periods=n_days))


def test_f4_patched_lowered_threshold_fires_and_restores():
    import engine.forward_distribution as fdist

    ohlcv = _spiky_ohlcv()
    ratio = fdist.realized_vol_ratio(ohlcv)
    assert 1.10 < ratio < 1.30  # the synthetic premise this test rests on
    orig_factor = fdist.realized_vol_widening_factor
    assert orig_factor(ohlcv) == 1.0  # shipped threshold: no fire
    with pp.f4_patched(threshold=1.10):
        patched = fdist.realized_vol_widening_factor(ohlcv)
        assert patched > 1.0  # fires under the lowered threshold
    assert fdist.realized_vol_widening_factor is orig_factor  # restored
    assert fdist.realized_vol_widening_factor(ohlcv) == 1.0


def test_f4_patched_cap_binds():
    import engine.forward_distribution as fdist

    ohlcv = _spiky_ohlcv()
    with pp.f4_patched(threshold=1.0, slope=10.0, max_widening=1.05):
        assert fdist.realized_vol_widening_factor(ohlcv) == pytest.approx(1.05)
    with pp.f4_patched(threshold=1.0, slope=10.0, max_widening=1.0):
        assert fdist.realized_vol_widening_factor(ohlcv) == pytest.approx(1.0)  # F4 OFF control


def test_f4_patched_widen_function_also_patched_and_restores_on_exception():
    import engine.forward_distribution as fdist

    ohlcv = _spiky_ohlcv()
    rets = np.random.default_rng(0).normal(0, 0.02, size=500)
    orig = fdist.realized_vol_widened_log_returns
    with pp.f4_patched(threshold=1.0, slope=10.0, max_widening=1.30):
        widened = fdist.realized_vol_widened_log_returns(rets, ohlcv)
    assert np.std(widened) > np.std(rets) * 1.2  # cap 1.30 binding via slope 10
    assert fdist.realized_vol_widened_log_returns is orig
    with pytest.raises(RuntimeError), pp.f4_patched(threshold=1.0):
        raise RuntimeError("boom")
    assert fdist.realized_vol_widened_log_returns is orig


def test_f4_patched_rejects_unknown_override():
    with pytest.raises(ValueError, match="unknown F4 override"):
        with pp.f4_patched(thresold=1.2):  # typo must fail loud, not silently no-op
            pass


def test_build_f4_sweep_table_rejects_off_grid_value():
    with pytest.raises(ValueError, match="pre-registered"):
        pp.build_f4_sweep_table(axis="threshold", value=1.33, tickers=["AAPL"], sample_dates=[])


# ---------------------------------------------------------------------------
# R11 offline sweep
# ---------------------------------------------------------------------------


def _r11_table() -> pd.DataFrame:
    """40 resolved rows: high-VIX high-prob rows breach 50%, everything else
    breaches 0% — a hand-checkable lift surface."""
    rows = []
    for i in range(40):
        high_vix = i % 4 == 0  # 10 rows
        high_prob = i % 2 == 0  # 20 rows
        flagged = high_vix and high_prob  # 10 rows (i % 4 == 0 implies i % 2 == 0)
        rows.append(
            {
                "date": f"2024-01-{(i % 8) + 1:02d}",
                "ticker": f"T{i}",
                "vix_entry": 30.0 if high_vix else 12.0,
                "prob_profit": 0.95 if high_prob else 0.60,
                "cvar_5": -1000.0,
                "realized_pnl": -2000.0 if (flagged and i % 8 == 0) else 100.0,
                "pnl_p25": -50.0,
                "pnl_p50": 50.0,
                "pnl_p75": 150.0,
                "ev_dollars": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_r11_sweep_counts_and_lift():
    out = pp.r11_sweep(_r11_table(), vix_grid=(25.0,), prob_grid=(0.90,), n_boot=100)
    assert out["n_rows"] == 40
    (cell,) = out["cells"]
    assert cell["shipped"] is True
    assert cell["n_flagged"] == 10  # vix 30 > 25 AND prob 0.95 > 0.90
    assert cell["n_unflagged_topbin"] == 10  # prob 0.95 at vix 12
    assert cell["flagged_breach_rate"] == pytest.approx(0.5)  # i in {0,8,16,24,32}
    assert cell["unflagged_breach_rate"] == pytest.approx(0.0)
    assert np.isnan(cell["lift"])  # unflagged rate 0 -> lift undefined, not inf


def test_r11_sweep_full_grid_shapes_and_shipped_mark():
    out = pp.r11_sweep(_r11_table(), n_boot=50)
    assert len(out["cells"]) == len(pp.R11_VIX_GRID) * len(pp.R11_PROB_GRID)
    shipped = [c for c in out["cells"] if c["shipped"]]
    assert len(shipped) == 1
    assert shipped[0]["vix_threshold"] == 25.0 and shipped[0]["prob_threshold"] == 0.90


def test_r11_sweep_ignores_unresolved_and_nan_vix():
    t = _r11_table()
    t.loc[0, "realized_pnl"] = np.nan
    t.loc[1, "vix_entry"] = np.nan
    out = pp.r11_sweep(t, vix_grid=(25.0,), prob_grid=(0.90,), n_boot=50)
    assert out["n_rows"] == 38


# ---------------------------------------------------------------------------
# Section 6.3 verdict logic
# ---------------------------------------------------------------------------


def test_plateau_verdict_plateau():
    out = pp.plateau_verdict(
        [1.1, 1.2, 1.3, 1.4, 1.5], [0.030, 0.031, 0.030, 0.029, 0.031], [0.01] * 5, shipped=1.3
    )
    assert out["verdict"] == "PLATEAU"


def test_plateau_verdict_peak():
    # Both neighbors dramatically worse than shipped -> the fitted-artifact signature.
    out = pp.plateau_verdict(
        [1.1, 1.2, 1.3, 1.4, 1.5], [0.20, 0.20, 0.03, 0.20, 0.20], [0.005] * 5, shipped=1.3
    )
    assert out["verdict"] == "PEAK"
    assert all(out["neighbors_signif_worse"].values())


def test_plateau_verdict_cliff_one_side():
    out = pp.plateau_verdict(
        [1.1, 1.2, 1.3, 1.4, 1.5], [0.031, 0.030, 0.030, 0.20, 0.21], [0.005] * 5, shipped=1.3
    )
    assert out["verdict"] == "CLIFF"


def test_plateau_verdict_dominated_only_with_guard():
    values = [1.1, 1.2, 1.3, 1.4, 1.5]
    rates = [0.030, 0.001, 0.030, 0.031, 0.030]  # 1.2 is much better
    ses = [0.005] * 5
    out = pp.plateau_verdict(values, rates, ses, shipped=1.3)
    assert out["verdict"] == "DOMINATED"
    assert out["dominated_by"] == [1.2]
    # Same numbers but 1.2 fails the rank guard -> not an admissible dominator.
    out2 = pp.plateau_verdict(
        values, rates, ses, shipped=1.3, guard_ok=[True, False, True, True, True]
    )
    assert out2["verdict"] == "PLATEAU"


def test_plateau_verdict_edge_shipped():
    # Shipped at the grid edge has one neighbor; PEAK impossible by construction.
    out = pp.plateau_verdict([1.0, 1.075, 1.15], [0.20, 0.03, 0.03], [0.005] * 3, shipped=1.0)
    assert out["verdict"] in {"CLIFF", "DOMINATED"}


def test_plateau_verdict_requires_shipped_in_grid():
    with pytest.raises(ValueError, match="not in swept values"):
        pp.plateau_verdict([1.0, 1.2], [0.1, 0.1], [0.01, 0.01], shipped=1.3)


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def _diag_frame(n: int, gpd_frac: float) -> pd.DataFrame:
    xi = np.where(np.arange(n) < int(n * gpd_frac), 0.25, np.nan)
    return pd.DataFrame(
        {
            "date": "2024-01-05",
            "ticker": [f"T{i}" for i in range(n)],
            "tail_xi": xi,
            "heavy_tail": np.isfinite(xi) & (xi > 0.3),
            "cvar_99_evt": np.where(np.isfinite(xi), -500.0, np.nan),
            "n_scenarios": np.where(np.isfinite(xi), 250.0, 35.0),
            "distribution_source": np.where(
                np.isfinite(xi), "empirical_overlapping", "empirical_non_overlapping"
            ),
        }
    )


def test_activation_not_powered_below_floor():
    out = pp.activation_from_frames([_diag_frame(100, 0.01)])
    assert out["gpd_fit_rate"] == pytest.approx(0.01)
    assert out["verdict"] == "NOT_POWERED"


def test_activation_powered_at_floor():
    out = pp.activation_from_frames([_diag_frame(100, 0.10)])
    assert out["verdict"] == "POWERED"
    assert out["n_scenarios_ge_200_rate"] == pytest.approx(0.10)


def test_activation_empty_and_missing_columns():
    assert pp.activation_from_frames([])["verdict"] == "NO_DATA"
    bare = pd.DataFrame({"date": ["2024-01-05"], "ticker": ["A"]})
    assert pp.activation_from_frames([bare])["verdict"] == "NO_DIAGNOSTIC_COLUMNS"


# ---------------------------------------------------------------------------
# F4 axis report shape (synthetic tables)
# ---------------------------------------------------------------------------


def _sweep_table(seed: int, fire_rate: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-03-02", periods=12, freq="10B")]
    rows = []
    for d in dates:
        for t in "ABCDEFGH":
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "ev_raw": float(rng.normal(50, 25)),
                    "ev_dollars": float(rng.normal(40, 25)),
                    "prob_profit": float(rng.uniform(0.4, 0.7)),
                    "n_scenarios": 35.0,
                    "distribution_source": "empirical_non_overlapping",
                    "hmm_regime": "normal",
                    "iv": 0.3,
                    "premium": 2.5,
                    "strike": 100.0,
                    "spot": 105.0,
                    "pnl_p25": -50.0,
                    "pnl_p50": 50.0,
                    "pnl_p75": 150.0,
                    "cvar_5": float(-rng.uniform(500, 1500)),
                    "cvar_99_evt": float("nan"),
                    "tail_widening_factor": 1.05 if rng.uniform() < fire_rate else 1.0,
                    "vix_entry": float(rng.choice([12.0, 20.0, 30.0])),
                    "expiration_date": "2099-01-01",
                    "spot_at_expiry": 105.0,
                    "realized_pnl": float(rng.normal(100, 400)),
                }
            )
    return pd.DataFrame(rows)


def test_f4_axis_report_shape_and_verdict_fields():
    tables = {v: _sweep_table(int(v * 100), fire_rate=0.1) for v in pp.F4_THRESHOLD_GRID}
    report = pp.f4_axis_report("threshold", tables, n_boot=100, block_len=3)
    assert report["axis"] == "threshold" and report["shipped"] == 1.30
    assert set(report["per_value"]) == {str(float(v)) for v in pp.F4_THRESHOLD_GRID}
    entry = report["per_value"]["1.3"]
    assert {"primary_elev_crisis", "pooled", "p25", "fire_rate", "guard_rho"} <= set(entry)
    assert 0.0 <= entry["fire_rate"] <= 1.0
    assert report["verdict"]["verdict"] in {"PLATEAU", "CLIFF", "PEAK", "DOMINATED"}
