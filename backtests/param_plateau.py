"""Parameter-plateau sweep (V3) — are the hand-set constants plateaus or peaks?

`docs/VALIDATION_PHASE_PLAN.md` section 6 is the pre-registered design this
module implements.  The Phase-0 inventory (`docs/PARAMETER_OOS.md` section 1)
lists every static constant chosen with full-history visibility; the sweep
question — "does the shipped value sit on a flat shelf (a design choice) or
on a narrow optimum (a fitted artifact)?" — applies to the constants that
actually act on the ranked path on this provider:

* **F4 vol-widening** (`threshold` 1.30 / `cap` 1.15) — swept with one
  engine pass per value via the section-6-sanctioned context-manager patch
  of the `engine.forward_distribution` module attributes (wheel_runner
  imports them function-locally, so the patch takes effect with zero trio
  changes).  Emits V1's TAIL_TABLE schema so the V1 statistics run
  unchanged; `tail_widening_factor` in the table gives the fire rate.
* **R11 cutoffs** (`vix` 25.0 / `top-bin prob` 0.90) — swept OFFLINE on the
  captured V1 tail tables (`vix_entry`, `prob_profit`, realized outcomes are
  all columns).
* **POT-GPD / heavy-tail / bootstrap constants** — ACTIVATION-GATED: the
  diagnostic here measures how often that machinery even runs (the GPD
  needs >= 200 scenarios; ~99% of Bloomberg rows ride the N~35
  non-overlapping tier).  Below the pre-registered 2% activation floor the
  sweeps are recorded NOT POWERED on this provider, not silently passed.

Verdicts per swept axis (pre-registered, section 6.3): PLATEAU / CLIFF /
PEAK (the fitted-artifact falsifier) / DOMINATED (a swept value beats
shipped significantly while the rank-quality guard holds — a finding, not a
pass).  Primary metric: cvar_5 breach rate in the elevated+crisis entry-VIX
strata (what F4 exists to protect); guard: top-tier per-date rank rho.

Scope / invariants (CLAUDE.md section 2): measurement-only.  Read-only
ranker calls, rail pinned off, patches scoped to context managers, nothing
feeds back, no production default changes; a "better" swept value is
reported as a finding, never shipped from here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shipped values + pre-registered sweep grids (plan doc section 6)
# ---------------------------------------------------------------------------

F4_SHIPPED: dict[str, float] = {"threshold": 1.30, "slope": 0.20, "max_widening": 1.15}
F4_THRESHOLD_GRID: tuple[float, ...] = (1.10, 1.20, 1.30, 1.40, 1.50)
F4_CAP_GRID: tuple[float, ...] = (1.00, 1.075, 1.15, 1.225, 1.30)
#: axis name -> (patched kwarg, grid, shipped value)
F4_AXES: dict[str, tuple[str, tuple[float, ...], float]] = {
    "threshold": ("threshold", F4_THRESHOLD_GRID, 1.30),
    "cap": ("max_widening", F4_CAP_GRID, 1.15),
}

R11_SHIPPED: tuple[float, float] = (25.0, 0.90)
R11_VIX_GRID: tuple[float, ...] = (20.0, 22.5, 25.0, 27.5, 30.0)
R11_PROB_GRID: tuple[float, ...] = (0.85, 0.90, 0.95)

#: Elevated+crisis entry-VIX stratum floor (the W-series calm ceiling).
PRIMARY_STRATUM_VIX_MIN = 15.0

#: Pre-registered activation floor below which the GPD/bootstrap constant
#: sweeps are recorded NOT POWERED on this provider (plan section 6.0).
ACTIVATION_FLOOR = 0.02


# ---------------------------------------------------------------------------
# The F4 patch lever
# ---------------------------------------------------------------------------


@contextmanager
def f4_patched(**overrides: float):
    """Scope-limited override of the F4 widening constants.

    Accepts any of ``threshold`` / ``slope`` / ``max_widening`` and pins them
    on BOTH `realized_vol_widening_factor` and
    `realized_vol_widened_log_returns` (wheel_runner calls the two
    separately).  wheel_runner never passes these kwargs itself, so the
    override is total on the rank path.
    """
    bad = set(overrides) - {"threshold", "slope", "max_widening"}
    if bad:
        raise ValueError(f"unknown F4 override(s): {sorted(bad)}")
    import engine.forward_distribution as fdist

    orig_factor = fdist.realized_vol_widening_factor
    orig_widen = fdist.realized_vol_widened_log_returns

    def _factor(ohlcv: pd.DataFrame, as_of: Any = None, **kw: Any) -> float:
        return orig_factor(ohlcv, as_of=as_of, **{**kw, **overrides})

    def _widen(log_returns: np.ndarray, ohlcv: pd.DataFrame, as_of: Any = None, **kw: Any):
        return orig_widen(log_returns, ohlcv, as_of=as_of, **{**kw, **overrides})

    fdist.realized_vol_widening_factor = _factor
    fdist.realized_vol_widened_log_returns = _widen
    try:
        yield
    finally:
        fdist.realized_vol_widening_factor = orig_factor
        fdist.realized_vol_widened_log_returns = orig_widen


def build_f4_sweep_table(
    *,
    axis: str,
    value: float,
    tickers: Sequence[str],
    sample_dates: Sequence[date],
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
    progress: bool = True,
) -> pd.DataFrame:
    """One engine pass with a single F4 axis pinned to ``value``.

    The shipped value produces the unpatched baseline by construction
    (the override equals the default), so one baseline table serves both
    axes.
    """
    from backtests.tail_exceedance import build_tail_table

    param, grid, _shipped = F4_AXES[axis]
    if value not in grid:
        raise ValueError(f"{value} not in the pre-registered {axis} grid {grid}")
    with f4_patched(**{param: value}):
        return build_tail_table(
            tickers=tickers,
            sample_dates=sample_dates,
            dte_target=dte_target,
            delta_target=delta_target,
            top_n=top_n,
            progress=progress,
        )


# ---------------------------------------------------------------------------
# Activation diagnostic (pure half — the driver supplies ranked frames)
# ---------------------------------------------------------------------------


def activation_from_frames(frames: Sequence[pd.DataFrame]) -> dict[str, Any]:
    """GPD / bootstrap activation rates from ranked diagnostic frames.

    Decides the section-6.0 gates: below ``ACTIVATION_FLOOR`` the POT /
    heavy-tail / block-length sweeps are NOT POWERED on this provider.
    """
    live = [f for f in frames if f is not None and len(f)]
    if not live:
        return {"n_rows": 0, "verdict": "NO_DATA"}
    df = pd.concat(live, ignore_index=True)
    n = len(df)

    def _num(name: str) -> np.ndarray | None:
        if name not in df.columns:
            return None
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    xi = _num("tail_xi")
    cvar99 = _num("cvar_99_evt")
    n_scen = _num("n_scenarios")
    heavy_rate = (
        float(df["heavy_tail"].fillna(False).astype(bool).mean())
        if "heavy_tail" in df.columns
        else float("nan")
    )
    gpd_rate = float(np.isfinite(xi).mean()) if xi is not None else float("nan")
    out: dict[str, Any] = {
        "n_rows": n,
        "n_dates": int(df["date"].nunique()) if "date" in df.columns else None,
        "gpd_fit_rate": gpd_rate,
        "heavy_tail_rate": heavy_rate,
        "cvar99_finite_rate": (
            float(np.isfinite(cvar99).mean()) if cvar99 is not None else float("nan")
        ),
        "n_scenarios_ge_200_rate": (
            float((n_scen >= 200).mean()) if n_scen is not None else float("nan")
        ),
        "distribution_source_mix": (
            df["distribution_source"].value_counts(normalize=True).round(4).to_dict()
            if "distribution_source" in df.columns
            else {}
        ),
        "activation_floor": ACTIVATION_FLOOR,
    }
    if np.isfinite(gpd_rate):
        out["verdict"] = "POWERED" if gpd_rate >= ACTIVATION_FLOOR else "NOT_POWERED"
    else:
        out["verdict"] = "NO_DIAGNOSTIC_COLUMNS"
    return out


# ---------------------------------------------------------------------------
# V3-a — R11 cutoff sweep (offline, pure)
# ---------------------------------------------------------------------------


def r11_sweep(
    table: pd.DataFrame,
    *,
    vix_grid: Sequence[float] = R11_VIX_GRID,
    prob_grid: Sequence[float] = R11_PROB_GRID,
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Sweep the R11 cutoffs over a captured tail table.

    For each (vix_threshold, prob_threshold) cell, the R11-flagged region is
    ``vix_entry > T AND prob_profit > P``; its comparison set is the
    UNFLAGGED remainder of the same prob bin (``prob_profit > P`` at calmer
    entry VIX).  R11's value claim is that the flagged region realizes
    materially worse tails — ``lift`` is flagged breach rate over unflagged
    top-bin breach rate.  PIT-clean: both conditioning columns are
    entry-time quantities.
    """
    from backtests.tail_exceedance import _resolved, date_clustered_rate_ci

    t = _resolved(table)
    t = t[np.isfinite(t["cvar_5"].to_numpy(dtype=float))]
    t = t[np.isfinite(t["vix_entry"].to_numpy(dtype=float))]
    realized = t["realized_pnl"].to_numpy(dtype=float)
    cvar = t["cvar_5"].to_numpy(dtype=float)
    breach = realized < cvar
    win = realized > 0.0
    vix = t["vix_entry"].to_numpy(dtype=float)
    prob = t["prob_profit"].to_numpy(dtype=float)
    dates = t["date"].to_numpy()

    cells: list[dict[str, Any]] = []
    for T in vix_grid:
        for P in prob_grid:
            flagged = (vix > T) & (prob > P)
            rest = (prob > P) & ~flagged
            nf, nr = int(flagged.sum()), int(rest.sum())
            cell: dict[str, Any] = {
                "vix_threshold": float(T),
                "prob_threshold": float(P),
                "shipped": (T, P) == R11_SHIPPED,
                "n_flagged": nf,
                "n_unflagged_topbin": nr,
            }
            if nf:
                cell["flagged_breach_rate"] = float(breach[flagged].mean())
                cell["flagged_mean_realized"] = float(np.mean(realized[flagged]))
                cell["flagged_breach_ci"] = date_clustered_rate_ci(
                    dates[flagged], breach[flagged], n_boot=n_boot, seed=seed
                )
                # The D23 rationale lens: R11 exists because the top bin's
                # realized WIN RATE under-delivers its forecast after
                # elevated-vol readings — the over-confidence gap
                # (forecast - realized) is the quantity R11 claims is worse
                # in the flagged region.
                cell["flagged_win_rate"] = float(win[flagged].mean())
                cell["flagged_mean_prob"] = float(np.mean(prob[flagged]))
                cell["flagged_overconfidence_gap"] = float(
                    np.mean(prob[flagged]) - win[flagged].mean()
                )
            if nr:
                cell["unflagged_breach_rate"] = float(breach[rest].mean())
                cell["unflagged_mean_realized"] = float(np.mean(realized[rest]))
                cell["unflagged_win_rate"] = float(win[rest].mean())
                cell["unflagged_mean_prob"] = float(np.mean(prob[rest]))
                cell["unflagged_overconfidence_gap"] = float(np.mean(prob[rest]) - win[rest].mean())
            if nf and nr and breach[rest].mean() > 0:
                cell["lift"] = float(breach[flagged].mean() / breach[rest].mean())
            else:
                cell["lift"] = float("nan")
            cells.append(cell)
    return {
        "n_rows": int(len(t)),
        "vix_grid": list(vix_grid),
        "prob_grid": list(prob_grid),
        "shipped": {"vix_threshold": R11_SHIPPED[0], "prob_threshold": R11_SHIPPED[1]},
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Section 6.3 verdicts (pure)
# ---------------------------------------------------------------------------


def plateau_verdict(
    values: Sequence[float],
    rates: Sequence[float],
    ses: Sequence[float],
    *,
    shipped: float,
    guard_ok: Sequence[bool] | None = None,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """PLATEAU / CLIFF / PEAK / DOMINATED per the pre-registered 6.3 rules.

    ``rates`` is the primary metric per value (breach rate: lower better);
    ``ses`` its date-clustered SEs.  Significance between two values uses
    the combined SE ``sqrt(se_a^2 + se_b^2)`` at 2 sigma.  ``guard_ok[i]``
    marks whether value i keeps the rank-quality guard (defaults to all
    True).  DOMINATED: some value beats shipped significantly with the
    guard intact — reported as a finding; it voids PLATEAU.
    """
    order = np.argsort(np.asarray(values, dtype=float))
    v = np.asarray(values, dtype=float)[order]
    r = np.asarray(rates, dtype=float)[order]
    s = np.asarray(ses, dtype=float)[order]
    g = (
        np.asarray(list(guard_ok), dtype=bool)[order]
        if guard_ok is not None
        else np.ones(len(v), dtype=bool)
    )
    sign = 1.0 if lower_is_better else -1.0
    idx = int(np.argmin(np.abs(v - shipped)))
    if not np.isclose(v[idx], shipped):
        raise ValueError(f"shipped value {shipped} not in swept values {list(v)}")

    def _signif_worse(i: int, j: int) -> bool:
        """Is value j significantly worse than value i?"""
        se = float(np.sqrt(s[i] ** 2 + s[j] ** 2))
        if not np.isfinite(se) or se <= 0:
            return False
        return sign * (r[j] - r[i]) > 2.0 * se

    neighbors = [k for k in (idx - 1, idx + 1) if 0 <= k < len(v)]
    worse = {k: _signif_worse(idx, k) for k in neighbors}
    n_worse = sum(worse.values())
    dominated_by = [
        float(v[k])
        for k in range(len(v))
        if k != idx and g[k] and _signif_worse(k, idx)  # shipped signif worse than k
    ]
    if len(neighbors) == 2 and n_worse == 2:
        verdict = "PEAK"
    elif dominated_by:
        verdict = "DOMINATED"
    elif n_worse == 1:
        verdict = "CLIFF"
    else:
        verdict = "PLATEAU"
    return {
        "verdict": verdict,
        "shipped": float(shipped),
        "values": [float(x) for x in v],
        "rates": [float(x) for x in r],
        "ses": [float(x) for x in s],
        "guard_ok": [bool(x) for x in g],
        "neighbors_signif_worse": {str(float(v[k])): bool(w) for k, w in worse.items()},
        "dominated_by": dominated_by,
    }


# ---------------------------------------------------------------------------
# V3-b — F4 axis report
# ---------------------------------------------------------------------------


def _breach_stats(
    table: pd.DataFrame, *, vix_min: float | None, n_boot: int, seed: int
) -> dict[str, Any]:
    from backtests.tail_exceedance import _resolved, date_clustered_rate_ci

    t = _resolved(table)
    t = t[np.isfinite(t["cvar_5"].to_numpy(dtype=float))]
    if vix_min is not None:
        t = t[pd.to_numeric(t["vix_entry"], errors="coerce").to_numpy(dtype=float) > vix_min]
    n = len(t)
    if n == 0:
        return {"n": 0, "rate": float("nan"), "se": float("nan")}
    breach = t["realized_pnl"].to_numpy(dtype=float) < t["cvar_5"].to_numpy(dtype=float)
    ci = date_clustered_rate_ci(t["date"].to_numpy(), breach, n_boot=n_boot, seed=seed)
    se = (ci["ci_high"] - ci["ci_low"]) / 3.92 if np.isfinite(ci["ci_high"]) else float("nan")
    return {"n": n, "rate": float(breach.mean()), "se": float(se), "ci": ci}


def f4_axis_report(
    axis: str,
    tables: dict[float, pd.DataFrame],
    *,
    n_boot: int = 2000,
    seed: int = 12345,
    block_len: int = 3,
    guard_tier: int = 15,
) -> dict[str, Any]:
    """Per-value metrics + the section-6.3 verdict for one F4 axis.

    Primary: elevated+crisis (entry VIX > 15) cvar_5 breach rate.  Guard:
    top-``guard_tier`` per-date rho must stay at or above the SHIPPED
    value's block-CI lower bound.  ``block_len`` scales the moving-block CI
    to the grid cadence (every-10-bday -> 3).
    """
    from backtests.parameter_oos import (
        cluster_bootstrap_ci,
        per_date_cross_sectional_rho,
        restrict_top_n_per_date,
    )
    from backtests.tail_exceedance import quantile_coverage_report

    _param, _grid, shipped = F4_AXES[axis]
    per_value: dict[float, dict[str, Any]] = {}
    for value, table in sorted(tables.items()):
        entry: dict[str, Any] = {
            "primary_elev_crisis": _breach_stats(
                table, vix_min=PRIMARY_STRATUM_VIX_MIN, n_boot=n_boot, seed=seed
            ),
            "pooled": _breach_stats(table, vix_min=None, n_boot=n_boot, seed=seed),
            "p25": {
                k: quantile_coverage_report(table, quantile_col="pnl_p25", nominal=0.25)[k]
                for k in ("n", "verdict")
            },
            "fire_rate": float(
                (
                    pd.to_numeric(table["tail_widening_factor"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    > 1.0 + 1e-9
                ).mean()
            ),
        }
        sub = restrict_top_n_per_date(table, guard_tier, signal_col="ev_dollars")
        entry["guard_rho"] = {
            "xsec": per_date_cross_sectional_rho(sub, "ev_dollars"),
            "ci_block": cluster_bootstrap_ci(
                sub,
                stat="cross_sectional",
                signal_col="ev_dollars",
                n_boot=max(200, n_boot // 4),
                seed=seed,
                block_len=block_len,
            ),
        }
        per_value[float(value)] = entry

    ship_ci_low = per_value[shipped]["guard_rho"]["ci_block"]["ci95"][0]
    values = sorted(per_value)
    guard_ok = [
        bool(per_value[val]["guard_rho"]["xsec"]["mean_rho"] >= ship_ci_low) for val in values
    ]
    verdict = plateau_verdict(
        values,
        [per_value[val]["primary_elev_crisis"]["rate"] for val in values],
        [per_value[val]["primary_elev_crisis"]["se"] for val in values],
        shipped=shipped,
        guard_ok=guard_ok,
    )
    return {
        "axis": axis,
        "shipped": shipped,
        "guard_tier": guard_tier,
        "guard_shipped_ci_low": float(ship_ci_low),
        "per_value": {str(val): per_value[val] for val in values},
        "verdict": verdict,
    }
