"""Tail-risk exceedance validation — formal backtesting of the engine's own
risk quantiles (``pnl_p25/p50/p75``) and expected shortfall (``cvar_5``).

Why this exists
---------------
The verification record established that ``prob_profit`` is calibrated
mid-range and over-confident in the top bin (``docs/PROB_PROFIT_CALIBRATION_
2026-05-28.md``), and the 2026-06-15 trader stress test found ``cvar_5``
understating single-name blowups 3-4.5x — but as an *anecdote*, not a
statistic.  This module turns the engine's per-candidate risk outputs into
formally backtested quantities using the standard desk toolkit:

* **Kupiec (1995) proportion-of-failures** — is the violation frequency of a
  modeled quantile consistent with its nominal coverage?
* **Date-clustered significance** — trades opened on the same day share one
  market path, so pooled binomial p-values overstate confidence (the same
  overlap-inflation caveat ``docs/PRODUCTION_READINESS.md`` flags on rank-rho
  p-values).  Every coverage statistic here therefore also carries a
  cluster-bootstrap CI that resamples *dates*, not rows, mirroring
  ``backtests.parameter_oos.cluster_bootstrap_ci``.
* **Violation clustering** — a risk model whose violations arrive in bursts
  (crisis onset) is broken in a different way than one that violates too
  often.  Tested as a permutation test on the first-order autocorrelation of
  the date-level violation-rate series (a panel-honest stand-in for
  Christoffersen's (1998) independence test, which assumes one series).
* **CVaR breach bound + severity** — for any distribution, ``P(X < ES_5) <
  P(X <= VaR_5) = 5%``, so a breach rate of ``realized < cvar_5``
  significantly above 5% is unambiguous evidence the modeled tail is too
  thin; conditional severity (``realized / cvar_5`` among breaches)
  quantifies *how far* beyond the modeled expected shortfall real losses go.

Scope / invariants (CLAUDE.md section 2)
----------------------------------------
Measurement-only.  ``build_tail_table`` calls
``WheelRunner.rank_candidates_by_ev`` READ-ONLY with the option-premium rail
pinned off (same discipline as ``backtests.regression._common``); every
statistic here is computed offline in numpy on the captured table.  Nothing
feeds back into EV, a verdict, or a gate; no production default is mutated;
the decision-layer trio is untouched.

Known, documented comparison offset: realized P&L uses the locked
``_forward_replay_realized_pnl`` convention ((premium - max(0, K - S_exp)) x
100, gross of entry costs and the $5 assignment fee) while the engine's
modeled per-path P&L nets entry costs and the ITM fee.  The offset is a few
dollars per contract, *favorable to the engine* on lower-tail tests — so a
FAIL here is conservative evidence, and a marginal PASS is not proof.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capture schema
# ---------------------------------------------------------------------------

#: Columns captured per ranked row.  Superset of the parameter-OOS table:
#: adds the modeled tail block (pnl quartiles, cvar_5, cvar_99_evt,
#: tail_widening_factor), sampling-honesty fields (n_scenarios,
#: distribution_source) and the entry-date market VIX for regime strata.
TAIL_TABLE_COLUMNS = (
    "date",
    "ticker",
    "ev_raw",
    "ev_dollars",
    "prob_profit",
    "n_scenarios",
    "distribution_source",
    "hmm_regime",
    "iv",
    "premium",
    "strike",
    "spot",
    "pnl_p25",
    "pnl_p50",
    "pnl_p75",
    "cvar_5",
    "cvar_99_evt",
    "tail_widening_factor",
    "vix_entry",
    "expiration_date",
    "spot_at_expiry",
    "realized_pnl",
)

#: Entry-VIX bands — the W-series convention (tests/test_w6_topbin_netcost.py):
#: calm <= 15, elevated (15, 25], crisis > 25.
VIX_CALM_MAX = 15.0
VIX_ELEVATED_MAX = 25.0

#: Minimum observations before any verdict other than INSUFFICIENT.
MIN_N_FOR_VERDICT = 30


def vix_band(vix: float | None) -> str:
    """Map an entry VIX level to the W-series band label."""
    if vix is None or not np.isfinite(vix) or vix <= 0:
        return "unknown"
    if vix <= VIX_CALM_MAX:
        return "calm"
    if vix <= VIX_ELEVATED_MAX:
        return "elevated"
    return "crisis"


# ---------------------------------------------------------------------------
# Engine pass — expensive one-time capture (mirrors parameter_oos.build_rank_table)
# ---------------------------------------------------------------------------


def build_tail_table(
    *,
    tickers: Sequence[str],
    sample_dates: Sequence[date],
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
    contracts: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the production ranker over ``sample_dates`` capturing the modeled
    tail block, then forward-replay each row to held-to-expiry realized P&L.

    Reuses the S27/S34 forward-replay helpers so the realized-P&L convention
    is byte-identical to the locked regression snapshots and to the
    parameter-OOS fixtures.  ``min_ev_dollars=-1e9`` captures the full EV
    range — quantile calibration needs the losers the trade filter would
    drop.  Deterministic (seeded HMM, rail pinned off).
    """
    import time

    from backtests.regression._common import (
        _forward_replay_realized_pnl,
        _next_business_day,
        _option_premium_rail_pinned_off,
        _spot_on_or_after,
    )
    from engine.wheel_runner import WheelRunner

    with _option_premium_rail_pinned_off():
        runner = WheelRunner()
        conn = runner.connector
    logger.info("build_tail_table: connector=%s", type(conn).__name__)

    def _f(row: Any, key: str) -> float:
        v = row.get(key)
        try:
            return float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    rows: list[dict] = []
    n = len(sample_dates)
    t0 = time.time()
    vix_cache: dict[str, float] = {}
    for i, today in enumerate(sample_dates):
        if progress and i and i % max(1, n // 20) == 0:
            el = time.time() - t0
            eta = (n - i) / (i / el) if el > 0 else 0.0
            print(
                f"[tail_exceedance] {i:4d}/{n} ({100 * i / n:5.1f}%) "
                f"elapsed {el / 60:5.1f}m ETA {eta / 60:5.1f}m",
                flush=True,
            )
        as_of = today.isoformat()
        # Entry-date market VIX for band stratification.  Advisory: failure
        # degrades to NaN -> band "unknown" (Q3 missing-evidence semantics).
        if as_of not in vix_cache:
            try:
                v = conn.get_vix_regime(as_of).get("vix")
                vix_cache[as_of] = float(v) if v is not None else float("nan")
            except Exception:  # noqa: BLE001 — VIX is advisory, never aborts capture
                vix_cache[as_of] = float("nan")
        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = runner.rank_candidates_by_ev(
                    tickers=list(tickers),
                    dte_target=dte_target,
                    delta_target=delta_target,
                    contracts=contracts,
                    top_n=top_n,
                    min_ev_dollars=-1e9,
                    as_of=as_of,
                    include_diagnostic_fields=True,
                )
        except Exception as exc:  # noqa: BLE001 — a bad day must not abort the pass
            logger.warning("rank failed @ %s: %s", today, exc)
            continue
        if frame is None or len(frame) == 0:
            continue
        for _, r in frame.iterrows():
            rows.append(
                {
                    "date": as_of,
                    "ticker": str(r.get("ticker", "")),
                    "ev_raw": _f(r, "ev_raw"),
                    "ev_dollars": _f(r, "ev_dollars"),
                    "prob_profit": _f(r, "prob_profit"),
                    "n_scenarios": _f(r, "n_scenarios"),
                    "distribution_source": str(r.get("distribution_source", "")),
                    "hmm_regime": str(r.get("hmm_regime", "unknown")),
                    "iv": _f(r, "iv"),
                    "premium": _f(r, "premium"),
                    "strike": _f(r, "strike"),
                    "spot": _f(r, "spot"),
                    "pnl_p25": _f(r, "pnl_p25"),
                    "pnl_p50": _f(r, "pnl_p50"),
                    "pnl_p75": _f(r, "pnl_p75"),
                    "cvar_5": _f(r, "cvar_5"),
                    "cvar_99_evt": _f(r, "cvar_99_evt"),
                    "tail_widening_factor": _f(r, "tail_widening_factor"),
                    "vix_entry": vix_cache[as_of],
                    "expiration_date": expiration_default.isoformat(),
                }
            )

    table = pd.DataFrame(
        rows, columns=[c for c in TAIL_TABLE_COLUMNS if c not in ("spot_at_expiry", "realized_pnl")]
    )
    spot_cache: dict[tuple[str, str], float | None] = {}
    spots: list[float] = []
    realized: list[float] = []
    for t in table.itertuples(index=False):
        key = (t.ticker, t.expiration_date)
        if key not in spot_cache:
            spot_cache[key] = _spot_on_or_after(
                conn, t.ticker, date.fromisoformat(t.expiration_date)
            )
        spot = spot_cache[key]
        if spot is None:
            spots.append(float("nan"))
            realized.append(float("nan"))
        else:
            spots.append(spot)
            realized.append(_forward_replay_realized_pnl(t.strike, t.premium, spot))
    table["spot_at_expiry"] = spots
    table["realized_pnl"] = realized
    return table


# ---------------------------------------------------------------------------
# Statistics — pure numpy/scipy, unit-tested, no engine import
# ---------------------------------------------------------------------------


def kupiec_pof(n_obs: int, n_viol: int, coverage: float) -> dict[str, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the true violation probability equals ``coverage``.  Returns the LR
    statistic (~ chi2(1) under H0) and its p-value.  Edge cases: ``n_obs <=
    0`` or ``coverage`` outside (0, 1) -> NaN; a violation count of exactly
    ``n * coverage`` gives LR ~ 0, p ~ 1.
    """
    from scipy.stats import chi2

    if n_obs <= 0 or not (0.0 < coverage < 1.0):
        return {
            "n": float(n_obs),
            "violations": float(n_viol),
            "rate": float("nan"),
            "lr": float("nan"),
            "p_value": float("nan"),
        }
    x = int(n_viol)
    n = int(n_obs)
    pi_hat = x / n

    # Log-likelihoods; 0*log(0) := 0 by continuity.
    def _ll(p: float) -> float:
        eps_terms = 0.0
        if x > 0:
            eps_terms += x * np.log(p)
        if n - x > 0:
            eps_terms += (n - x) * np.log(1.0 - p)
        return eps_terms

    if pi_hat in (0.0, 1.0):
        lr = -2.0 * (_ll(coverage) - _ll(max(min(pi_hat, 1.0 - 1e-12), 1e-12)))
    else:
        lr = -2.0 * (_ll(coverage) - _ll(pi_hat))
    lr = max(0.0, float(lr))
    return {
        "n": float(n),
        "violations": float(x),
        "rate": pi_hat,
        "lr": lr,
        "p_value": float(chi2.sf(lr, df=1)),
    }


def date_clustered_rate_ci(
    dates: np.ndarray,
    viol: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
    ci: float = 0.95,
) -> dict[str, float]:
    """Cluster-bootstrap CI for a violation rate, resampling DATES.

    Trades opened the same day share one market path, so rows are not
    independent; resampling whole dates (with replacement) is the honest
    resampling unit — the same convention as
    ``backtests.parameter_oos.cluster_bootstrap_ci``.
    """
    dates = np.asarray(dates)
    viol = np.asarray(viol, dtype=float)
    uniq = np.unique(dates)
    n_dates = len(uniq)
    if n_dates == 0 or len(viol) == 0:
        return {
            "rate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_dates": 0.0,
            "n_boot": float(n_boot),
        }
    by_date = {d: viol[dates == d] for d in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n_dates, replace=True)
        vals = np.concatenate([by_date[d] for d in pick])
        stats[b] = float(np.mean(vals)) if len(vals) else float("nan")
    alpha = (1.0 - ci) / 2.0
    return {
        "rate": float(np.mean(viol)),
        "ci_low": float(np.nanpercentile(stats, 100 * alpha)),
        "ci_high": float(np.nanpercentile(stats, 100 * (1 - alpha))),
        "n_dates": float(n_dates),
        "n_boot": float(n_boot),
    }


def date_rate_autocorr_test(
    dates: np.ndarray,
    viol: np.ndarray,
    *,
    n_perm: int = 2000,
    seed: int = 12345,
) -> dict[str, float]:
    """Violation-clustering test on the DATE-LEVEL violation-rate series.

    Panel-honest stand-in for Christoffersen (1998) independence: compute the
    per-date mean violation rate ordered by date, take its lag-1
    autocorrelation, and get a permutation p-value by shuffling the date
    order.  Under a correctly specified, regime-aware risk model the
    violation rate has no persistence; positive autocorrelation means
    violations arrive in bursts (the crisis-onset failure mode I3-E
    documented).  One-sided: only positive clustering counts against H0.
    """
    dates = np.asarray(dates)
    viol = np.asarray(viol, dtype=float)
    order = np.argsort(dates, kind="stable")
    dates, viol = dates[order], viol[order]
    uniq, idx = np.unique(dates, return_inverse=True)
    if len(uniq) < 5:
        return {"n_dates": float(len(uniq)), "autocorr": float("nan"), "p_value": float("nan")}
    rate = np.array([viol[idx == k].mean() for k in range(len(uniq))])
    if np.nanstd(rate) < 1e-12:
        return {"n_dates": float(len(uniq)), "autocorr": 0.0, "p_value": 1.0}

    def _ac1(x: np.ndarray) -> float:
        a, b = x[:-1], x[1:]
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))

    obs = _ac1(rate)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        perm[b] = _ac1(rng.permutation(rate))
    p = float((np.sum(perm >= obs) + 1) / (n_perm + 1))
    return {"n_dates": float(len(uniq)), "autocorr": obs, "p_value": p}


def heterogeneous_coverage_z(prob: np.ndarray, win: np.ndarray) -> dict[str, float]:
    """Sum-of-Bernoulli z-test for per-trade heterogeneous win probabilities.

    Each trade carries its own ``prob_profit``; under H0 (probabilities
    correct) total wins ~ Normal(sum p, sum p(1-p)) by Lindeberg.  Two-sided.
    Complements the binned reliability tables the repo already publishes with
    a single pooled coverage statistic.
    """
    from scipy.stats import norm

    prob = np.asarray(prob, dtype=float)
    win = np.asarray(win, dtype=float)
    ok = np.isfinite(prob) & np.isfinite(win)
    prob, win = prob[ok], win[ok]
    n = len(prob)
    if n == 0:
        return {
            "n": 0.0,
            "z": float("nan"),
            "p_value": float("nan"),
            "expected_wins": float("nan"),
            "observed_wins": float("nan"),
        }
    mu = float(np.sum(prob))
    var = float(np.sum(prob * (1.0 - prob)))
    obs = float(np.sum(win))
    if var <= 0:
        return {
            "n": float(n),
            "z": float("nan"),
            "p_value": float("nan"),
            "expected_wins": mu,
            "observed_wins": obs,
        }
    z = (obs - mu) / np.sqrt(var)
    return {
        "n": float(n),
        "z": float(z),
        "p_value": float(2.0 * norm.sf(abs(z))),
        "expected_wins": mu,
        "observed_wins": obs,
    }


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _resolved(table: pd.DataFrame) -> pd.DataFrame:
    """Rows whose realized outcome resolved (expiry inside the data frontier)."""
    return table[np.isfinite(table["realized_pnl"].to_numpy(dtype=float))].copy()


def _verdict_lower_quantile(nominal: float, ci: dict[str, float], n: int) -> str:
    """PASS/WARN/FAIL ladder for a lower-tail coverage test.

    FAIL when the cluster-CI *lower* bound sits above nominal (violations
    provably too frequent -> modeled quantile too optimistic).  WARN when the
    point rate exceeds nominal but the CI straddles it.  Coverage *below*
    nominal (model too conservative) is reported but never FAILs — for a
    short-vol book, a too-fat modeled tail is the safe direction.
    """
    if n < MIN_N_FOR_VERDICT:
        return "INSUFFICIENT"
    if np.isfinite(ci["ci_low"]) and ci["ci_low"] > nominal:
        return "FAIL"
    if np.isfinite(ci["rate"]) and ci["rate"] > nominal:
        return "WARN"
    return "PASS"


def quantile_coverage_report(
    table: pd.DataFrame,
    *,
    quantile_col: str,
    nominal: float,
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Coverage test for one modeled P&L quantile column.

    A violation is ``realized_pnl < quantile`` — expected with probability
    ``nominal`` when the modeled distribution is correct.

    **Point-mass restriction (V1-a finding).** A short-put P&L distribution
    carries a point mass at max profit: with ``prob_profit >= 1 - nominal``
    the ``nominal``-quantile sits ON the win value, ``realized < quantile``
    degenerates to "any loss", and the continuous-coverage nominal no longer
    applies (the first 24t run showed p50 and p75 with byte-identical 0.245
    violation rates — vacuous, not conservative).  The test is therefore
    restricted to rows where the quantile is in the loss region:
    ``prob_profit < 1 - nominal`` — an entry-time-modeled stratum, so the
    conditioning is PIT-clean.  Excluded counts are reported so vacuity is
    visible, never silent.
    """
    t = _resolved(table)
    t = t[np.isfinite(t[quantile_col].to_numpy(dtype=float))]
    n_candidates = len(t)
    informative = t["prob_profit"].to_numpy(dtype=float) < (1.0 - nominal)
    t = t[informative]
    n = len(t)
    out: dict[str, Any] = {
        "quantile_col": quantile_col,
        "nominal": nominal,
        "n": n,
        "n_excluded_point_mass": int(n_candidates - n),
    }
    if n == 0:
        out["verdict"] = "INSUFFICIENT"
        return out
    viol = t["realized_pnl"].to_numpy(dtype=float) < t[quantile_col].to_numpy(dtype=float)
    dates = t["date"].to_numpy()
    out["kupiec"] = kupiec_pof(n, int(viol.sum()), nominal)
    out["clustered"] = date_clustered_rate_ci(dates, viol, n_boot=n_boot, seed=seed)
    out["clustering"] = date_rate_autocorr_test(dates, viol, seed=seed)
    out["by_vix_band"] = {}
    bands = np.array([vix_band(v) for v in t["vix_entry"].to_numpy(dtype=float)])
    for band in ("calm", "elevated", "crisis", "unknown"):
        m = bands == band
        nb = int(m.sum())
        if nb == 0:
            continue
        out["by_vix_band"][band] = {
            "n": nb,
            "rate": float(viol[m].mean()),
            "kupiec_p": kupiec_pof(nb, int(viol[m].sum()), nominal)["p_value"],
        }
    out["verdict"] = _verdict_lower_quantile(nominal, out["clustered"], n)
    return out


def cvar_breach_report(
    table: pd.DataFrame,
    *,
    var_coverage_bound: float = 0.05,
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Expected-shortfall breach test.

    Because ``ES_5 <= VaR_5``, ``P(realized < cvar_5) < 5%`` for ANY
    correctly modeled distribution — so a breach rate significantly above 5%
    is unambiguous tail understatement without needing the (unemitted) VaR_5.
    Severity among breaches quantifies how far past the modeled expected
    shortfall real losses run (the 2026-06-15 stress-test's 3-4.5x finding,
    formalized).  Strata: entry-VIX band, top-bin membership
    (``prob_profit > 0.90``), and the traded region (``ev_dollars > 0``).
    """
    from scipy.stats import binomtest

    t = _resolved(table)
    t = t[np.isfinite(t["cvar_5"].to_numpy(dtype=float))]
    n = len(t)
    out: dict[str, Any] = {"bound": var_coverage_bound, "n": n}
    if n == 0:
        out["verdict"] = "INSUFFICIENT"
        return out
    realized = t["realized_pnl"].to_numpy(dtype=float)
    cvar = t["cvar_5"].to_numpy(dtype=float)
    breach = realized < cvar
    x = int(breach.sum())
    out["breaches"] = x
    out["rate"] = float(x / n)
    out["binom_p_one_sided"] = float(
        binomtest(x, n, var_coverage_bound, alternative="greater").pvalue
    )
    out["clustered"] = date_clustered_rate_ci(
        t["date"].to_numpy(), breach, n_boot=n_boot, seed=seed
    )
    out["clustering"] = date_rate_autocorr_test(t["date"].to_numpy(), breach, seed=seed)
    if x > 0:
        sev_dollars = realized[breach] - cvar[breach]
        with np.errstate(divide="ignore", invalid="ignore"):
            sev_mult = np.where(cvar[breach] < 0, realized[breach] / cvar[breach], np.nan)
        sev_mult = sev_mult[np.isfinite(sev_mult)]
        worst_idx = np.argsort(realized)[: min(10, n)]
        out["severity"] = {
            "mean_excess_dollars": float(np.mean(sev_dollars)),
            "median_excess_dollars": float(np.median(sev_dollars)),
            "mean_realized_over_cvar": float(np.mean(sev_mult)) if len(sev_mult) else float("nan"),
            "median_realized_over_cvar": (
                float(np.median(sev_mult)) if len(sev_mult) else float("nan")
            ),
        }
        out["worst_cases"] = [
            {
                "date": str(t.iloc[int(i)]["date"]),
                "ticker": str(t.iloc[int(i)]["ticker"]),
                "realized": float(t.iloc[int(i)]["realized_pnl"]),
                "cvar_5": float(t.iloc[int(i)]["cvar_5"]),
                "prob_profit": float(t.iloc[int(i)]["prob_profit"]),
                "vix_entry": float(t.iloc[int(i)]["vix_entry"]),
            }
            for i in worst_idx
            if bool(breach[int(i)])
        ]
    strata: dict[str, Any] = {}
    bands = np.array([vix_band(v) for v in t["vix_entry"].to_numpy(dtype=float)])
    for band in ("calm", "elevated", "crisis", "unknown"):
        m = bands == band
        if m.sum():
            strata[f"vix_{band}"] = {"n": int(m.sum()), "rate": float(breach[m].mean())}
    top = t["prob_profit"].to_numpy(dtype=float) > 0.90
    if top.sum():
        strata["top_bin"] = {"n": int(top.sum()), "rate": float(breach[top].mean())}
    traded = t["ev_dollars"].to_numpy(dtype=float) > 0.0
    if traded.sum():
        strata["traded_region"] = {"n": int(traded.sum()), "rate": float(breach[traded].mean())}
    out["strata"] = strata
    if n < MIN_N_FOR_VERDICT:
        out["verdict"] = "INSUFFICIENT"
    elif (
        np.isfinite(out["clustered"]["ci_low"]) and out["clustered"]["ci_low"] > var_coverage_bound
    ):
        out["verdict"] = "FAIL"
    elif out["rate"] > var_coverage_bound:
        out["verdict"] = "WARN"
    else:
        out["verdict"] = "PASS"
    return out


def full_report(table: pd.DataFrame, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the complete exceedance report for a captured tail table."""
    resolved = _resolved(table)
    win = (resolved["realized_pnl"].to_numpy(dtype=float) > 0.0).astype(float)
    report: dict[str, Any] = {
        "meta": dict(meta or {}),
        "rows_total": int(len(table)),
        "rows_resolved": int(len(resolved)),
        "quantiles": {
            "p25": quantile_coverage_report(table, quantile_col="pnl_p25", nominal=0.25),
            "p50": quantile_coverage_report(table, quantile_col="pnl_p50", nominal=0.50),
            "p75": quantile_coverage_report(table, quantile_col="pnl_p75", nominal=0.75),
        },
        "cvar_5": cvar_breach_report(table),
        "prob_profit_pooled": heterogeneous_coverage_z(
            resolved["prob_profit"].to_numpy(dtype=float), win
        ),
    }
    verdicts = [report["quantiles"][k]["verdict"] for k in ("p25", "p50", "p75")]
    verdicts.append(report["cvar_5"]["verdict"])
    if all(v == "INSUFFICIENT" for v in verdicts):
        overall = "INSUFFICIENT"
    elif "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    else:
        overall = "PASS"
    report["overall_verdict"] = overall
    return report
