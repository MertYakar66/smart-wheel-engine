"""Parameter out-of-sample (parameter-OOS) validation harness.

**What this measures — and why it is NOT S35.**  Every locked backtest
(S27/S32/S34/S35) reports the *shipped* engine's rank quality on some window.
S35 is out-of-*window*: it runs the same fixed parameterization on a period the
*window* had not been evaluated on, but the static magic numbers (regime state
weights, dealer clamp, POT-GPD threshold, R11 cutoffs, ...) were all hand-set
with full-history visibility (caveat **E5**,
``docs/BACKTEST_REGRESSION_CAMPAIGN.md``). So S35 does not answer "is the edge
fit to the sample?".

This harness answers exactly that, for the one parameter surface that is
*active* on the Bloomberg provider — the regime (HMM state-weight) overlay.  It:

1. Runs the production ranker ONCE over a sampled date grid and forward-replays
   every ranked row to its held-to-expiry realized P&L — the same rho signal
   ``_common._compute_metrics`` locks for S27/S34 — capturing the diagnostic
   ``ev_raw`` / ``hmm_regime`` / ``regime_multiplier`` columns so the parameter
   overlay can be *re-derived offline* without re-running the engine.
2. Splits the rows into a TRAIN partition and a temporally-DISJOINT holdout with
   an embargo wide enough that **every train row's option has already expired
   before the first holdout ranking date** (``assert_no_leakage`` enforces this
   per-row — the airtight leakage proof).
3. Re-selects the regime overlay (per-regime scalars, and a multiplier-tilt
   exponent γ) on TRAIN ONLY, then evaluates the re-fit on the holdout, so the
   in-sample→out-of-parameter optimism gap is explicit.

Nothing here feeds a trade, a ranking, or ``ev_dollars``: it CALLS
``WheelRunner.rank_candidates_by_ev`` read-only and does all re-weighting in
numpy on the captured table.  The engine's production defaults are never
mutated — grid values are applied *offline* to the ``ev_raw`` column
(CLAUDE.md §2 / task invariant 3).

The shipped regime weights (``engine/regime_hmm.py`` ``position_multiplier``)
are ``{crisis: 0.2, bear: 0.5, normal: 1.0, bull_quiet: 1.25}``; the engine
applies them as a posterior-weighted blend, so ``regime_multiplier`` varies
*within* a label.  The offline re-fit keys on the dominant-state label
(``hmm_regime``) — a documented approximation of the weight vector, interpretable
and mapping 1:1 to the four shipped constants.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical dominant-state labels emitted by ``GaussianHMM`` state labelling.
REGIME_LABELS = ("crisis", "bear", "normal", "bull_quiet")

#: Shipped per-label weights (engine/regime_hmm.py ``position_multiplier``).
SHIPPED_REGIME_WEIGHTS: dict[str, float] = {
    "crisis": 0.2,
    "bear": 0.5,
    "normal": 1.0,
    "bull_quiet": 1.25,
}

#: Columns captured per ranked row.  ``ev_raw`` is the pre-overlay EV; the
#: overlay is recovered offline as ``ev_dollars / ev_raw``.
RANK_TABLE_COLUMNS = (
    "date",
    "ticker",
    "ev_raw",
    "ev_dollars",
    "hmm_regime",
    "hmm_multiplier",
    "dealer_multiplier",
    "regime_multiplier",
    "prob_profit",
    "iv",
    "premium",
    "strike",
    "expiration_date",
    "spot_at_expiry",
    "realized_pnl",
)


# ---------------------------------------------------------------------------
# Date sampling
# ---------------------------------------------------------------------------


def sample_business_days(start: str, end: str, every_n: int) -> list[date]:
    """Every ``every_n``-th business day in ``[start, end]`` inclusive.

    Sub-sampling keeps the one engine pass affordable; rank rho is a
    cross-sectional statistic so a coarser calendar grid does not bias it.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    bdays = [d.date() for d in pd.bdate_range(start, end)]
    return bdays[::every_n]


# ---------------------------------------------------------------------------
# Engine pass — build the per-row rank table (the expensive, one-time step)
# ---------------------------------------------------------------------------


def build_rank_table(
    *,
    tickers: Sequence[str],
    sample_dates: Sequence[date],
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
    contracts: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the production ranker over ``sample_dates`` and forward-replay each
    ranked row to its held-to-expiry realized P&L.

    Reuses the S27/S34 forward-replay helpers from
    ``backtests.regression._common`` so the realized-P&L convention (and thus
    the rho it feeds) is byte-identical to the locked regression snapshots.
    Deterministic: fixed data + HMM ``random_state=42``; the option-premium
    rail is pinned OFF exactly as the regression driver does.
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
    logger.info("build_rank_table: connector=%s", type(conn).__name__)

    rows: list[dict] = []
    n = len(sample_dates)
    t0 = time.time()
    for i, today in enumerate(sample_dates):
        if progress and i and i % max(1, n // 20) == 0:
            el = time.time() - t0
            eta = (n - i) / (i / el) if el > 0 else 0.0
            print(
                f"[param_oos] {i:4d}/{n} ({100 * i / n:5.1f}%) "
                f"elapsed {el / 60:5.1f}m ETA {eta / 60:5.1f}m",
                flush=True,
            )
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
                    as_of=today.isoformat(),
                    include_diagnostic_fields=True,
                )
        except Exception as exc:  # noqa: BLE001 — a bad day should not abort the pass
            logger.warning("rank failed @ %s: %s", today, exc)
            continue
        if frame is None or len(frame) == 0:
            continue
        for _, r in frame.iterrows():
            rows.append(
                {
                    "date": today.isoformat(),
                    "ticker": str(r.get("ticker", "")),
                    "ev_raw": float(r.get("ev_raw", float("nan"))),
                    "ev_dollars": float(r.get("ev_dollars", 0.0)),
                    "hmm_regime": str(r.get("hmm_regime", "unknown")),
                    "hmm_multiplier": float(r.get("hmm_multiplier", float("nan"))),
                    "dealer_multiplier": float(r.get("dealer_multiplier", float("nan"))),
                    "regime_multiplier": float(r.get("regime_multiplier", float("nan"))),
                    "prob_profit": float(r.get("prob_profit", float("nan"))),
                    "iv": float(r.get("iv", 0.0)),
                    "premium": float(r.get("premium", 0.0)),
                    "strike": float(r.get("strike", 0.0)),
                    "expiration_date": expiration_default.isoformat(),
                }
            )

    table = pd.DataFrame(rows, columns=[c for c in RANK_TABLE_COLUMNS if c != "spot_at_expiry"])
    # Forward-replay realized P&L: spot at expiration (cache per (ticker, exp)).
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
# Pure-numpy analysis (no engine; runs in the fast CI lane)
# ---------------------------------------------------------------------------


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ; NaN for n<2 or constant input (caller handles)."""
    from scipy.stats import spearmanr

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    xv, yv = x[m], y[m]
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return float("nan")
    res = spearmanr(xv, yv)
    return float(res.correlation)


def brier_score(prob: np.ndarray, win: np.ndarray) -> float:
    """Mean squared error of the ``prob_profit`` forecast vs the realized win
    indicator.  Lower is better; 0.25 is the coin-flip reference."""
    m = np.isfinite(prob) & np.isfinite(win)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((prob[m] - win[m]) ** 2))


def expected_calibration_error(prob: np.ndarray, win: np.ndarray, *, n_bins: int = 10) -> float:
    """Sample-size-weighted mean |forecast − realized| across probability bins.

    The engine's ``prob_profit`` is the forecast; ``win`` is ``realized_pnl>0``.
    Captures the top-bin over-confidence the heavy-verify campaign flagged
    (forecast ~0.96 vs realized ~0.57 in crisis).
    """
    m = np.isfinite(prob) & np.isfinite(win)
    if m.sum() == 0:
        return float("nan")
    p, w = prob[m], win[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        ece += (sel.sum() / total) * abs(p[sel].mean() - w[sel].mean())
    return float(ece)


def _win(table: pd.DataFrame) -> np.ndarray:
    return (table["realized_pnl"].to_numpy() > 0).astype(float)


def scorecard(table: pd.DataFrame, *, signal_col: str = "ev_dollars") -> dict[str, Any]:
    """Rank-quality + calibration scorecard for a set of replayed rows.

    ``rho`` is Spearman(signal, realized_pnl) — the S27/S34 metric.
    ``hit_rate`` is P(realized_pnl>0).  ``mean_realized`` is the per-contract
    dollar edge (a portfolio-NAV *proxy*; it is NOT capital-constrained
    portfolio NAV — that needs the full tracker backtest, see the driver).
    """
    replayed = table.dropna(subset=["realized_pnl"])
    n = int(len(replayed))
    out: dict[str, Any] = {
        "n": n,
        "rho": float("nan"),
        "hit_rate": float("nan"),
        "mean_realized": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }
    if n >= 2:
        sig = replayed[signal_col].to_numpy(dtype=float)
        pnl = replayed["realized_pnl"].to_numpy(dtype=float)
        prob = replayed["prob_profit"].to_numpy(dtype=float)
        win = _win(replayed)
        out.update(
            {
                "rho": spearman_rho(sig, pnl),
                "hit_rate": float(win.mean()),
                "mean_realized": float(pnl.mean()),
                "brier": brier_score(prob, win),
                "ece": expected_calibration_error(prob, win),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Offline re-weighting of the regime overlay
# ---------------------------------------------------------------------------


def apply_regime_scalars(table: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """``ev_raw × weights[label]`` per row — the offline re-parameterized signal.

    ``weights`` maps a dominant-state label to a scalar.  Missing labels default
    to 1.0 (neutral).  Never touches the engine; pure numpy on the captured
    ``ev_raw`` column.
    """
    ev_raw = table["ev_raw"].to_numpy(dtype=float)
    lbl = table["hmm_regime"].to_numpy()
    scal = np.array([weights.get(str(x), 1.0) for x in lbl], dtype=float)
    return ev_raw * scal


def apply_tilt_exponent(table: pd.DataFrame, gamma: float) -> np.ndarray:
    """``ev_raw × (ev_dollars/ev_raw)**gamma`` — dial the shipped multiplier
    tilt up/down.  γ=0 ⇒ ev_raw (no overlay); γ=1 ⇒ shipped ev_dollars.

    Recovers the exact per-row combined multiplier from the two captured
    columns, so it exercises the FULL overlay (regime × any active dealer/
    heavy-tail), not just the per-label approximation.
    """
    ev_raw = table["ev_raw"].to_numpy(dtype=float)
    ev_dollars = table["ev_dollars"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = np.where(ev_raw != 0.0, ev_dollars / ev_raw, 1.0)
        # Multiplier is >=0 by construction; guard the pow domain.
        mult = np.clip(mult, 0.0, None)
        return ev_raw * np.power(mult, gamma)


def _rho_of_signal(table: pd.DataFrame, signal: np.ndarray) -> float:
    pnl = table["realized_pnl"].to_numpy(dtype=float)
    return spearman_rho(signal, pnl)


def refit_regime_scalars(
    train: pd.DataFrame,
    *,
    grid: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    gauge_label: str = "normal",
) -> tuple[dict[str, float], float]:
    """Grid-search per-label scalars that maximise TRAIN Spearman rho.

    ``normal`` is pinned to 1.0 as the gauge (only relative cross-label tilt
    changes a rank correlation), so the search is over the three
    non-gauge labels.  Returns ``(best_weights, best_train_rho)``.

    Leakage note: uses ONLY ``train`` rows (and their already-realized
    outcomes).  The caller must have passed a train partition whose options all
    expire before the holdout begins (``assert_no_leakage``).
    """
    replayed = train.dropna(subset=["realized_pnl"])
    non_gauge = [lbl for lbl in REGIME_LABELS if lbl != gauge_label]
    best_w = dict.fromkeys(REGIME_LABELS, 1.0)
    best_rho = _rho_of_signal(replayed, apply_regime_scalars(replayed, best_w))
    # Full grid over the three non-gauge labels.
    for a in grid:
        for b in grid:
            for c in grid:
                w = {gauge_label: 1.0, non_gauge[0]: a, non_gauge[1]: b, non_gauge[2]: c}
                rho = _rho_of_signal(replayed, apply_regime_scalars(replayed, w))
                if np.isfinite(rho) and rho > best_rho:
                    best_rho, best_w = rho, w
    return best_w, float(best_rho)


def refit_tilt_exponent(
    train: pd.DataFrame,
    *,
    grid: Sequence[float] = tuple(round(g, 2) for g in np.arange(0.0, 3.01, 0.25)),
) -> tuple[float, float]:
    """Grid-search the multiplier-tilt exponent γ that maximises TRAIN rho.
    Returns ``(best_gamma, best_train_rho)``."""
    replayed = train.dropna(subset=["realized_pnl"])
    best_g, best_rho = 1.0, _rho_of_signal(replayed, apply_tilt_exponent(replayed, 1.0))
    for g in grid:
        rho = _rho_of_signal(replayed, apply_tilt_exponent(replayed, float(g)))
        if np.isfinite(rho) and rho > best_rho:
            best_rho, best_g = rho, float(g)
    return best_g, float(best_rho)


# ---------------------------------------------------------------------------
# Leakage proof + partitioning
# ---------------------------------------------------------------------------


@dataclass
class Partition:
    """A train/holdout split with a machine-checkable no-leakage certificate."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    train_end: date
    holdout_start: date
    embargo_days: int
    certificate: dict[str, Any] = field(default_factory=dict)


def assert_no_leakage(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    """Prove the split is leakage-free and return the certificate.

    The airtight, per-row-checkable invariant: **every train row's option
    expires strictly before the earliest holdout ranking date**.  Because a
    train row's realized P&L becomes knowable only at its expiration, this
    guarantees no train outcome carries information dated at/after any holdout
    ranking decision — so a parameter re-fit on train cannot have "seen"
    holdout.  Raises ``AssertionError`` on any violation.
    """
    if train.empty or holdout.empty:
        raise AssertionError("empty partition — cannot certify leakage-free")
    max_train_exp = pd.to_datetime(train["expiration_date"]).max().date()
    min_hold_asof = pd.to_datetime(holdout["date"]).min().date()
    max_train_asof = pd.to_datetime(train["date"]).max().date()
    ok = max_train_exp < min_hold_asof
    cert = {
        "max_train_ranking_date": max_train_asof.isoformat(),
        "max_train_expiration_date": max_train_exp.isoformat(),
        "min_holdout_ranking_date": min_hold_asof.isoformat(),
        "gap_days_expiry_to_holdout": (min_hold_asof - max_train_exp).days,
        "leakage_free": bool(ok),
    }
    if not ok:
        raise AssertionError(
            "LEAKAGE: a train option expires on/after the first holdout ranking date "
            f"({max_train_exp} >= {min_hold_asof}). Widen the embargo."
        )
    return cert


def make_partition(table: pd.DataFrame, *, train_end: str, holdout_start: str) -> Partition:
    """Split ``table`` at explicit dates and certify no leakage.

    ``train`` = ranking date ≤ ``train_end``; ``holdout`` = ranking date ≥
    ``holdout_start``.  Rows in the embargo gap ``(train_end, holdout_start)``
    are dropped.  Rows whose option never resolved (realized_pnl NaN) are kept
    in the table but ignored by every rho/scorecard computation.
    """
    te = date.fromisoformat(train_end)
    hs = date.fromisoformat(holdout_start)
    if hs <= te:
        raise ValueError(f"holdout_start {hs} must be after train_end {te}")
    d = pd.to_datetime(table["date"]).dt.date
    train = table[d <= te].copy()
    holdout = table[d >= hs].copy()
    cert = assert_no_leakage(train, holdout)
    return Partition(
        train=train,
        holdout=holdout,
        train_end=te,
        holdout_start=hs,
        embargo_days=(hs - te).days,
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# Walk-forward folds (Phase 1) + parameter hold-out (Phase 2) reports
# ---------------------------------------------------------------------------


def rolling_folds(
    table: pd.DataFrame, *, n_folds: int, embargo_days: int = 45
) -> list[dict[str, Any]]:
    """Contiguous rolling test folds over the sampled span, each scored with
    the FIXED shipped ``ev_dollars``.

    This is an out-of-*window* stability scorecard for the shipped
    parameterization (the online HMM/POT-GPD components are PIT-clean; the
    static constants saw all folds during hand-tuning — caveat E5).  Each
    fold's rho/hit/calibration stands alone; ``embargo_days`` only documents
    the intended purge (adjacent folds share no ranking date, and each row's
    realized P&L is intrinsically observable by its own expiration).
    """
    d = pd.to_datetime(table["date"]).dt.date
    lo, hi = d.min(), d.max()
    span = (hi - lo).days
    step = span // n_folds
    folds = []
    for k in range(n_folds):
        f_start = lo + timedelta(days=k * step)
        f_end = lo + timedelta(days=(k + 1) * step) if k < n_folds - 1 else hi
        sel = (d >= f_start) & (d <= f_end)
        sub = table[sel]
        sc = scorecard(sub, signal_col="ev_dollars")
        folds.append(
            {
                "fold": k + 1,
                "start": f_start.isoformat(),
                "end": f_end.isoformat(),
                **sc,
            }
        )
    return folds


def parameter_holdout_report(part: Partition) -> dict[str, Any]:
    """The Phase-2 in-sample→out-of-parameter table for the regime overlay.

    Compares three parameterizations of the regime overlay on TRAIN and on the
    disjoint HOLDOUT:

    * ``ev_raw`` — no overlay (γ=0);
    * ``shipped`` — the production weights / ev_dollars (γ=1);
    * ``refit`` — per-regime scalars (and γ) re-selected on TRAIN ONLY.

    The optimism gap = TRAIN rho at the re-fit optimum − HOLDOUT rho at that
    same re-fit.  A positive gap that the shipped constant does not close on the
    holdout is the E5 overfitting signal.
    """
    train, holdout = part.train, part.holdout

    # Re-fit on TRAIN only.
    w_star, train_rho_scalars = refit_regime_scalars(train)
    g_star, train_rho_gamma = refit_tilt_exponent(train)

    def _rho(tbl: pd.DataFrame, signal: np.ndarray) -> float:
        return spearman_rho(signal, tbl["realized_pnl"].to_numpy(dtype=float))

    rep: dict[str, Any] = {
        "leakage_certificate": part.certificate,
        "train_n": int(train.dropna(subset=["realized_pnl"]).shape[0]),
        "holdout_n": int(holdout.dropna(subset=["realized_pnl"]).shape[0]),
        "refit_regime_weights": w_star,
        "refit_tilt_exponent": g_star,
        "shipped_regime_weights": dict(SHIPPED_REGIME_WEIGHTS),
        "variants": {},
    }

    # ev_raw (no overlay)
    rep["variants"]["ev_raw_no_overlay"] = {
        "train_rho": _rho(train, apply_regime_scalars(train, {})),
        "holdout_rho": _rho(holdout, apply_regime_scalars(holdout, {})),
    }
    # shipped ev_dollars (γ=1 on the full overlay)
    rep["variants"]["shipped"] = {
        "train_rho": _rho(train, train["ev_dollars"].to_numpy(dtype=float)),
        "holdout_rho": _rho(holdout, holdout["ev_dollars"].to_numpy(dtype=float)),
    }
    # per-regime scalar re-fit (train-optimal)
    rep["variants"]["refit_regime_scalars"] = {
        "train_rho": train_rho_scalars,
        "holdout_rho": _rho(holdout, apply_regime_scalars(holdout, w_star)),
    }
    # tilt-exponent re-fit (train-optimal)
    rep["variants"]["refit_tilt_exponent"] = {
        "train_rho": train_rho_gamma,
        "holdout_rho": _rho(holdout, apply_tilt_exponent(holdout, g_star)),
    }

    # Regime-conditional holdout breakdown (leakage-clean: pure conditioning of
    # the already-disjoint holdout).  Surfaces WHERE the fixed overlay is
    # (un)reliable OOS — the crisis bin is the i9 unreliable one (realized rate
    # swings 0.37-0.93); this reports its holdout ρ/calibration directly rather
    # than making a leaky cross-time leave-one-crisis-out claim.
    holdout_replayed = holdout.dropna(subset=["realized_pnl"])
    by_regime: dict[str, Any] = {}
    for lbl in REGIME_LABELS:
        sub = holdout_replayed[holdout_replayed["hmm_regime"] == lbl]
        by_regime[lbl] = scorecard(sub, signal_col="ev_dollars")
    rep["holdout_by_regime"] = by_regime

    # Headline optimism gaps.
    v = rep["variants"]
    rep["optimism_gap_regime_scalars"] = (
        v["refit_regime_scalars"]["train_rho"] - v["refit_regime_scalars"]["holdout_rho"]
    )
    rep["optimism_gap_tilt_exponent"] = (
        v["refit_tilt_exponent"]["train_rho"] - v["refit_tilt_exponent"]["holdout_rho"]
    )
    # Did honest train-only re-fitting beat the shipped constant OUT of sample?
    rep["refit_beats_shipped_on_holdout"] = bool(
        v["refit_regime_scalars"]["holdout_rho"] > v["shipped"]["holdout_rho"]
    )
    # Does the overlay add anything OOS beyond ev_raw?
    rep["overlay_adds_oos_value"] = bool(
        v["shipped"]["holdout_rho"] > v["ev_raw_no_overlay"]["holdout_rho"]
    )
    return rep


def split_robustness_report(
    table: pd.DataFrame, splits: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    """The optimism gap across several leakage-certified train/holdout splits.

    Guards against the "you cherry-picked the split date" critique: re-runs the
    regime-overlay re-fit at each ``(train_end, holdout_start)`` and reports the
    gap.  A finding that only holds at one split is not robust; a finding that
    holds across most splits is.  Every split is independently leakage-certified.
    """
    out: list[dict[str, Any]] = []
    for train_end, holdout_start in splits:
        part = make_partition(table, train_end=train_end, holdout_start=holdout_start)
        rep = parameter_holdout_report(part)
        v = rep["variants"]
        out.append(
            {
                "train_end": train_end,
                "holdout_start": holdout_start,
                "train_n": rep["train_n"],
                "holdout_n": rep["holdout_n"],
                "leakage_free": part.certificate["leakage_free"],
                "refit_train_rho": v["refit_regime_scalars"]["train_rho"],
                "refit_holdout_rho": v["refit_regime_scalars"]["holdout_rho"],
                "shipped_holdout_rho": v["shipped"]["holdout_rho"],
                "optimism_gap": rep["optimism_gap_regime_scalars"],
            }
        )
    return out
