"""Distributional simulated-portfolio reporting for the wheel book.

What this is
------------
A **reporting overlay** that turns the engine's existing-but-dormant Monte
Carlo / copula machinery into a live distributional view of a simulated
wheel book. Given the day-by-day equity curve of a WheelTracker-driven
forward book (produced by the production ranker via
``backtests.regression._common.run_backtest`` / ``run_backtest_multi_friction``),
this module derives a strategy-return series and projects the **full outcome
distribution** — an equity fan (p5/p25/p50/p75/p95), a terminal-return
distribution, a drawdown distribution, and a correlation-to-1 tail-stress
number — instead of the single deterministic backtest path.

Where it sits relative to the decision layer (CLAUDE.md §2)
-----------------------------------------------------------
**Nowhere on it.** This module is pure post-hoc analytics:

* It never imports ``engine.ev_engine`` / ``engine.wheel_runner`` /
  ``engine.candidate_dossier`` and never constructs or mutates an EV verdict.
* Every number it produces is *reporting* — a ``model`` projection
  (Monte Carlo / copula) or an ``engine-measured`` realized statistic. None
  of it feeds back into ``ev_dollars``, ``ev_raw``, ``prob_profit``, a
  reviewer verdict, the dealer/regime multipliers, or the R7/R8 VaR/stress
  gate thresholds. The copula tail is a portfolio-level *observation* of the
  coordinated-drawdown risk the per-name-independent EV path cannot see —
  surfacing it is the point; feeding it back would violate §2 invariant 5.

It reuses (does not reimplement):

* :class:`engine.monte_carlo.BlockBootstrap` — the block-bootstrap engine.
* :func:`engine.portfolio_copula.portfolio_cvar_copula` — Gaussian vs
  Student-t copula tail.
* :mod:`engine.performance_metrics` — return / drawdown / Sharpe helpers.

Honest-reporting caveats these functions preserve (do not paper over)
---------------------------------------------------------------------
The projections inherit every caveat of the backtest they are built on:

* **E1** ~92% of the backtest NAV gain was equity-beta on assigned stock,
  not put-selection alpha; **E3** P&L was single-name-dominated (BKNG).
* **E5** locked engine claims are parameter-*in-sample* (HMM / POT-GPD /
  dealer clamp tuned on full history).
* **D19** ``ev_dollars`` nets only the entry-leg cost (~$1-4/ct optimistic);
  **D21** the forward-distribution samplers index trading-day bars against
  calendar DTE (~46% horizon over-dispersion).
* **Bootstrap-specific:** the strategy-return series is derived from the
  equity curve, which the tracker only marks on days with an open position
  — so it is the book's **deployed-capital** return, not a calendar return.
  The block bootstrap resamples those steps i.i.d.-in-blocks; it assumes the
  future return-generating process resembles the sampled window and cannot
  invent regimes absent from it (survivorship: universe = current members).

These are surfaced in :func:`build_sim_report`'s ``caveats`` block so any
consumer sees them alongside the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .monte_carlo import BlockBootstrap
from .portfolio_copula import portfolio_cvar_copula

# Canonical reproducibility seed (matches engine.monte_carlo).
CANONICAL_SEED = 42

# Quantile ladder for the equity fan / terminal / drawdown bands.
_DEFAULT_QUANTILES = (5, 25, 50, 75, 95)


# ---------------------------------------------------------------------------
# 1. Strategy-return derivation
# ---------------------------------------------------------------------------


def strategy_returns_from_equity_curve(
    equity_curve: list[dict] | pd.DataFrame,
    *,
    value_key: str = "portfolio_value",
) -> np.ndarray:
    """Per-step strategy returns from a tracker equity curve.

    ``equity_curve`` is the ``WheelTracker.equity_curve`` list (or a
    DataFrame of it) — rows ``{date, portfolio_value, cash, num_positions}``
    appended by :meth:`WheelTracker.mark_to_market` on every day the book
    holds at least one position. Returns ``diff / prev`` over the
    ``portfolio_value`` column, dropping the leading NaN and any non-finite
    steps.

    The result is the **deployed-capital** return series (see module
    caveats): each element is one mark-to-mark step, not a guaranteed single
    calendar day. Callers wanting a calendar-aligned series should resample
    the curve on ``date`` first.
    """
    if isinstance(equity_curve, pd.DataFrame):
        df = equity_curve
    else:
        df = pd.DataFrame(equity_curve)
    if df.empty or value_key not in df.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(df[value_key], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.asarray([], dtype=float)
    rets = np.diff(values) / values[:-1]
    return rets[np.isfinite(rets)]


# ---------------------------------------------------------------------------
# 2. Monte Carlo equity-fan bands
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloBands:
    """Distributional projection of a strategy-return series.

    Every field here is a **model** output (Monte Carlo), not an
    engine-measured realized value. ``to_dict`` is JSON-safe for persistence.
    """

    # Equity fan: for each quantile q, the per-day band value across sims.
    # Length = n_days + 1 (index 0 == initial_capital for a clean fan start).
    equity_bands: dict[str, list[float]]
    band_days: list[int]  # 0..n_days step index for the fan x-axis
    quantiles: list[int]

    # Terminal outcome distribution (fraction returns, e.g. 0.12 == +12%).
    terminal_return_quantiles: dict[str, float]
    terminal_value_quantiles: dict[str, float]

    # Drawdown distribution — magnitudes (positive fractions), per-sim worst DD.
    max_drawdown_quantiles: dict[str, float]

    # Headline risk metrics (all from engine.monte_carlo.BootstrapResult).
    median_return: float
    mean_return: float
    var_5: float
    cvar_5: float
    prob_loss: float
    prob_severe_loss: float
    return_ci_95: tuple[float, float]
    return_ci_99: tuple[float, float]
    median_sharpe: float

    # Reproducibility / provenance.
    n_simulations: int
    n_days: int
    block_size: int
    seed: int | None
    initial_capital: float
    n_hist_returns: int

    def to_dict(self) -> dict:
        return {
            "kind": "model",  # Monte Carlo projection — not engine-measured
            "method": "block_bootstrap",
            "equity_bands": self.equity_bands,
            "band_days": self.band_days,
            "quantiles": self.quantiles,
            "terminal_return_quantiles": self.terminal_return_quantiles,
            "terminal_value_quantiles": self.terminal_value_quantiles,
            "max_drawdown_quantiles": self.max_drawdown_quantiles,
            "median_return": self.median_return,
            "mean_return": self.mean_return,
            "var_5": self.var_5,
            "cvar_5": self.cvar_5,
            "prob_loss": self.prob_loss,
            "prob_severe_loss": self.prob_severe_loss,
            "return_ci_95": list(self.return_ci_95),
            "return_ci_99": list(self.return_ci_99),
            "median_sharpe": self.median_sharpe,
            "n_simulations": self.n_simulations,
            "n_days": self.n_days,
            "block_size": self.block_size,
            "seed": self.seed,
            "initial_capital": self.initial_capital,
            "n_hist_returns": self.n_hist_returns,
        }


def monte_carlo_bands(
    daily_returns: np.ndarray,
    *,
    initial_capital: float,
    n_days: int | None = None,
    n_simulations: int = 10_000,
    block_size: int = 21,
    seed: int | None = CANONICAL_SEED,
    quantiles: tuple[int, ...] = _DEFAULT_QUANTILES,
) -> MonteCarloBands:
    """Project a strategy-return series into a distributional equity fan.

    Wraps :class:`engine.monte_carlo.BlockBootstrap`. By default the horizon
    ``n_days`` equals ``len(daily_returns)`` — this is the horizon at which
    the **median** bootstrap terminal reconciles with the deterministic
    backtest's realized final NAV (E[log terminal] over the empirical return
    distribution equals the realized log terminal when the projected step
    count matches the sampled step count), which
    :func:`reconcile_with_backtest` checks.

    Args:
        daily_returns: The per-step strategy returns (fractions).
        initial_capital: Starting capital for the fan (typically the book's
            initial NAV — reconciliation is scale-free but the fan is in $).
        n_days: Projection horizon in steps. Defaults to the history length.
        n_simulations: Bootstrap path count (10k canonical).
        block_size: Block length in steps (21 ~ one trading month; must be
            <= len(daily_returns)).
        seed: PRNG seed (42 canonical for determinism).
        quantiles: Percentile ladder for the fan / terminal / drawdown bands.

    Returns:
        A :class:`MonteCarloBands`.

    Raises:
        ValueError: if the return series is shorter than ``block_size`` (the
            bootstrap cannot form a block) — the caller must widen the
            window or shrink the block.
    """
    rets = np.asarray(daily_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    n_hist = len(rets)
    if n_hist < block_size:
        raise ValueError(
            f"strategy-return series too short for block bootstrap: got {n_hist} "
            f"finite returns, need >= block_size={block_size}. Widen the backtest "
            f"window or lower block_size."
        )
    horizon = int(n_days) if n_days is not None else n_hist

    boot = BlockBootstrap(block_size=block_size, n_simulations=n_simulations, seed=seed)
    res = boot.simulate(rets, n_days=horizon, initial_capital=float(initial_capital))

    # Equity fan: percentile across sims, per day. Prepend day 0 = capital.
    curves = res.equity_curves  # (n_sim, horizon)
    band_qs = list(quantiles)
    equity_bands: dict[str, list[float]] = {}
    for q in band_qs:
        per_day = np.percentile(curves, q, axis=0)  # (horizon,)
        equity_bands[f"p{q}"] = [float(initial_capital)] + [float(v) for v in per_day]
    band_days = list(range(horizon + 1))

    # Terminal distributions.
    total_returns = (res.terminal_values / float(initial_capital)) - 1.0
    terminal_return_quantiles = {f"p{q}": float(np.percentile(total_returns, q)) for q in band_qs}
    terminal_value_quantiles = {
        f"p{q}": float(np.percentile(res.terminal_values, q)) for q in band_qs
    }

    # Drawdown distribution — report magnitudes (positive). BootstrapResult
    # stores the most-negative drawdown per sim; flip sign so p95 == worst.
    dd_mag = np.abs(res.max_drawdown_dist)
    max_drawdown_quantiles = {f"p{q}": float(np.percentile(dd_mag, q)) for q in band_qs}

    return MonteCarloBands(
        equity_bands=equity_bands,
        band_days=band_days,
        quantiles=band_qs,
        terminal_return_quantiles=terminal_return_quantiles,
        terminal_value_quantiles=terminal_value_quantiles,
        max_drawdown_quantiles=max_drawdown_quantiles,
        median_return=float(res.median_return),
        mean_return=float(res.mean_return),
        var_5=float(res.var_5),
        cvar_5=float(res.cvar_5),
        prob_loss=float(res.prob_loss),
        prob_severe_loss=float(res.prob_severe_loss),
        return_ci_95=(float(res.return_ci_95[0]), float(res.return_ci_95[1])),
        return_ci_99=(float(res.return_ci_99[0]), float(res.return_ci_99[1])),
        median_sharpe=float(np.median(res.sharpe_dist)),
        n_simulations=n_simulations,
        n_days=horizon,
        block_size=block_size,
        seed=seed,
        initial_capital=float(initial_capital),
        n_hist_returns=n_hist,
    )


# ---------------------------------------------------------------------------
# 3. Reconciliation with the deterministic backtest
# ---------------------------------------------------------------------------


def reconcile_with_backtest(
    bands: MonteCarloBands,
    *,
    realized_final_nav: float,
    initial_capital: float,
    median_tol_pct: float = 10.0,
) -> dict:
    """Sanity-check the MC median against the deterministic backtest NAV.

    ``initial_capital`` here MUST be the same base the bands were built with,
    and — for the reconciliation to measure bootstrap fidelity rather than a
    base-ratio artifact — that base should be the **first equity mark**
    (``build_sim_report`` passes it as such). Then the resampled series
    (returns derived off the first mark) compounds from that base to the
    realized final NAV, so the gap isolates sampling error.

    When the MC horizon equals the history length (the default), the *median*
    bootstrap terminal **approximately** tracks the realized final NAV. The
    relation is approximate, not an exact identity, for two reasons: (a) the
    resampled *log*-terminal has expectation ``n_days · mean(log(1+r))`` which
    at ``n_days == n_hist`` equals the realized ``sum(log(1+r))`` — but that is
    the mean of the log, whereas we check the **median** of the terminal
    *level*, and ``median(exp(·)) ≠ exp(mean(·))`` under skew (Jensen); (b)
    block resampling adds variance. So expect a small systematic offset plus
    noise, not zero. Two checks:

    1. **In-band:** the realized total return lies within the MC p5-p95
       terminal-return band (the realized path is a plausible draw). Note this
       is a weak check when ``n_days == n_hist`` (the band is centred on the
       realized path by construction); it mainly catches a grossly wrong base.
    2. **Median-close:** the MC median terminal is within ``median_tol_pct``
       (default 10%) of the realized final NAV. The tolerance is deliberately
       loose to absorb the (a)+(b) offset; the *reported* gap (typically
       ~1-2%) is the honest number, not the threshold.

    ``reconciled`` is the AND of both. All figures are returned so a caller
    can report an honest near-miss rather than a bare bool.
    """
    realized_return = float(realized_final_nav) / float(initial_capital) - 1.0
    p5 = bands.terminal_return_quantiles.get("p5")
    p95 = bands.terminal_return_quantiles.get("p95")
    median_return = bands.median_return
    mc_median_nav = float(initial_capital) * (1.0 + median_return)

    in_band = p5 is not None and p95 is not None and (p5 <= realized_return <= p95)
    median_abs_pct_gap = (
        abs(mc_median_nav - float(realized_final_nav)) / abs(float(realized_final_nav)) * 100.0
        if realized_final_nav
        else float("inf")
    )
    median_close = median_abs_pct_gap <= median_tol_pct

    return {
        "realized_final_nav": float(realized_final_nav),
        "realized_total_return": realized_return,
        "mc_median_return": float(median_return),
        "mc_median_nav": float(mc_median_nav),
        "mc_terminal_return_p5": None if p5 is None else float(p5),
        "mc_terminal_return_p95": None if p95 is None else float(p95),
        "median_abs_pct_gap": float(median_abs_pct_gap),
        "median_tol_pct": float(median_tol_pct),
        "in_band": bool(in_band),
        "median_close": bool(median_close),
        "reconciled": bool(in_band and median_close),
    }


# ---------------------------------------------------------------------------
# 4. Portfolio correlation tail (copula) — REPORTING / STRESS OVERLAY ONLY
# ---------------------------------------------------------------------------


@dataclass
class CorrelationTail:
    """Copula tail report for a book of underlyings.

    REPORTING / STRESS OVERLAY ONLY (CLAUDE.md §2 invariant 5): the numbers
    here are a portfolio-level observation of coordinated-drawdown risk. They
    NEVER feed ``ev_dollars`` / a verdict / the R7-R8 gate thresholds.

    Two scenarios are reported:

    * ``empirical`` — the copula tail under the book's realized cross-name
      correlation.
    * ``stress_corr_to_1`` — the correlation-to-1 spoke: every pairwise
      correlation forced near 1.0, i.e. "everything crashes together." This
      is the tail the per-name-independent EV path structurally cannot see.
    """

    names: list[str]
    weights: dict[str, float]
    n_obs: int
    mean_abs_correlation: float
    empirical: dict = field(default_factory=dict)
    stress_corr_to_1: dict = field(default_factory=dict)
    #: Derived corr-to-1 headline: how much the tail worsens if every pairwise
    #: correlation goes to 1, vs the book's realized correlation. This — not
    #: the t-vs-Gaussian ``tail_amplification`` — is the right corr-to-1
    #: number: at near-perfect correlation the Gaussian copula already models
    #: the fully-coordinated crash, so t/Gaussian collapses toward 1 while the
    #: absolute CVaR still jumps. Populated by :func:`portfolio_correlation_tail`.
    stress_vs_empirical: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": "model",  # copula projection — reporting/stress overlay only
            "reporting_only": True,
            "feeds_ev": False,
            "names": self.names,
            "weights": self.weights,
            "n_obs": self.n_obs,
            "mean_abs_correlation": self.mean_abs_correlation,
            "empirical": self.empirical,
            "stress_corr_to_1": self.stress_corr_to_1,
            "stress_vs_empirical": self.stress_vs_empirical,
        }


def portfolio_correlation_tail(
    per_name_returns: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    confidence: float = 0.95,
    n_samples: int = 10_000,
    seed: int | None = CANONICAL_SEED,
    t_copula_df: float = 5.0,
    stress_correlation: float = 0.99,
) -> CorrelationTail:
    """Report the book's coordinated-drawdown tail via a Student-t copula.

    Builds the cross-name correlation matrix from ``per_name_returns`` and
    calls :func:`engine.portfolio_copula.portfolio_cvar_copula` twice: once at
    the empirical correlation, once at a near-1 stress correlation (the
    "everything crashes together" spoke). Reports Gaussian vs t-copula
    VaR/CVaR, the tail amplification, and the verdict for each.

    The portfolio return per sample is ``returns @ w`` with ``w`` the
    fractional dollar weights — so VaR/CVaR are in **portfolio-return**
    units (a fraction). ``tail_amplification`` (t_cvar / gaussian_cvar) is
    scale-free, so the verdict is unaffected by the weight normalization.

    Args:
        per_name_returns: ``{ticker: daily_return_array}``. Arrays MUST be
            date-aligned by the caller and share the same END date (the driver
            inner-joins on the trading calendar). This function tail-aligns to
            the shortest series by count, then drops any ROW where any name is
            non-finite (**listwise** deletion) — it never filters non-finite
            values per name, because per-name filtering would shift one
            column's dates relative to the others and understate the
            correlation. Names with fewer than 2 observations are dropped.
        weights: ``{ticker: signed_dollar_weight}`` — the book's per-name
            exposure. Signs are preserved (short-put books are long the
            underlying → positive weight). Normalized to sum-of-abs == 1 for
            the return-unit tail.
        confidence: VaR confidence (0.95 / 0.99).
        n_samples: Copula draw count.
        seed: PRNG seed (42 canonical).
        t_copula_df: Student-t dof (5 = moderate tail dependence).
        stress_correlation: Off-diagonal correlation for the stress spoke.

    Returns:
        A :class:`CorrelationTail`. If fewer than 2 names survive, or fewer
        than 2 joint (all-finite) rows remain after listwise deletion,
        ``empirical`` / ``stress_corr_to_1`` carry a ``skipped`` reason
        instead of tail numbers (a 1-name book has no cross-name tail).
    """

    def _skip(names_: list[str], reason: str) -> CorrelationTail:
        s = {"skipped": True, "reason": reason}
        return CorrelationTail(
            names=names_,
            weights={t: float(weights[t]) for t in names_},
            n_obs=0,
            mean_abs_correlation=float("nan"),
            empirical=dict(s),
            stress_corr_to_1=dict(s),
            stress_vs_empirical=dict(s),
        )

    # Keep names with a weight and a length>=2 array — but do NOT filter
    # non-finite values per name (that would de-align dates); rows are dropped
    # jointly below.
    usable = {
        t: np.asarray(r, dtype=float)
        for t, r in per_name_returns.items()
        if t in weights and len(np.asarray(r, dtype=float)) >= 2
    }
    names = sorted(usable.keys())
    if len(names) < 2:
        return _skip(names, "fewer_than_2_names_with_series")

    # Tail-align to the shortest series (shared END date assumed), then drop
    # any ROW that is non-finite for any name (listwise deletion keeps every
    # surviving row date-aligned across names).
    min_len = min(len(usable[t]) for t in names)
    mat = np.column_stack([usable[t][-min_len:] for t in names])  # (min_len, N)
    row_ok = np.all(np.isfinite(mat), axis=1)
    mat = mat[row_ok]
    if mat.shape[0] < 2:
        return _skip(names, "fewer_than_2_joint_finite_observations")

    corr = np.corrcoef(mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Fractional signed weights (sum of abs == 1) for return-unit tail.
    raw_w = np.asarray([float(weights[t]) for t in names], dtype=float)
    denom = np.sum(np.abs(raw_w))
    w = raw_w / denom if denom > 0 else np.full(len(names), 1.0 / len(names))
    weight_map = {t: float(wi) for t, wi in zip(names, w, strict=True)}

    marginals = [mat[:, i] for i in range(len(names))]

    empirical = portfolio_cvar_copula(
        marginals,
        corr,
        w,
        confidence=confidence,
        n_samples=n_samples,
        seed=seed,
        t_copula_df=t_copula_df,
    )

    # Correlation-to-1 stress spoke: force every pairwise correlation near 1.
    n = len(names)
    stress_corr = np.full((n, n), float(stress_correlation))
    np.fill_diagonal(stress_corr, 1.0)
    stress = portfolio_cvar_copula(
        marginals,
        stress_corr,
        w,
        confidence=confidence,
        n_samples=n_samples,
        seed=seed,
        t_copula_df=t_copula_df,
    )
    stress["stress_correlation"] = float(stress_correlation)

    # Mean absolute off-diagonal correlation — a scalar "how correlated is
    # this book" for the report header.
    iu = np.triu_indices(n, k=1)
    mean_abs_corr = float(np.mean(np.abs(corr[iu]))) if len(iu[0]) else float("nan")

    # Corr-to-1 headline: absolute tail worsening (magnitude), NOT the
    # t-vs-Gaussian ratio (which collapses toward 1 at near-perfect corr).
    def _ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("inf")

    stress_vs_empirical = {
        "t_cvar_multiple": _ratio(stress["t_cvar"], empirical["t_cvar"]),
        "t_var_multiple": _ratio(stress["t_var"], empirical["t_var"]),
        "empirical_t_cvar": float(empirical["t_cvar"]),
        "stress_t_cvar": float(stress["t_cvar"]),
        "note": "t_cvar_multiple = how many times worse the book's tail CVaR "
        "gets if every pairwise correlation goes to 1 vs its realized "
        "correlation. This is the corr-to-1 headline; the per-scenario "
        "tail_amplification is t-copula vs Gaussian within that scenario.",
    }

    return CorrelationTail(
        names=names,
        weights=weight_map,
        n_obs=int(mat.shape[0]),  # joint (all-finite) rows actually used
        mean_abs_correlation=mean_abs_corr,
        empirical=empirical,
        stress_corr_to_1=stress,
        stress_vs_empirical=stress_vs_empirical,
    )


# ---------------------------------------------------------------------------
# 5. Report assembly
# ---------------------------------------------------------------------------

#: Caveats carried on every report so consumers see them beside the numbers.
SIM_CAVEATS: dict[str, str] = {
    "E1": "~92% of the backtest NAV gain was equity-beta on assigned stock, "
    "not put-selection alpha.",
    "E3": "Backtest P&L was single-name-dominated (BKNG) — concentration risk "
    "the aggregate fan hides.",
    "E5": "Locked engine claims are parameter-IN-sample (HMM/POT-GPD/dealer "
    "clamp tuned on full history); this projection inherits that.",
    "D19": "ev_dollars nets only the entry-leg cost (~$1-4/ct optimistic); "
    "the realized backtest P&L this fan is built on shares that bias.",
    "D21": "forward-distribution samplers index trading-day bars against "
    "calendar DTE (~46% horizon over-dispersion in the underlying engine).",
    "bootstrap": "The strategy-return series is the book's DEPLOYED-capital "
    "return (marked only on days with an open position), resampled i.i.d.-in-"
    "blocks; it cannot invent regimes absent from the sampled window "
    "(survivorship: universe = current members).",
    "copula": "The correlation tail is a REPORTING/STRESS overlay only — it "
    "never feeds ev_dollars, a verdict, or the R7/R8 gate thresholds (§2).",
}


def build_sim_report(
    *,
    equity_curve: list[dict] | pd.DataFrame,
    initial_capital: float,
    realized_final_nav: float | None = None,
    per_name_returns: dict[str, np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
    n_simulations: int = 10_000,
    block_size: int = 21,
    seed: int | None = CANONICAL_SEED,
    label: str = "sim_forward_book",
) -> dict:
    """Assemble the full distributional report from a book's equity curve.

    Pure orchestration over the functions above — no I/O. Returns a JSON-safe
    dict with a ``model``-vs-``engine-measured`` split and the ``caveats``
    block. Persist it with the driver (``scripts/run_forward_sim.py``).

    **Reconciliation base.** The MC fan and the reconciliation are built on the
    **first equity mark** (``book_base_nav``), not the nominal
    ``initial_capital``. The strategy-return series is derived off the first
    mark, so the bootstrap compounds from that base to the realized final NAV —
    basing the fan there makes the reconciliation gap measure bootstrap
    fidelity, not the (book-specific) drift between nominal capital and the
    first marked NAV. Both figures are reported in ``engine_measured``.

    Args:
        equity_curve: The tracker's ``equity_curve`` list/DataFrame.
        initial_capital: Nominal book starting capital (reported for context;
            the fan/reconciliation base is the first equity mark).
        realized_final_nav: The deterministic backtest final NAV (for the
            reconciliation block). Defaults to the last equity-curve value.
        per_name_returns / weights: If both supplied, the copula correlation
            tail is included; otherwise it is omitted.
        n_simulations / block_size / seed: Bootstrap parameters.
        label: A tag echoed into the report header.

    Returns:
        ``{label, engine_measured, model, reconciliation, correlation_tail?,
        caveats}``.

    Raises:
        ValueError: if the strategy-return series is shorter than
            ``block_size`` (propagated from :func:`monte_carlo_bands`). Callers
            that may pass an empty/short book should catch this and emit an
            ``insufficient_history`` report (the driver does).
    """
    rets = strategy_returns_from_equity_curve(equity_curve)
    df = equity_curve if isinstance(equity_curve, pd.DataFrame) else pd.DataFrame(equity_curve)

    # Base the fan + reconciliation on the FIRST equity mark so the bootstrap
    # (returns derived off that mark) compounds to the realized final NAV.
    have_curve = not df.empty and "portfolio_value" in df.columns
    pv = (
        pd.to_numeric(df["portfolio_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if have_curve
        else np.asarray([], dtype=float)
    )
    book_base_nav = float(pv[0]) if len(pv) else float(initial_capital)
    if realized_final_nav is None:
        realized_final_nav = float(pv[-1]) if len(pv) else float(initial_capital)

    bands = monte_carlo_bands(
        rets,
        initial_capital=book_base_nav,
        n_simulations=n_simulations,
        block_size=block_size,
        seed=seed,
    )
    reconciliation = reconcile_with_backtest(
        bands, realized_final_nav=realized_final_nav, initial_capital=book_base_nav
    )

    report = {
        "label": label,
        "engine_measured": {
            "kind": "engine-measured",
            "initial_capital": float(initial_capital),
            "book_base_nav": book_base_nav,  # first equity mark = fan/reconciliation base
            "realized_final_nav": float(realized_final_nav),
            # Headline return vs the nominal capital the operator deployed.
            "realized_total_return": float(realized_final_nav) / float(initial_capital) - 1.0,
            # Return over the marked series (vs the first mark) — the base the
            # reconciliation actually uses.
            "realized_return_from_base": float(realized_final_nav) / book_base_nav - 1.0,
            "n_equity_marks": int(len(df)),
            "n_strategy_returns": int(len(rets)),
        },
        "model": bands.to_dict(),
        "reconciliation": reconciliation,
        "caveats": dict(SIM_CAVEATS),
    }

    if per_name_returns is not None and weights is not None:
        report["correlation_tail"] = portfolio_correlation_tail(
            per_name_returns, weights, seed=seed
        ).to_dict()

    return report
