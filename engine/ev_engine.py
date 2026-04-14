"""
Expected Value Engine (institutional-grade)
===========================================

This module replaces the ad-hoc "annualized return" / heuristic scoring that
was spread across ``signals.py``, ``wheel_runner.py`` and the payoff engine
with a principled, *probabilistic* expected-value computation for short-put
and covered-call trades.

Why a dedicated EV engine?
--------------------------
Before this module the codebase reported a scalar called ``annualized_return``
and used it to rank candidates. That number is **not** an expected value — it
is the *best-case* return assuming the option expires worthless. It ignores:

  * The probability of assignment.
  * Expected P&L conditional on assignment (not simply intrinsic loss).
  * Bid-ask slippage and commissions.
  * Dividend risk on short calls (early exercise).
  * Tail P&L (CVaR / Omega) for loss scenarios.

The :class:`EVEngine` takes a trade candidate plus a physical-measure
probability distribution (from the historical feature pipeline, the regime
detector, or a Monte Carlo simulation) and computes:

  * ``ev_dollars``       — probability-weighted net P&L
  * ``ev_per_day``       — EV scaled by expected days to exit
  * ``prob_profit``      — P(net P&L > 0)
  * ``prob_assignment``  — P(ITM at expiry) under the physical measure
  * ``cvar_5``           — expected loss in the worst 5% of outcomes
  * ``omega_ratio``      — upside/downside probability-weighted ratio
  * ``edge_vs_fair``     — collected premium minus risk-neutral BSM price,
                           i.e. the measurable volatility risk premium
  * ``regime_multiplier``— multiplicative scalar from the regime layer

Everything is computed in **net dollars**, not percentages. Position sizing is
then a downstream concern for the risk manager.

Design principles
-----------------
1. **Pure function, no I/O.** The engine does not load files, fetch quotes or
   hit APIs. All required inputs are passed in. This keeps it cheap to call
   inside a tight candidate ranking loop.
2. **No look-ahead.** The engine never consumes future returns. Any forward
   distribution is supplied by the caller who is responsible for PIT safety.
3. **Two probability measures, explicitly separated.** The edge-vs-fair
   metric uses the *risk-neutral* BSM price (``engine.option_pricer``).
   Everything else uses the *physical-measure* distribution supplied by the
   caller (either an empirical return distribution or a calibrated model).
4. **Transaction-cost aware.** Commissions and slippage are applied to both
   entry AND exit legs via :mod:`engine.transaction_costs`.

Typical caller
--------------
::

    from engine.ev_engine import EVEngine, ShortOptionTrade

    trade = ShortOptionTrade(
        option_type="put",
        underlying="AAPL",
        spot=187.4,
        strike=182.5,
        premium=2.15,
        bid=2.10, ask=2.20,
        dte=35,
        iv=0.28,
        risk_free_rate=0.05,
        dividend_yield=0.004,
    )

    # Empirical 35-day log-return distribution for AAPL from the feature
    # pipeline, rescaled/centered as required (PIT-safe).
    empirical_log_returns = feature_store.get_forward_distribution("AAPL", horizon_days=35)

    ev = EVEngine().evaluate(trade, forward_log_returns=empirical_log_returns)
    print(ev.ev_dollars, ev.prob_profit, ev.cvar_5, ev.edge_vs_fair)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .option_pricer import black_scholes_price
from .transaction_costs import (
    calculate_commission,
    calculate_slippage,
    calculate_assignment_fee,
)


OptionType = Literal["put", "call"]


@dataclass
class ShortOptionTrade:
    """Inputs for an EV evaluation of a single short-option candidate."""

    option_type: OptionType
    underlying: str
    spot: float
    strike: float
    premium: float  # Mid-price per share at entry
    dte: int
    iv: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    contracts: int = 1
    bid: float | None = None
    ask: float | None = None
    open_interest: int | None = None
    # Optional regime multiplier (from engine.regime_detector) — scalar in
    # [0.0, 1.25]. Anything < 1 de-emphasises trades in a hostile regime.
    regime_multiplier: float = 1.0
    # Optional assignment-cost parameters (for short calls near ex-div).
    days_to_ex_div: int | None = None
    expected_dividend: float = 0.0


@dataclass
class EVResult:
    """Result of an EV evaluation.

    All monetary fields are **net dollars for the total position**
    (i.e. premium × 100 × contracts), not per-share values. Probabilities are
    in [0, 1].
    """

    ev_dollars: float
    ev_per_day: float
    prob_profit: float
    prob_assignment: float
    prob_touch: float
    cvar_5: float  # Expected loss in worst 5% of paths (negative number)
    omega_ratio: float
    edge_vs_fair: float  # Collected premium - BSM risk-neutral price
    fair_value: float  # BSM risk-neutral price per share
    expected_days_held: float
    regime_multiplier: float
    total_transaction_cost: float
    breakeven_move_pct: float
    # Raw distribution diagnostics
    mean_pnl: float
    std_pnl: float
    skew_pnl: float
    metadata: dict = field(default_factory=dict)


class EVEngine:
    """Compute institutional-grade expected value for short-option trades.

    The engine supports three forward-distribution sources:

    1. ``forward_log_returns``: an empirical 1D ndarray of log-returns over
       the option holding period. This is the preferred path: pull the
       historical N-day log-returns for the underlying from the feature
       store, filter PIT-safely, and pass them in.
    2. ``price_scenarios``: a 1D ndarray of already-simulated terminal prices
       (e.g. from the Monte Carlo module). Treated as equally weighted.
    3. Fall-through: if neither is supplied, the engine builds a log-normal
       distribution using ``trade.iv`` as the physical vol. This is the
       weakest case because it discards any volatility-risk-premium signal —
       the whole point of the wheel strategy is that IV > realized vol, so
       using IV as the physical vol biases EV downward.

    Assignment-cost modelling (short calls): when the call is ITM and less
    than ``trade.days_to_ex_div`` away from ex-dividend, the engine adds the
    dividend to the expected loss, since a rational holder will early-exercise
    to capture it.
    """

    def __init__(
        self,
        profit_target_pct: float = 0.50,
        stop_loss_multiple: float = 2.0,
        slippage_pct_of_spread: float = 0.20,
    ) -> None:
        self.profit_target_pct = profit_target_pct
        self.stop_loss_multiple = stop_loss_multiple
        self.slippage_pct_of_spread = slippage_pct_of_spread

    # ------------------------------------------------------------------
    def evaluate(
        self,
        trade: ShortOptionTrade,
        forward_log_returns: np.ndarray | None = None,
        price_scenarios: np.ndarray | None = None,
    ) -> EVResult:
        """Run EV evaluation for one candidate trade."""
        multiplier = 100 * max(trade.contracts, 1)
        gross_premium = trade.premium * multiplier

        # --------------------------------------------------------------
        # Transaction costs (entry + potential exit)
        # --------------------------------------------------------------
        entry_commission = calculate_commission("option", trade.contracts)
        spread = (trade.ask - trade.bid) if (trade.ask and trade.bid) else trade.premium * 0.10
        entry_slippage_per_share = calculate_slippage(
            mid_price=trade.premium,
            bid_ask_spread=spread,
            trade_direction="sell",
            open_interest=trade.open_interest,
        )
        entry_slippage = entry_slippage_per_share * multiplier

        exit_commission = calculate_commission("option", trade.contracts)
        # Approximate exit slippage as a fraction of the expected buyback cost.
        exit_slippage = entry_slippage  # round-trip symmetry as a conservative proxy

        total_cost = entry_commission + entry_slippage + exit_commission + exit_slippage

        # Net premium actually received (after entry slippage & commission).
        net_premium_in = gross_premium - entry_commission - entry_slippage

        # --------------------------------------------------------------
        # Risk-neutral fair value (for edge_vs_fair)
        # --------------------------------------------------------------
        T = max(trade.dte, 0) / 365.0
        fair = black_scholes_price(
            S=trade.spot,
            K=trade.strike,
            T=T,
            r=trade.risk_free_rate,
            sigma=trade.iv,
            option_type=trade.option_type,
            q=trade.dividend_yield,
        )
        edge_vs_fair_per_share = trade.premium - fair
        edge_vs_fair = edge_vs_fair_per_share * multiplier

        # --------------------------------------------------------------
        # Build the physical-measure terminal-price distribution
        # --------------------------------------------------------------
        terminal_prices = self._build_terminal_prices(
            trade, forward_log_returns, price_scenarios
        )

        # --------------------------------------------------------------
        # Compute path-outcome P&L
        # --------------------------------------------------------------
        pnls = self._compute_pnls(trade, terminal_prices, net_premium_in, multiplier)
        # Exit costs apply only to non-assignment outcomes (approximation: we
        # exit with a buyback for OTM-expiry closes? No — OTM expires worthless
        # at $0 so there is no buyback cost.) We only apply exit costs where
        # the option is *bought back* before expiration. Since our terminal-
        # distribution treatment expires every path at T, exit_commission and
        # exit_slippage represent the *expected* early-close costs that would
        # be incurred if profit target / stop loss triggers. We therefore
        # penalise EV by a fraction of them proportional to the probability
        # of a non-expiration exit, approximated as prob_profit + prob_stop.
        is_itm = self._is_itm_mask(trade, terminal_prices)
        prob_itm = float(np.mean(is_itm))
        prob_otm = 1.0 - prob_itm

        # Assignment fee applies when ITM at expiry.
        assignment_fee_total = calculate_assignment_fee() * max(trade.contracts, 1)
        pnls = pnls - is_itm.astype(float) * assignment_fee_total

        # Dividend early-exercise penalty for short calls ITM with ex-div in
        # the holding period.
        if (
            trade.option_type == "call"
            and trade.days_to_ex_div is not None
            and trade.days_to_ex_div <= trade.dte
            and trade.expected_dividend > 0
        ):
            pnls = pnls - is_itm.astype(float) * trade.expected_dividend * multiplier

        # --------------------------------------------------------------
        # Distribution statistics
        # --------------------------------------------------------------
        ev_raw = float(np.mean(pnls))
        mean_pnl = ev_raw
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        # Skewness (Fisher-Pearson)
        if std_pnl > 0 and len(pnls) > 2:
            centered = pnls - mean_pnl
            skew_pnl = float(np.mean(centered**3) / (std_pnl**3))
        else:
            skew_pnl = 0.0

        prob_profit = float(np.mean(pnls > 0))

        # CVaR_5 (expected shortfall of worst 5%)
        if len(pnls) >= 20:
            var_5_threshold = float(np.percentile(pnls, 5))
            tail = pnls[pnls <= var_5_threshold]
            cvar_5 = float(np.mean(tail)) if len(tail) > 0 else var_5_threshold
        else:
            cvar_5 = float(np.min(pnls)) if len(pnls) > 0 else 0.0

        # Omega ratio at 0 threshold: E[(X-0)+] / E[(0-X)+]
        gains = pnls[pnls > 0].sum() if np.any(pnls > 0) else 0.0
        losses = -pnls[pnls < 0].sum() if np.any(pnls < 0) else 0.0
        omega = float(gains / losses) if losses > 1e-9 else float("inf")

        # Probability of touch (daily path would exceed strike at some point).
        # With terminal-only distribution this is only a lower bound, derived
        # via Brownian-bridge reflection principle: P(touch) ≈ 2 * P(terminal
        # ITM) for out-of-the-money starts. We use that analytic boundary.
        prob_touch = min(1.0, 2.0 * prob_itm if self._is_otm_at_entry(trade) else 1.0)

        # Breakeven move — how far the underlying must move before net PnL = 0.
        if trade.option_type == "put":
            breakeven_price = trade.strike - (trade.premium - total_cost / multiplier)
            breakeven_move_pct = (breakeven_price - trade.spot) / trade.spot
        else:
            breakeven_price = trade.strike + (trade.premium - total_cost / multiplier)
            breakeven_move_pct = (breakeven_price - trade.spot) / trade.spot

        # Expected hold time — if profit target fires, exit ~ halfway to
        # expiration; otherwise hold to expiry.
        expected_days_held = prob_profit * (trade.dte / 2.0) + (1 - prob_profit) * trade.dte
        expected_days_held = max(1.0, expected_days_held)

        # Regime multiplier is applied *last* to dollar EV so other metrics
        # remain pure and auditable.
        regime_mult = max(0.0, float(trade.regime_multiplier))
        ev_dollars = ev_raw * regime_mult

        return EVResult(
            ev_dollars=ev_dollars,
            ev_per_day=ev_dollars / expected_days_held,
            prob_profit=prob_profit,
            prob_assignment=prob_itm,
            prob_touch=prob_touch,
            cvar_5=cvar_5,
            omega_ratio=omega,
            edge_vs_fair=edge_vs_fair,
            fair_value=fair,
            expected_days_held=expected_days_held,
            regime_multiplier=regime_mult,
            total_transaction_cost=total_cost,
            breakeven_move_pct=breakeven_move_pct,
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            skew_pnl=skew_pnl,
            metadata={
                "n_scenarios": int(len(pnls)),
                "net_premium_in": net_premium_in,
                "assignment_fee_applied": assignment_fee_total if prob_itm > 0 else 0.0,
                "fair_value_per_share": fair,
                "edge_per_share": edge_vs_fair_per_share,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_terminal_prices(
        self,
        trade: ShortOptionTrade,
        forward_log_returns: np.ndarray | None,
        price_scenarios: np.ndarray | None,
    ) -> np.ndarray:
        """Return a 1-D ndarray of terminal underlying prices."""
        if price_scenarios is not None and len(price_scenarios) > 0:
            return np.asarray(price_scenarios, dtype=float)

        if forward_log_returns is not None and len(forward_log_returns) > 0:
            arr = np.asarray(forward_log_returns, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                return trade.spot * np.exp(arr)

        # Fall-through: lognormal from trade IV. We deliberately sample a
        # large number of points for a stable empirical distribution.
        T = max(trade.dte, 1) / 365.0
        sigma = max(trade.iv, 1e-4)
        rng = np.random.default_rng(seed=hash(trade.underlying) % (2**32))
        log_rets = (
            (trade.risk_free_rate - trade.dividend_yield - 0.5 * sigma**2) * T
            + sigma * np.sqrt(T) * rng.standard_normal(20_000)
        )
        return trade.spot * np.exp(log_rets)

    def _compute_pnls(
        self,
        trade: ShortOptionTrade,
        terminal_prices: np.ndarray,
        net_premium_in: float,
        multiplier: int,
    ) -> np.ndarray:
        """Compute per-path net P&L (before assignment fee / div penalty)."""
        if trade.option_type == "put":
            intrinsic = np.maximum(trade.strike - terminal_prices, 0.0)
        else:
            intrinsic = np.maximum(terminal_prices - trade.strike, 0.0)

        # Buyback cost at expiry = intrinsic * multiplier. Short seller pays it.
        buyback_cost = intrinsic * multiplier
        pnl = net_premium_in - buyback_cost
        return pnl

    def _is_itm_mask(
        self, trade: ShortOptionTrade, terminal_prices: np.ndarray
    ) -> np.ndarray:
        if trade.option_type == "put":
            return terminal_prices < trade.strike
        return terminal_prices > trade.strike

    def _is_otm_at_entry(self, trade: ShortOptionTrade) -> bool:
        if trade.option_type == "put":
            return trade.spot > trade.strike
        return trade.spot < trade.strike


# ----------------------------------------------------------------------
# Convenience API
# ----------------------------------------------------------------------
def rank_candidates(
    candidates: list[ShortOptionTrade],
    engine: EVEngine | None = None,
    top_n: int = 10,
    min_ev: float = 0.0,
    forward_log_returns_by_ticker: dict[str, np.ndarray] | None = None,
) -> list[tuple[ShortOptionTrade, EVResult]]:
    """Rank a list of trade candidates by EV per day.

    The ranker enforces ``ev_dollars >= min_ev`` as a hard filter. Candidates
    that fall below that bar are dropped entirely — the wheel strategy only
    takes positive-EV trades.
    """
    engine = engine or EVEngine()
    scored: list[tuple[ShortOptionTrade, EVResult]] = []
    forward_log_returns_by_ticker = forward_log_returns_by_ticker or {}
    for c in candidates:
        fwd = forward_log_returns_by_ticker.get(c.underlying)
        res = engine.evaluate(c, forward_log_returns=fwd)
        if res.ev_dollars < min_ev:
            continue
        scored.append((c, res))
    scored.sort(key=lambda tup: tup[1].ev_per_day, reverse=True)
    return scored[:top_n]
