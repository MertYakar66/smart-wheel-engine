"""
Wheel Strategy Position Tracker
Manages the full lifecycle: Short Put → Stock Assignment → Covered Call → Exit
"""

import json
from dataclasses import dataclass, field, fields
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .ev_engine import EVEngine, ShortOptionTrade
from .option_pricer import black_scholes_price
from .transaction_costs import (
    calculate_assignment_costs,
    calculate_reg_t_margin_short_put,
    calculate_total_entry_cost,
    calculate_total_exit_cost,
)

# ----------------------------------------------------------------------
# Schema + helpers for WheelTracker.suggest_rolls / suggest_call_rolls
# ----------------------------------------------------------------------
# Output columns pinned at module scope so the empty-result path returns
# a same-shaped (zero-row) DataFrame for stable downstream consumption.
_ROLL_COLUMNS = [
    "new_strike",
    "new_expiry",
    "new_dte",
    "target_delta",
    "new_premium",
    "buyback_cost",
    "net_credit_debit",
    "new_ev_dollars",
    "roll_ev",
    "hold_ev",
    "prob_otm",
    "recommend",
]


def _solve_put_strike(
    spot: float, T: float, r: float, q: float, iv: float, target_delta: float
) -> float | None:
    """Solve K such that the BSM put delta equals ``-target_delta``.

    Mirrors the in-line solver in
    :meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev` so
    :meth:`WheelTracker.suggest_rolls` enumerates roll candidates with
    the same convention as the ranker. Returns ``None`` when no
    solution exists in ``[spot*0.5, spot*0.99]``.
    """
    if T <= 0 or iv <= 0 or spot <= 0 or not (0.0 < target_delta < 1.0):
        return None

    def err(K: float) -> float:
        if K <= 0:
            return 1.0
        d1 = (np.log(spot / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
        put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1.0)
        return put_delta + target_delta

    try:
        return brentq(err, spot * 0.5, spot * 0.99, xtol=1e-2)
    except (ValueError, RuntimeError):
        return None


def _solve_call_strike(
    spot: float, T: float, r: float, q: float, iv: float, target_delta: float
) -> float | None:
    """Solve K such that the BSM call delta equals ``target_delta``.

    The call-leg parallel of :func:`_solve_put_strike`, so
    :meth:`WheelTracker.suggest_call_rolls` enumerates roll candidates
    with the same delta convention as the put roller. Returns ``None``
    when no solution exists in ``[spot*1.01, spot*2.0]`` -- the OTM
    region a covered call is sold into (strike above spot).
    """
    if T <= 0 or iv <= 0 or spot <= 0 or not (0.0 < target_delta < 1.0):
        return None

    def err(K: float) -> float:
        if K <= 0:
            return 1.0
        d1 = (np.log(spot / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
        call_delta = np.exp(-q * T) * norm.cdf(d1)
        return call_delta - target_delta

    try:
        return brentq(err, spot * 1.01, spot * 2.0, xtol=1e-2)
    except (ValueError, RuntimeError):
        return None


class PositionState(Enum):
    """State machine for Wheel positions"""

    NO_POSITION = "no_position"
    SHORT_PUT = "short_put"
    STOCK_OWNED = "stock_owned"
    COVERED_CALL = "covered_call"


@dataclass
class WheelPosition:
    """
    Tracks a single Wheel position through its lifecycle.
    All prices in dollars, P&L in dollars (not per-share).
    """

    ticker: str
    state: PositionState
    entry_date: date

    # Short put phase
    put_strike: float | None = None
    put_premium: float | None = None  # Per share
    put_entry_date: date | None = None
    put_dte_at_entry: int | None = None
    put_entry_iv: float | None = None
    put_expiration_date: date | None = None

    # Stock ownership phase
    stock_shares: int = 0
    stock_basis: float | None = None  # Cost per share
    stock_acquisition_date: date | None = None

    # Covered call phase
    call_strike: float | None = None
    call_premium: float | None = None  # Per share
    call_entry_date: date | None = None
    call_dte_at_entry: int | None = None
    call_entry_iv: float | None = None
    call_expiration_date: date | None = None

    # P&L tracking (cumulative, in dollars)
    realized_pnl: float = 0.0
    transaction_costs: float = 0.0

    # Metadata
    notes: list[str] = field(default_factory=list)

    # The six ``date``-typed fields. No type annotation → this is a
    # plain class constant, not a dataclass field. Used by the
    # (de)serialisation methods below.
    _DATE_FIELDS = (
        "entry_date",
        "put_entry_date",
        "put_expiration_date",
        "stock_acquisition_date",
        "call_entry_date",
        "call_expiration_date",
    )

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict.

        ``state`` becomes its enum ``.value`` string; the six ``date``
        fields become ISO-8601 strings (or ``None``); ``notes`` is
        copied; everything else is already JSON-safe. Inverse of
        :meth:`from_dict`.
        """
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "state":
                out[f.name] = value.value
            elif f.name in self._DATE_FIELDS:
                out[f.name] = value.isoformat() if value is not None else None
            elif f.name == "notes":
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "WheelPosition":
        """Rebuild a :class:`WheelPosition` from a :meth:`to_dict` dict.

        Unknown keys are ignored (forward-compatible with a newer
        schema); missing optional keys fall back to the dataclass
        defaults.
        """
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        kwargs["state"] = PositionState(kwargs["state"])
        for name in cls._DATE_FIELDS:
            if kwargs.get(name) is not None:
                kwargs[name] = date.fromisoformat(kwargs[name])
        return cls(**kwargs)


# ----------------------------------------------------------------------
# Serialisation helpers for WheelTracker.to_dict / from_dict
# ----------------------------------------------------------------------
# ``closed_positions`` and ``equity_curve`` are append-only record
# dicts with a single known builder each (``_finalize_position`` and
# ``mark_to_market``). Their only ``date``-typed values live under these
# fixed keys, so they round-trip with explicit per-key conversion — no
# guessing which strings are dates.
_CLOSED_POSITION_DATE_KEYS = ("entry_date", "exit_date")
_EQUITY_CURVE_DATE_KEYS = ("date",)


def _record_to_jsonable(record: dict, date_keys: tuple[str, ...]) -> dict:
    """Copy a record dict, converting the named ``date`` keys to ISO strings."""
    out = dict(record)
    for key in date_keys:
        value = out.get(key)
        if isinstance(value, date):
            out[key] = value.isoformat()
    return out


def _record_from_jsonable(record: dict, date_keys: tuple[str, ...]) -> dict:
    """Inverse of :func:`_record_to_jsonable` — ISO strings back to ``date``."""
    out = dict(record)
    for key in date_keys:
        value = out.get(key)
        if isinstance(value, str):
            out[key] = date.fromisoformat(value)
    return out


class WheelTracker:
    """
    Portfolio-level tracker for all Wheel positions.
    Handles state transitions and P&L accounting.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        require_ev_authority: bool = False,
        connector: Any | None = None,
    ):
        """
        Args:
            initial_capital: Starting cash.
            require_ev_authority: When True, ``open_short_put`` and
                ``open_covered_call`` REQUIRE an ``ev_authority_token``
                argument produced by :meth:`issue_ev_authority_token`
                below, which itself should only be called from within
                :meth:`WheelRunner.rank_candidates_by_ev` when the
                candidate has passed the EV ranker. This is the hard
                launch-gate that prevents heuristic / manual / webhook
                paths from entering a position bypassing the EV engine.
                Default False for backwards compatibility with tests
                and research usage; production/live tracker instances
                should set it True.
            connector: Optional data connector
                (``MarketDataConnector`` / ``ThetaConnector``-style).
                Used only by :meth:`suggest_rolls` to resolve
                ``get_risk_free_rate(as_of)`` and ``get_ohlcv(ticker)``
                when those values are not passed explicitly. Default
                ``None`` keeps the tracker dependency-free.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, WheelPosition] = {}
        self.closed_positions: list[dict] = []
        self.equity_curve: list[dict] = []
        self.require_ev_authority = require_ev_authority
        self._ev_authority_tokens: set[str] = set()
        self._ev_authority_log: list[dict] = []
        self.connector = connector

    # ------------------------------------------------------------------
    # EV authority token issuance (audit launch-gate)
    # ------------------------------------------------------------------
    def issue_ev_authority_token(self, ev_row: dict) -> str:
        """Issue a one-time token proving a candidate passed EV ranking.

        The token is a SHA-256 hex digest of the canonicalised EV row
        (ticker, strike, premium, dte, ev_dollars, prob_profit,
        distribution_source). The same row always produces the same
        token; downstream :meth:`open_short_put` verifies it against
        the accepted-token set.

        Single-use: once consumed, the token is removed. This prevents
        a captured token being re-played to enter multiple positions.
        """
        import hashlib
        import json as _json

        canonical = {
            "ticker": str(ev_row.get("ticker", "")),
            "strike": float(ev_row.get("strike", 0) or 0),
            "premium": float(ev_row.get("premium", 0) or 0),
            "dte": int(ev_row.get("dte", 0) or 0),
            "ev_dollars": float(ev_row.get("ev_dollars", 0) or 0),
            "prob_profit": float(ev_row.get("prob_profit", 0) or 0),
            "distribution_source": str(ev_row.get("distribution_source", "")),
        }
        token = hashlib.sha256(_json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        self._ev_authority_tokens.add(token)
        self._ev_authority_log.append({"action": "issue", "token": token, "row": canonical})
        return token

    def _consume_ev_authority_token(self, token: str | None, ticker: str) -> bool:
        """Verify and consume an EV authority token. Single-use."""
        if not token:
            return False
        if token not in self._ev_authority_tokens:
            self._ev_authority_log.append(
                {"action": "reject", "reason": "unknown_token", "ticker": ticker}
            )
            return False
        self._ev_authority_tokens.discard(token)
        self._ev_authority_log.append({"action": "consume", "token": token, "ticker": ticker})
        return True

    def open_short_put(
        self,
        ticker: str,
        strike: float,
        premium: float,
        entry_date: date,
        expiration_date: date,  # CHANGED: explicit date instead of dte
        iv: float,
        ev_authority_token: str | None = None,
    ) -> bool:
        """
        Enter short put position.

        Args:
            ticker: Stock ticker
            strike: Put strike price
            premium: Premium collected per share
            entry_date: Trade entry date
            expiration_date: Explicit calendar expiration date
            iv: Implied volatility at entry
            ev_authority_token: When the tracker was created with
                ``require_ev_authority=True``, this must be a valid
                single-use token from :meth:`issue_ev_authority_token`.
                Trades without a valid token are rejected outright —
                this is the launch-gate that prevents heuristic /
                manual / webhook paths from bypassing the EV engine.

        Returns:
            True if position opened successfully
        """
        # Launch-gate: reject trades without EV authority in strict mode.
        if self.require_ev_authority:
            if not self._consume_ev_authority_token(ev_authority_token, ticker):
                return False

        # Check if already have position in this ticker
        if ticker in self.positions:
            return False

        # Check buying power using proper Reg-T margin calculation
        underlying_price = strike  # Approximate if not provided
        margin_required = calculate_reg_t_margin_short_put(
            strike=strike, underlying_price=underlying_price, premium=premium
        )
        if self.cash < margin_required:
            return False

        # Calculate entry costs (commission + slippage) and net premium collected
        cost_details = calculate_total_entry_cost(
            premium_per_share=premium,
            bid_ask_spread=None,  # Will use fallback
            bid=None,
            ask=None,
            trade_type="option",
        )

        premium_collected = cost_details["net_premium_collected"]
        cost_details["commission"]
        cost_details["slippage"]

        # Credit net premium to cash (per-contract)
        self.cash += premium_collected

        # Create position (store explicit expiration date and derived DTE)
        derived_dte = (expiration_date - entry_date).days
        self.positions[ticker] = WheelPosition(
            ticker=ticker,
            state=PositionState.SHORT_PUT,
            entry_date=entry_date,
            put_strike=strike,
            put_premium=premium,
            put_entry_date=entry_date,
            put_dte_at_entry=derived_dte,  # Derived from dates
            put_entry_iv=iv,
            put_expiration_date=expiration_date,  # NEW: store explicit date
            realized_pnl=cost_details["gross_premium"],
            transaction_costs=cost_details["total_cost"],
        )

        self.positions[ticker].notes.append(
            f"Sold {derived_dte}d {strike}P for ${premium:.2f} premium"
        )

        return True

    def close_short_put(
        self, ticker: str, buyback_price: float, exit_date: date, reason: str = "early_exit"
    ) -> dict | None:
        """
        Buy back short put (early exit before expiration).

        Args:
            ticker: Stock ticker
            buyback_price: Price paid to buy back put (per share)
            exit_date: Exit date
            reason: Exit reason (profit_target, stop_loss, time_decay)

        Returns:
            Closed position summary dict
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return None

        # Calculate exit costs (buyback + slippage + commission)
        cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option",
        )

        # AUDIT-VIII P1: ``realized_pnl`` is a running gross-P&L accumulator
        # that already holds the gross premium received on open (and any
        # prior roll credit/debit). Subtract the buyback cost here rather
        # than overwriting, so that positions which were rolled one or more
        # times don't silently lose their earlier leg P&L when the final
        # close wipes the accumulator. Covered calls already use ``+=``
        # (see close_covered_call); this aligns the put path.
        pos.realized_pnl -= cost_details["gross_buyback_cost"]
        pos.transaction_costs += cost_details["total_cost"]

        # Deduct total buyback cash (includes slippage and commission)
        self.cash -= cost_details["total_buyback_cost"]

        # Record closed position
        closed = self._finalize_position(pos, exit_date, reason)
        del self.positions[ticker]

        return closed

    def handle_put_assignment(self, ticker: str, assignment_date: date, stock_price: float) -> bool:
        """
        Handle put assignment: acquire 100 shares at strike price.

        Args:
            ticker: Stock ticker
            assignment_date: Assignment date
            stock_price: Current stock price (for unrealized P&L tracking)

        Returns:
            True if assignment handled successfully
        """
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return False

        # Acquire stock at strike price (include assignment fee)
        assignment_details = calculate_assignment_costs(strike_price=pos.put_strike, shares=100)

        stock_cost = assignment_details["stock_cost"]
        assignment_fee = assignment_details["assignment_fee"]

        if self.cash < assignment_details["total_cash_required"]:
            # In reality, broker would margin call or auto-liquidate
            # For simulation, we'll allow it but flag
            pos.notes.append(
                f"WARNING: Assignment required ${stock_cost:.2f}, only ${self.cash:.2f} available"
            )

        self.cash -= assignment_details["total_cash_required"]

        # Update position state.
        # AUDIT-VIII P1.2: ``stock_basis`` is now the **raw cash cost per
        # share** (= put strike), not a "net basis after premium credit".
        # Rationale: the premium received on the short put is already
        # accumulated into ``realized_pnl`` by ``open_short_put``, and
        # the assignment fee is already accumulated into
        # ``transaction_costs``. The prior ``stock_basis = strike -
        # premium + fee/100`` fed those same two corrections back into
        # ``stock_pnl = (call_strike - basis) * 100`` on call assignment,
        # causing a double-count that silently overstated wheel-cycle
        # P&L by exactly ``(premium * 100 - fee)``. Keeping basis at
        # raw cash cost makes ``stock_pnl`` a pure stock delta and
        # keeps the three ledgers (realized_pnl, transaction_costs,
        # stock_basis) orthogonal.
        pos.state = PositionState.STOCK_OWNED
        pos.stock_shares = 100
        premium_received = pos.put_premium if pos.put_premium else 0.0
        pos.stock_basis = pos.put_strike
        pos.stock_acquisition_date = assignment_date
        pos.transaction_costs += assignment_fee

        pos.notes.append(
            f"Assigned: Bought 100 shares at ${pos.put_strike:.2f} "
            f"(basis: ${pos.stock_basis:.2f}, premium ${premium_received:.2f}/sh "
            f"already credited to realized_pnl, market: ${stock_price:.2f})"
        )

        return True

    def handle_partial_put_assignment(
        self, ticker: str, assignment_date: date, stock_price: float, shares_assigned: int
    ) -> bool:
        """
        Handle partial put assignment: acquire fewer than 100 shares at strike price.

        Args:
            ticker: Stock ticker
            assignment_date: Assignment date
            stock_price: Current stock price (for unrealized P&L tracking)
            shares_assigned: Number of shares assigned (1-100)

        Returns:
            True if assignment handled successfully
        """
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return False

        # Validate shares_assigned range
        if shares_assigned < 1 or shares_assigned > 100:
            return False

        # Calculate assignment costs for the partial lot
        assignment_details = calculate_assignment_costs(
            strike_price=pos.put_strike, shares=shares_assigned
        )

        stock_cost = assignment_details["stock_cost"]
        assignment_fee = assignment_details["assignment_fee"]

        if self.cash < assignment_details["total_cash_required"]:
            pos.notes.append(
                f"WARNING: Partial assignment required ${stock_cost:.2f}, "
                f"only ${self.cash:.2f} available"
            )

        self.cash -= assignment_details["total_cash_required"]

        # Transition to STOCK_OWNED with partial share count.
        # AUDIT-VIII P1.2: see handle_put_assignment — basis is raw cash
        # cost; premium and fee flow through realized_pnl /
        # transaction_costs respectively.
        pos.state = PositionState.STOCK_OWNED
        pos.stock_shares = shares_assigned
        premium_received = pos.put_premium if pos.put_premium else 0.0
        pos.stock_basis = pos.put_strike
        pos.stock_acquisition_date = assignment_date
        pos.transaction_costs += assignment_fee

        pos.notes.append(
            f"Partial assignment: Bought {shares_assigned} shares at ${pos.put_strike:.2f} "
            f"(basis: ${pos.stock_basis:.2f}, premium ${premium_received:.2f}/sh "
            f"already credited to realized_pnl, market: ${stock_price:.2f})"
        )

        return True

    def roll_put(
        self,
        ticker: str,
        roll_date: date,
        new_strike: float,
        new_premium: float,
        new_expiration: date,
        new_iv: float,
        buyback_price: float,
    ) -> dict | None:
        """
        Roll a short put: close current put and open a new one atomically.

        Args:
            ticker: Stock ticker
            roll_date: Date of the roll
            new_strike: New put strike price
            new_premium: New premium collected per share
            new_expiration: New expiration date
            new_iv: Implied volatility of new put
            buyback_price: Price paid to buy back current put (per share)

        Returns:
            Dict with roll details (net_credit_debit, old_strike, new_strike),
            or None if roll fails
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return None

        old_strike = pos.put_strike
        old_premium = pos.put_premium

        # Calculate exit costs for buying back the current put
        exit_cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option",
        )

        # Calculate entry costs for the new put
        entry_cost_details = calculate_total_entry_cost(
            premium_per_share=new_premium,
            bid_ask_spread=new_premium * 0.10,
            trade_type="option",
        )

        # Apply close: deduct buyback cost from cash
        self.cash -= exit_cost_details["total_buyback_cost"]

        # AUDIT-VIII P1: accumulate running gross-P&L rather than overwrite.
        # The prior ``realized_pnl = gross_close_pnl`` pattern silently
        # dropped all leg P&L on the *second* roll (first roll's credit
        # was lost). Subtracting only the new buyback leaves any prior
        # credit intact; the new premium is credited below.
        pos.realized_pnl -= exit_cost_details["gross_buyback_cost"]
        pos.transaction_costs += exit_cost_details["total_cost"]

        # Apply open: credit new premium to cash
        self.cash += entry_cost_details["net_premium_collected"]

        # Update position fields to the new put
        derived_dte = (new_expiration - roll_date).days
        pos.put_strike = new_strike
        pos.put_premium = new_premium
        pos.put_entry_date = roll_date
        pos.put_dte_at_entry = derived_dte
        pos.put_entry_iv = new_iv
        pos.put_expiration_date = new_expiration
        pos.realized_pnl += entry_cost_details["gross_premium"]
        pos.transaction_costs += entry_cost_details["total_cost"]

        # Net credit/debit of the roll (positive = credit)
        net_credit_debit = (
            entry_cost_details["net_premium_collected"] - exit_cost_details["total_buyback_cost"]
        )

        pos.notes.append(
            f"Rolled put: closed ${old_strike:.2f}P at ${buyback_price:.2f}, "
            f"opened ${new_strike:.2f}P for ${new_premium:.2f} premium "
            f"(net {'credit' if net_credit_debit >= 0 else 'debit'}: "
            f"${abs(net_credit_debit):.2f})"
        )

        return {
            "old_strike": old_strike,
            "new_strike": new_strike,
            "old_premium": old_premium,
            "new_premium": new_premium,
            "buyback_price": buyback_price,
            "net_credit_debit": net_credit_debit,
            "roll_date": roll_date,
            "new_expiration": new_expiration,
        }

    def roll_call(
        self,
        ticker: str,
        roll_date: date,
        new_strike: float,
        new_premium: float,
        new_expiration: date,
        new_iv: float,
        buyback_price: float,
    ) -> dict | None:
        """
        Roll a covered call: close current call and open a new one atomically.

        Args:
            ticker: Stock ticker
            roll_date: Date of the roll
            new_strike: New call strike price
            new_premium: New premium collected per share
            new_expiration: New expiration date
            new_iv: Implied volatility of new call
            buyback_price: Price paid to buy back current call (per share)

        Returns:
            Dict with roll details (net_credit_debit, old_strike, new_strike),
            or None if roll fails
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None

        old_strike = pos.call_strike
        old_premium = pos.call_premium

        # Calculate exit costs for buying back the current call
        exit_cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option",
        )

        # Calculate entry costs for the new call
        entry_cost_details = calculate_total_entry_cost(
            premium_per_share=new_premium,
            bid_ask_spread=new_premium * 0.10,
            trade_type="option",
        )

        # Apply close: deduct buyback cost from cash
        self.cash -= exit_cost_details["total_buyback_cost"]

        # AUDIT-VIII P1: the original ``open_covered_call`` already
        # added ``old_premium * 100`` to ``realized_pnl``. Adding it
        # again here as ``old_premium*100 - buyback`` double-counted
        # the entry premium on every call roll. Only subtract the new
        # buyback so the accumulator preserves prior leg P&L.
        pos.realized_pnl -= exit_cost_details["gross_buyback_cost"]
        pos.transaction_costs += exit_cost_details["total_cost"]

        # Apply open: credit new premium to cash
        self.cash += entry_cost_details["net_premium_collected"]

        # Update position fields to the new call
        derived_dte = (new_expiration - roll_date).days
        pos.call_strike = new_strike
        pos.call_premium = new_premium
        pos.call_entry_date = roll_date
        pos.call_dte_at_entry = derived_dte
        pos.call_entry_iv = new_iv
        pos.call_expiration_date = new_expiration
        pos.realized_pnl += entry_cost_details["gross_premium"]
        pos.transaction_costs += entry_cost_details["total_cost"]

        # Net credit/debit of the roll (positive = credit)
        net_credit_debit = (
            entry_cost_details["net_premium_collected"] - exit_cost_details["total_buyback_cost"]
        )

        pos.notes.append(
            f"Rolled call: closed ${old_strike:.2f}C at ${buyback_price:.2f}, "
            f"opened ${new_strike:.2f}C for ${new_premium:.2f} premium "
            f"(net {'credit' if net_credit_debit >= 0 else 'debit'}: "
            f"${abs(net_credit_debit):.2f})"
        )

        return {
            "old_strike": old_strike,
            "new_strike": new_strike,
            "old_premium": old_premium,
            "new_premium": new_premium,
            "buyback_price": buyback_price,
            "net_credit_debit": net_credit_debit,
            "roll_date": roll_date,
            "new_expiration": new_expiration,
        }

    def handle_partial_call_assignment(
        self, ticker: str, assignment_date: date, shares_called: int
    ) -> dict | None:
        """
        Handle partial call assignment: sell some shares at call strike.

        Args:
            ticker: Stock ticker
            assignment_date: Assignment date
            shares_called: Number of shares called away

        Returns:
            Dict with assignment details, or None if invalid
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None

        if shares_called < 1 or shares_called > pos.stock_shares:
            return None

        # Capture strike before any state changes
        call_strike = pos.call_strike

        # Sell shares at call strike (account for assignment fee)
        stock_proceeds = call_strike * shares_called
        assignment_details = calculate_assignment_costs(
            strike_price=call_strike, shares=shares_called
        )
        assignment_fee = assignment_details["assignment_fee"]

        self.cash += stock_proceeds - assignment_fee

        # Calculate stock P&L for shares sold
        stock_pnl = (call_strike - pos.stock_basis) * shares_called
        pos.realized_pnl += stock_pnl
        pos.transaction_costs += assignment_fee

        remaining_shares = pos.stock_shares - shares_called

        if remaining_shares > 0:
            # Still have shares, revert to STOCK_OWNED
            pos.state = PositionState.STOCK_OWNED
            pos.stock_shares = remaining_shares
            pos.call_strike = None
            pos.call_premium = None
            pos.call_entry_date = None
            pos.call_dte_at_entry = None
            pos.call_entry_iv = None
            pos.call_expiration_date = None

            pos.notes.append(
                f"Partial call assignment: Sold {shares_called} shares at "
                f"${call_strike:.2f}, "
                f"{remaining_shares} shares remaining (basis: ${pos.stock_basis:.2f})"
            )

            return {
                "ticker": ticker,
                "shares_called": shares_called,
                "remaining_shares": remaining_shares,
                "stock_pnl": stock_pnl,
                "assignment_fee": assignment_fee,
                "cash_after": self.cash,
                "state": PositionState.STOCK_OWNED.value,
            }
        else:
            # All shares called away — close position
            pos.notes.append(
                f"Called away: Sold {shares_called} shares at "
                f"${call_strike:.2f} (basis: ${pos.stock_basis:.2f})"
            )
            pos.notes.append(f"Wheel cycle complete: Total P&L = ${pos.realized_pnl:.2f}")

            closed = self._finalize_position(pos, assignment_date, "call_assigned")
            del self.positions[ticker]

            return closed

    def handle_put_expiration(
        self, ticker: str, expiry_date: date, stock_price: float
    ) -> dict | None:
        """
        Handle put expiration: either assign or expire worthless.

        Args:
            ticker: Stock ticker
            expiry_date: Expiration date
            stock_price: Stock price at expiration

        Returns:
            Closed position summary if expired worthless, else None
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            return None

        if stock_price < pos.put_strike:
            # Assigned
            self.handle_put_assignment(ticker, expiry_date, stock_price)
            return None  # Position remains open (now stock)
        else:
            # Expired worthless (keep full premium)
            pos.notes.append(f"Put expired worthless (stock at ${stock_price:.2f})")
            closed = self._finalize_position(pos, expiry_date, "put_expired_otm")
            del self.positions[ticker]
            return closed

    def open_covered_call(
        self,
        ticker: str,
        strike: float,
        premium: float,
        entry_date: date,
        expiration_date: date,  # CHANGED: explicit date
        iv: float,
    ) -> bool:
        """
        Sell covered call on owned stock.

        Args:
            ticker: Stock ticker
            strike: Call strike price
            premium: Premium collected per share
            entry_date: Trade entry date
            expiration_date: Explicit calendar expiration date
            iv: Implied volatility at entry

        Returns:
            True if call opened successfully
        """
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        if pos.state != PositionState.STOCK_OWNED:
            return False

        # Calculate entry costs for covered call (commission + slippage)
        cost_details = calculate_total_entry_cost(
            premium_per_share=premium, bid_ask_spread=premium * 0.10, trade_type="option"
        )

        # Credit net premium to cash
        self.cash += cost_details["net_premium_collected"]

        # Update position state and store explicit expiration date
        derived_dte = (expiration_date - entry_date).days
        pos.state = PositionState.COVERED_CALL
        pos.call_strike = strike
        pos.call_premium = premium
        pos.call_entry_date = entry_date
        pos.call_dte_at_entry = derived_dte  # Derived
        pos.call_entry_iv = iv
        pos.call_expiration_date = expiration_date  # NEW: explicit date
        pos.realized_pnl += cost_details["gross_premium"]
        pos.transaction_costs += cost_details["total_cost"]

        pos.notes.append(f"Sold {derived_dte}d {strike}C for ${premium:.2f} premium")

        return True

    def close_covered_call(
        self, ticker: str, buyback_price: float, exit_date: date, reason: str = "early_exit"
    ) -> dict | None:
        """
        Buy back an outstanding covered call early (before expiration).

        Returns a dict summarizing the buyback including the call-leg P&L.
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None

        # Compute exit costs including slippage and commission
        cost_details = calculate_total_exit_cost(
            buyback_price_per_share=buyback_price,
            bid_ask_spread=buyback_price * 0.10,
            trade_type="option",
        )

        # AUDIT-VIII P1: the prior call to ``open_covered_call`` already
        # credited ``call_premium * 100`` to ``realized_pnl`` via ``+=``.
        # Adding ``premium - buyback`` here would double-count the
        # premium. Subtract only the gross buyback cost to keep the
        # accumulator semantics consistent with close_short_put after
        # the audit-VIII fix. ``call_leg_pnl`` in the return payload is
        # the isolated P&L of this particular call leg
        # (premium collected minus gross buyback cost), independent of
        # the running accumulator.
        gross_pnl = (pos.call_premium or 0.0) * 100 - cost_details["gross_buyback_cost"]
        pos.realized_pnl -= cost_details["gross_buyback_cost"]
        pos.transaction_costs += cost_details["total_cost"]

        # Deduct total cash paid to buy back the call
        self.cash -= cost_details["total_buyback_cost"]

        # Revert to stock-owned state (we keep the shares)
        pos.state = PositionState.STOCK_OWNED
        pos.call_strike = None
        pos.call_premium = None
        pos.call_entry_date = None
        pos.call_dte_at_entry = None
        pos.call_entry_iv = None
        pos.call_expiration_date = None

        pos.notes.append(f"Bought back call for ${buyback_price:.2f} due to {reason}")

        return {
            "ticker": ticker,
            "call_leg_pnl": gross_pnl,
            "transaction_costs": cost_details["total_cost"],
            "cash_after": self.cash,
            "reason": reason,
        }

    def handle_call_assignment(self, ticker: str, assignment_date: date) -> dict | None:
        """
        Handle call assignment: sell stock at call strike, close Wheel cycle.

        Args:
            ticker: Stock ticker
            assignment_date: Assignment date

        Returns:
            Closed position summary
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return None

        # Sell stock at call strike (account for assignment fee)
        stock_proceeds = pos.call_strike * 100
        assignment_details = calculate_assignment_costs(strike_price=pos.call_strike, shares=100)
        assignment_fee = assignment_details["assignment_fee"]

        self.cash += stock_proceeds - assignment_fee

        # Calculate stock P&L
        stock_pnl = (pos.call_strike - pos.stock_basis) * 100
        pos.realized_pnl += stock_pnl
        pos.transaction_costs += assignment_fee

        pos.notes.append(
            f"Called away: Sold 100 shares at ${pos.call_strike:.2f} (basis: ${pos.stock_basis:.2f})"
        )
        pos.notes.append(f"Wheel cycle complete: Total P&L = ${pos.realized_pnl:.2f}")

        # Close position
        closed = self._finalize_position(pos, assignment_date, "call_assigned")
        del self.positions[ticker]

        return closed

    def handle_call_expiration(self, ticker: str, expiry_date: date, stock_price: float) -> bool:
        """
        Handle call expiration: either assign or keep stock.

        Args:
            ticker: Stock ticker
            expiry_date: Expiration date
            stock_price: Stock price at expiration

        Returns:
            True if handled successfully
        """
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            return False

        if stock_price > pos.call_strike:
            # Assigned
            self.handle_call_assignment(ticker, expiry_date)
        else:
            # Expired worthless, keep stock
            pos.state = PositionState.STOCK_OWNED
            pos.call_strike = None
            pos.call_premium = None
            pos.notes.append(
                f"Call expired worthless (stock at ${stock_price:.2f}), still holding shares"
            )

        return True

    def available_buying_power(self) -> float:
        """Cash genuinely deployable for new positions, net of CSP collateral.

        ``self.cash`` is correct as *brokerage cash* but overstates what
        can still be deployed: :meth:`open_short_put` credits the premium
        to ``cash`` yet never reserves the strike collateral a
        cash-secured put ties up. S2 / S4 / S8 logged this gap three
        times — ``cash`` answers "what's in the account", not "what can
        I still put to work".

        This returns ``cash`` minus the collateral reserved by every open
        cash-secured put: ``put_strike * 100`` per open
        :attr:`PositionState.SHORT_PUT` position. The tracker is one
        contract — 100 shares — per position (``WheelPosition`` has no
        contract-count field; :meth:`handle_put_assignment` acquires a
        flat 100 shares), so the per-position term is ``strike * 100``
        with no contract multiplier. ``STOCK_OWNED`` and ``COVERED_CALL``
        positions reserve nothing: the cash for the shares was already
        spent at assignment, and the short call is covered by the held
        stock, not by cash.

        This is the **cash-secured** definition only. Reg-T margin is a
        different account model — :meth:`open_short_put` happens to check
        Reg-T margin via ``calculate_reg_t_margin_short_put``, but that
        is a separate concern; a Reg-T buying-power mode is a possible
        follow-up and is deliberately out of scope here.

        Returns:
            ``cash - Σ reserved collateral``. **May be negative** — a
            negative result signals an over-committed book: the open
            cash-secured puts reserve more collateral than there is cash
            to secure them. The value is returned raw, never clamped to
            zero, so callers can see the size of the shortfall.
        """
        reserved = 0.0
        for pos in self.positions.values():
            if pos.state == PositionState.SHORT_PUT and pos.put_strike is not None:
                # One contract per position → 100 shares of collateral.
                reserved += pos.put_strike * 100.0
        return self.cash - reserved

    def mark_to_market(
        self,
        current_date: date,
        prices: dict[str, float],
        risk_free_rate: float = 0.04,
        current_ivs: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate current portfolio value (cash + stock + option liabilities).

        Args:
            current_date: Current date for mark.
            prices: Dict of ``{ticker: current_stock_price}``.
            risk_free_rate: Risk-free rate for option pricing (default 4%).
            current_ivs: Optional dict of ``{ticker: live_iv}`` (decimal,
                e.g. ``0.28`` for 28%). When supplied the short-option
                liability is marked at the current market IV; when
                omitted we fall back to the leg's entry IV. The fallback
                is a documented approximation — positions held through
                a vol regime change will be mis-marked if ``current_ivs``
                is not provided. Pass live IV from the broker / options
                chain for production equity curves.

        Returns:
            Total portfolio value including option liabilities.
        """
        from .option_pricer import estimate_option_price_from_iv

        current_ivs = current_ivs or {}
        total_value = self.cash

        for ticker, pos in self.positions.items():
            if ticker not in prices:
                continue

            stock_price = prices[ticker]

            # Add stock value (if owned)
            if pos.state in [PositionState.STOCK_OWNED, PositionState.COVERED_CALL]:
                total_value += stock_price * pos.stock_shares

            # Subtract short put liability
            if pos.state == PositionState.SHORT_PUT:
                if pos.put_expiration_date and current_date < pos.put_expiration_date:
                    days_to_expiry = (pos.put_expiration_date - current_date).days
                    if days_to_expiry > 0:
                        live_iv = current_ivs.get(ticker)
                        iv_used = live_iv if (live_iv and live_iv > 0) else pos.put_entry_iv
                        put_value = estimate_option_price_from_iv(
                            underlying_price=stock_price,
                            strike=pos.put_strike,
                            dte=days_to_expiry,
                            iv=iv_used,
                            risk_free_rate=risk_free_rate,
                            option_type="put",
                        )
                        total_value -= put_value * 100  # Short = liability

            # Subtract short call liability
            if pos.state == PositionState.COVERED_CALL:
                if pos.call_expiration_date and current_date < pos.call_expiration_date:
                    days_to_expiry = (pos.call_expiration_date - current_date).days
                    if days_to_expiry > 0:
                        live_iv = current_ivs.get(ticker)
                        iv_used = live_iv if (live_iv and live_iv > 0) else pos.call_entry_iv
                        call_value = estimate_option_price_from_iv(
                            underlying_price=stock_price,
                            strike=pos.call_strike,
                            dte=days_to_expiry,
                            iv=iv_used,
                            risk_free_rate=risk_free_rate,
                            option_type="call",
                        )
                        total_value -= call_value * 100  # Short = liability

        # Record equity curve
        self.equity_curve.append(
            {
                "date": current_date,
                "portfolio_value": total_value,
                "cash": self.cash,
                "num_positions": len(self.positions),
            }
        )

        return total_value

    def _finalize_position(self, pos: WheelPosition, exit_date: date, exit_reason: str) -> dict:
        """Internal: Convert position to closed trade record"""
        closed = {
            "ticker": pos.ticker,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "exit_reason": exit_reason,
            "hold_days": (exit_date - pos.entry_date).days,
            "realized_pnl": pos.realized_pnl,
            "transaction_costs": pos.transaction_costs,
            # ADD COMMENT: Net P&L = Gross P&L - All Transaction Costs (computed once)
            "net_pnl": pos.realized_pnl - pos.transaction_costs,
            "put_premium": pos.put_premium * 100 if pos.put_premium else 0,
            "call_premium": pos.call_premium * 100 if pos.call_premium else 0,
            "notes": " | ".join(pos.notes),
        }

        self.closed_positions.append(closed)
        return closed

    def get_performance_summary(self) -> pd.DataFrame:
        """Generate performance report"""
        if not self.closed_positions:
            return pd.DataFrame()

        df = pd.DataFrame(self.closed_positions)

        # largest_win / largest_loss must be taken over the winning /
        # losing subsets respectively — not over all trades. ``min()``
        # over every trade returns the smallest *win* when there are no
        # losers, so an all-green book would report a profit as its
        # "largest loss". Empty subset → 0.0.
        wins = df.loc[df["net_pnl"] > 0, "net_pnl"]
        losses = df.loc[df["net_pnl"] < 0, "net_pnl"]

        summary = {
            "total_trades": len(df),
            "winners": len(wins),
            "losers": len(losses),
            "win_rate": len(wins) / len(df),
            "total_pnl": df["net_pnl"].sum(),
            "avg_pnl_per_trade": df["net_pnl"].mean(),
            "total_commissions": df["transaction_costs"].sum(),
            "largest_win": float(wins.max()) if not wins.empty else 0.0,
            "largest_loss": float(losses.min()) if not losses.empty else 0.0,
        }

        return pd.DataFrame([summary])

    # ------------------------------------------------------------------
    # Persistence — JSON round-trip of the whole tracker (S2)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise the whole tracker to a JSON-safe dict.

        Captures everything needed to resume a session: ``cash``,
        ``initial_capital``, the open ``positions``, the closed-trade
        and equity-curve history, and the EV-authority token set /
        audit log. ``PositionState`` enums and ``date`` fields are
        converted to JSON-safe forms (see :meth:`WheelPosition.to_dict`
        and :func:`_record_to_jsonable`).

        The ``connector`` is a live data-source object and is
        deliberately **not** serialised — :meth:`from_dict` and
        :meth:`load` take an optional ``connector`` to re-attach it.

        Inverse of :meth:`from_dict`; :meth:`save` / :meth:`load` wrap
        this with JSON file I/O.
        """
        return {
            "schema_version": 1,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "require_ev_authority": self.require_ev_authority,
            "positions": {tk: p.to_dict() for tk, p in self.positions.items()},
            "closed_positions": [
                _record_to_jsonable(rec, _CLOSED_POSITION_DATE_KEYS)
                for rec in self.closed_positions
            ],
            "equity_curve": [
                _record_to_jsonable(rec, _EQUITY_CURVE_DATE_KEYS) for rec in self.equity_curve
            ],
            "ev_authority_tokens": sorted(self._ev_authority_tokens),
            "ev_authority_log": [dict(entry) for entry in self._ev_authority_log],
        }

    @classmethod
    def from_dict(cls, data: dict, connector: Any | None = None) -> "WheelTracker":
        """Rebuild a tracker from a :meth:`to_dict` dict.

        ``connector`` is re-attached here — it is never part of the
        serialised state. Unknown top-level keys are ignored and absent
        optional keys fall back to empty, so the loader is
        forward-compatible with a newer ``schema_version``.
        """
        tracker = cls(
            initial_capital=data["initial_capital"],
            require_ev_authority=data.get("require_ev_authority", False),
            connector=connector,
        )
        tracker.cash = data["cash"]
        tracker.positions = {
            tk: WheelPosition.from_dict(p) for tk, p in data.get("positions", {}).items()
        }
        tracker.closed_positions = [
            _record_from_jsonable(rec, _CLOSED_POSITION_DATE_KEYS)
            for rec in data.get("closed_positions", [])
        ]
        tracker.equity_curve = [
            _record_from_jsonable(rec, _EQUITY_CURVE_DATE_KEYS)
            for rec in data.get("equity_curve", [])
        ]
        tracker._ev_authority_tokens = set(data.get("ev_authority_tokens", []))
        tracker._ev_authority_log = [dict(entry) for entry in data.get("ev_authority_log", [])]
        return tracker

    def save(self, path: str | Path) -> None:
        """Write :meth:`to_dict` to ``path`` as indented UTF-8 JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path, connector: Any | None = None) -> "WheelTracker":
        """Read a tracker from a JSON file written by :meth:`save`.

        ``connector`` is re-attached (it is never serialised).
        """
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), connector=connector)

    # ------------------------------------------------------------------
    # Roll-suggestion workflow (management-layer integration)
    # ------------------------------------------------------------------
    def suggest_rolls(
        self,
        ticker: str,
        as_of: date,
        current_spot: float,
        current_iv: float,
        risk_free_rate: float | None = None,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        min_net_credit: float = 0.0,
        dividend_yield: float = 0.0,
        forward_log_returns: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Rank candidate rolls for an open short put by forward EV.

        Closes the management-layer gap surfaced by the rolling-campaign
        UX review: when a short put goes adverse the engine has all the
        pieces (:meth:`EVEngine.evaluate`, BSM pricing, :meth:`roll_put`
        mechanics) but no integration that says "here are three
        candidate rolls ranked by EV, with net credit/debit, and whether
        each beats holding the current position." This is that
        integration.

        Parameters
        ----------
        ticker, as_of, current_spot, current_iv:
            Identify the open position and the current market state.
            ``current_iv`` is a decimal (e.g. ``0.28`` for 28%). The
            position must be in :class:`PositionState.SHORT_PUT`.
        risk_free_rate:
            Decimal risk-free rate at ``as_of``. If ``None``, falls
            back to ``self.connector.get_risk_free_rate(as_of)`` (the
            connector documents a decimal return). If neither is
            available, ``ValueError`` is raised.
        target_dtes, target_deltas:
            Cartesian product of new expiries and target put deltas
            (positive numbers; sign handled internally). For each
            ``(DTE, delta)`` pair the method solves for the strike at
            that delta under the current state, prices the new put at
            BSM fair value, and runs it through
            :meth:`EVEngine.evaluate`. Default grid spans 3-9 weeks
            out and 15-30 delta — the conventional wheel-roll
            territory.
        min_net_credit:
            Dollar filter on ``net_credit_debit``. Candidates below
            this threshold are pruned. Default ``0`` keeps only
            credit-rolls (the conventional wheel discipline). Pass a
            negative value to allow rescue debit rolls (e.g. ``-200``
            for up to a $200 debit).
        dividend_yield:
            Annual dividend yield (decimal). Default ``0.0``. Per the
            AUDIT-IX unit contract on
            :meth:`engine.data_connector.MarketDataConnector.get_fundamentals`,
            callers integrating with the connector should normalise
            ``eqy_dvd_yld_12m`` (percent) by dividing by 100.
        forward_log_returns:
            Optional explicit empirical forward log-return distribution
            used for all evaluations (hold + each roll). When ``None``
            (default), per-horizon distributions are pulled via
            ``self.connector.get_ohlcv(ticker)`` and
            :func:`engine.forward_distribution.best_available_forward_distribution`.
            When neither is available, ``EVEngine`` falls back to its
            lognormal sampler — the comparison between candidates
            remains internally consistent, but absolute EV magnitudes
            collapse toward zero (a documented bias).

        Returns
        -------
        pandas.DataFrame
            Zero or more rows, one per surviving candidate, sorted by
            ``roll_ev`` descending. Columns:

            * ``new_strike``, ``new_expiry``, ``new_dte``,
              ``target_delta`` — describe the candidate contract
            * ``new_premium``, ``buyback_cost`` — per-share BSM marks
              (new put fair value, and current open put's fair value
              used as the buyback price)
            * ``net_credit_debit`` — dollar cash flow at the roll
              moment, ``(new_premium - buyback_cost) * 100``; positive
              = credit roll, negative = debit roll
            * ``new_ev_dollars`` — :attr:`EVResult.ev_dollars` on the
              new trade (diagnostic / audit)
            * ``roll_ev`` — marginal forward EV of rolling (the
              headline metric, see below)
            * ``hold_ev`` — marginal forward EV of holding the
              existing position to its expiry (the comparison anchor)
            * ``prob_otm`` — :attr:`EVResult.prob_profit` on the new
              trade
            * ``recommend`` — ``True`` iff ``roll_ev > hold_ev``

        Notes — the EV metric
        ---------------------
        Both ``hold_ev`` and ``roll_ev`` express the **expected dollar
        change in account value from this decision moment forward**,
        so they are directly comparable.

        ``hold_ev`` — keep the current put to expiry::

            synthetic_hold = ShortOptionTrade(
                strike=pos.put_strike,
                premium=buyback_value_per_share,
                dte=dte_remaining,
                iv=current_iv, ...)
            hold_ev = ev_dollars(synthetic_hold)
                      - buyback_value_per_share * 100

        The synthetic re-sells the existing put at its current fair
        value, so we can pipe it through :meth:`EVEngine.evaluate` and
        reuse the engine's empirical forward distribution; we then
        subtract the notional re-sell premium (no premium is actually
        re-collected when holding) to recover the pure forward P&L.

        ``roll_ev`` — close the old, open the new::

            roll_ev = ev_dollars(new_trade) - buyback_total_dollars

        ``ev_dollars(new_trade)`` already contains the new premium
        (the engine's ``gross_premium``); the only additional real
        cash flow at the roll moment is the buyback, subtracted once.
        ``buyback_total_dollars`` includes BSM price plus exit-side
        transaction costs.

        *Why not* ``ev_dollars(new) + net_credit_debit``? Adding
        ``net_credit_debit = (new_premium - buyback) * 100`` to
        ``ev_dollars(new)`` would **double-count** the new premium —
        ``ev_dollars(new)`` already contains it. The single-count
        formula above is the mathematically correct apples-to-apples
        comparison against ``hold_ev``.

        ``recommend`` is ``True`` iff ``roll_ev > hold_ev``.

        §2 invariant
        -------------
        Every candidate's EV — both ``hold_ev`` and each ``roll_ev`` —
        runs through :meth:`EVEngine.evaluate` directly with a
        properly-constructed :class:`ShortOptionTrade`. No
        approximations, no side-channel ranking. Pre-filtering
        (``min_net_credit``) and delta-based strike enumeration are
        purely about *which candidates to score*; the EV authority is
        untouched.

        Out of scope (follow-up): ``suggest_call_rolls`` for the
        covered-call leg. Same shape, operating on
        :class:`PositionState.COVERED_CALL` positions.
        """
        # ---------------- validate position ----------------
        if ticker not in self.positions:
            raise ValueError(f"No open position for {ticker}")
        pos = self.positions[ticker]
        if pos.state != PositionState.SHORT_PUT:
            raise ValueError(
                f"suggest_rolls supports SHORT_PUT positions only; {ticker} is in {pos.state.name}"
            )
        if pos.put_strike is None or pos.put_premium is None or pos.put_expiration_date is None:
            raise ValueError(
                f"position {ticker} is missing put_strike / put_premium / "
                "put_expiration_date - cannot mark"
            )

        # ---------------- validate inputs ----------------
        if current_spot <= 0:
            raise ValueError(f"current_spot must be positive, got {current_spot}")
        if not (0.0 < current_iv <= 3.0):
            raise ValueError(f"current_iv must be a decimal in (0, 3]; got {current_iv}")

        # ---------------- resolve risk-free rate ----------------
        if risk_free_rate is None:
            if self.connector is None:
                raise ValueError(
                    "risk_free_rate is None and WheelTracker has no connector "
                    "to resolve it from. Pass risk_free_rate explicitly, or "
                    "construct the tracker with connector=<MarketDataConnector>."
                )
            try:
                rf_raw = self.connector.get_risk_free_rate(as_of.isoformat())
            except Exception as e:
                raise ValueError(f"connector.get_risk_free_rate raised: {e}") from e
            if rf_raw is None or not np.isfinite(float(rf_raw)):
                risk_free_rate = 0.04
            else:
                risk_free_rate = float(rf_raw)
                # The connector documents a DECIMAL return (AUDIT-VIII),
                # but belt-and-suspenders: an unnormalised percent would
                # blow up BSM downstream.
                if risk_free_rate > 1.0:
                    risk_free_rate = risk_free_rate / 100.0
                risk_free_rate = max(0.0, min(0.25, risk_free_rate))
        if not (0.0 <= risk_free_rate <= 0.25):
            raise ValueError(f"risk_free_rate {risk_free_rate} outside [0, 0.25]")

        # ---------------- remaining DTE / buyback ----------------
        dte_remaining = (pos.put_expiration_date - as_of).days
        if dte_remaining <= 0:
            # At/past expiry — rolling is moot, caller handles expiry/assignment.
            return pd.DataFrame(columns=_ROLL_COLUMNS)
        T_old = dte_remaining / 365.0
        multiplier = 100  # per-contract; suggest_rolls is per-contract by convention

        buyback_value_per_share = black_scholes_price(
            S=current_spot,
            K=float(pos.put_strike),
            T=T_old,
            r=risk_free_rate,
            sigma=current_iv,
            option_type="put",
            q=dividend_yield,
        )
        if buyback_value_per_share <= 0.0:
            # Essentially worthless put — holding wins, no roll can beat that.
            return pd.DataFrame(columns=_ROLL_COLUMNS)

        # Full dollar cost to close the current put: BSM principal plus
        # exit-side transaction costs -- the "total_buyback_cost" key.
        # ("total_cost" is txn-costs-only; netting that instead omitted
        # the buyback principal from roll_ev, making every roll look
        # like a rescue vs hold_ev -- which nets the principal in full.)
        buyback_costs = calculate_total_exit_cost(
            buyback_price_per_share=buyback_value_per_share,
            bid_ask_spread=buyback_value_per_share * 0.10,
            bid=None,
            ask=None,
            trade_type="option",
        )
        buyback_total_dollars = buyback_costs["total_buyback_cost"]

        # ---------------- forward-distribution cache ----------------
        fwd_cache: dict[int, np.ndarray | None] = {}

        def _fwd_for(horizon: int):
            if forward_log_returns is not None:
                return forward_log_returns
            if horizon in fwd_cache:
                return fwd_cache[horizon]
            arr = None
            if self.connector is not None:
                try:
                    from .forward_distribution import (
                        best_available_forward_distribution,
                    )

                    oh = self.connector.get_ohlcv(ticker)
                    if oh is not None and len(oh) > 0:
                        arr, _ = best_available_forward_distribution(
                            oh,
                            horizon_days=int(horizon),
                            as_of=as_of.isoformat(),
                        )
                except Exception:
                    arr = None
            fwd_cache[horizon] = arr
            return arr

        # ---------------- hold_ev ----------------
        ev_engine = EVEngine()  # no event_gate — this is a management decision
        hold_trade = ShortOptionTrade(
            option_type="put",
            underlying=ticker,
            spot=current_spot,
            strike=float(pos.put_strike),
            premium=buyback_value_per_share,
            dte=int(dte_remaining),
            iv=current_iv,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            contracts=1,
            bid=buyback_value_per_share * 0.95,
            ask=buyback_value_per_share * 1.05,
            open_interest=1000,
            regime_multiplier=1.0,
        )
        hold_result = ev_engine.evaluate(
            hold_trade,
            forward_log_returns=_fwd_for(int(dte_remaining)),
        )
        hold_ev = hold_result.ev_dollars - buyback_value_per_share * multiplier

        # ---------------- enumerate roll candidates ----------------
        rows: list[dict] = []
        for new_dte in target_dtes:
            if new_dte <= 0:
                continue
            T_new = new_dte / 365.0
            new_expiry = as_of + timedelta(days=int(new_dte))
            for tgt_delta in target_deltas:
                new_strike_raw = _solve_put_strike(
                    spot=current_spot,
                    T=T_new,
                    r=risk_free_rate,
                    q=dividend_yield,
                    iv=current_iv,
                    target_delta=tgt_delta,
                )
                if new_strike_raw is None:
                    continue
                new_strike = round(new_strike_raw * 2) / 2  # nearest $0.50
                if new_strike <= 0 or new_strike >= current_spot:
                    continue
                new_premium = black_scholes_price(
                    S=current_spot,
                    K=new_strike,
                    T=T_new,
                    r=risk_free_rate,
                    sigma=current_iv,
                    option_type="put",
                    q=dividend_yield,
                )
                if new_premium < 0.05:
                    continue  # too thin to trade
                net_credit_debit = (new_premium - buyback_value_per_share) * multiplier
                if net_credit_debit < min_net_credit:
                    continue
                new_trade = ShortOptionTrade(
                    option_type="put",
                    underlying=ticker,
                    spot=current_spot,
                    strike=float(new_strike),
                    premium=new_premium,
                    dte=int(new_dte),
                    iv=current_iv,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    contracts=1,
                    bid=new_premium * 0.95,
                    ask=new_premium * 1.05,
                    open_interest=1000,
                    regime_multiplier=1.0,
                )
                new_result = ev_engine.evaluate(
                    new_trade,
                    forward_log_returns=_fwd_for(int(new_dte)),
                )
                roll_ev = new_result.ev_dollars - buyback_total_dollars
                rows.append(
                    {
                        "new_strike": new_strike,
                        "new_expiry": new_expiry,
                        "new_dte": int(new_dte),
                        "target_delta": tgt_delta,
                        "new_premium": round(new_premium, 3),
                        "buyback_cost": round(buyback_value_per_share, 3),
                        "net_credit_debit": round(net_credit_debit, 2),
                        "new_ev_dollars": round(new_result.ev_dollars, 2),
                        "roll_ev": round(roll_ev, 2),
                        "hold_ev": round(hold_ev, 2),
                        "prob_otm": round(new_result.prob_profit, 4),
                        "recommend": bool(roll_ev > hold_ev),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=_ROLL_COLUMNS)
        df = pd.DataFrame(rows, columns=_ROLL_COLUMNS)
        df = df.sort_values("roll_ev", ascending=False).reset_index(drop=True)
        return df

    def suggest_call_rolls(
        self,
        ticker: str,
        as_of: date,
        current_spot: float,
        current_iv: float,
        risk_free_rate: float | None = None,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        min_net_credit: float = 0.0,
        dividend_yield: float = 0.0,
        forward_log_returns: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Rank candidate rolls for an open covered call by forward EV.

        The covered-call-leg parallel of :meth:`suggest_rolls` -- the
        deliberately-deferred S3 follow-up, logged again by S8. When a
        covered call goes adverse (the stock has rallied through the
        strike and assignment looms) or simply for premium harvest, this
        gives the trader candidate call rolls ranked by forward EV, not
        just :meth:`roll_call` mechanics.

        A covered call is a short call, so each candidate is scored as a
        :class:`ShortOptionTrade` with ``option_type="call"`` through
        :meth:`EVEngine.evaluate` -- exactly as :meth:`suggest_rolls`
        scores puts. The position must be in
        :class:`PositionState.COVERED_CALL`.

        Parameters
        ----------
        ticker, as_of, current_spot, current_iv:
            Identify the open covered call and current market state.
            ``current_iv`` is a decimal (e.g. ``0.28`` for 28%).
        risk_free_rate:
            Decimal risk-free rate at ``as_of``. If ``None``, falls back
            to ``self.connector.get_risk_free_rate(as_of)``; if neither
            is available, ``ValueError`` is raised.
        target_dtes, target_deltas:
            Cartesian product of new expiries and target call deltas
            (positive numbers). For each ``(DTE, delta)`` pair the method
            solves the strike at that call delta, prices the new call at
            BSM fair value, and runs it through :meth:`EVEngine.evaluate`.
        min_net_credit:
            Dollar filter on ``net_credit_debit``; candidates below it
            are pruned. Default ``0`` keeps credit-rolls only. Pass a
            negative value to allow rescue debit rolls.
        dividend_yield:
            Annual dividend yield (decimal). Default ``0.0``. Flows into
            BSM pricing and the :class:`ShortOptionTrade` so the engine
            prices the call's dividend drag.
        forward_log_returns:
            Optional explicit empirical forward log-return distribution
            used for all evaluations. When ``None`` (default) per-horizon
            distributions are pulled via the connector, mirroring
            :meth:`suggest_rolls`.

        Returns
        -------
        pandas.DataFrame
            Zero or more rows, one per surviving candidate, sorted by
            ``roll_ev`` descending. Same :data:`_ROLL_COLUMNS` schema as
            :meth:`suggest_rolls`: ``new_strike`` is the new call strike,
            ``target_delta`` the call delta, ``prob_otm`` the probability
            the short call expires worthless. Empty (but correctly
            shaped) when no candidate survives.

        Notes -- the EV metric
        ----------------------
        Identical single-count form to :meth:`suggest_rolls` (see that
        method for the full derivation). Both express the expected
        dollar change in account value from this decision moment::

            hold_ev = ev_dollars(synthetic_hold) - buyback_per_share * 100
            roll_ev = ev_dollars(new_trade)      - buyback_total_dollars

        ``ev_dollars(new_trade)`` already contains the new call premium
        via the engine's ``gross_premium``; the buyback is the only extra
        cash flow at the roll moment, subtracted once. Adding
        ``net_credit_debit`` on top would double-count the new premium --
        it is deliberately not added (the bug S3's ledger note warns of).
        ``recommend`` is ``True`` iff ``roll_ev > hold_ev``.

        Section 2 invariant
        -------------------
        Every candidate's EV -- ``hold_ev`` and each ``roll_ev`` -- runs
        through :meth:`EVEngine.evaluate` directly with a
        properly-constructed ``option_type="call"``
        :class:`ShortOptionTrade`. Strike enumeration and the
        ``min_net_credit`` filter only choose *which* candidates to
        score; the EV authority is untouched.
        """
        # ---------------- validate position ----------------
        if ticker not in self.positions:
            raise ValueError(f"No open position for {ticker}")
        pos = self.positions[ticker]
        if pos.state != PositionState.COVERED_CALL:
            raise ValueError(
                f"suggest_call_rolls supports COVERED_CALL positions only; "
                f"{ticker} is in {pos.state.name}"
            )
        if pos.call_strike is None or pos.call_premium is None or pos.call_expiration_date is None:
            raise ValueError(
                f"position {ticker} is missing call_strike / call_premium / "
                "call_expiration_date - cannot mark"
            )

        # ---------------- validate inputs ----------------
        if current_spot <= 0:
            raise ValueError(f"current_spot must be positive, got {current_spot}")
        if not (0.0 < current_iv <= 3.0):
            raise ValueError(f"current_iv must be a decimal in (0, 3]; got {current_iv}")

        # ---------------- resolve risk-free rate ----------------
        if risk_free_rate is None:
            if self.connector is None:
                raise ValueError(
                    "risk_free_rate is None and WheelTracker has no connector "
                    "to resolve it from. Pass risk_free_rate explicitly, or "
                    "construct the tracker with connector=<MarketDataConnector>."
                )
            try:
                rf_raw = self.connector.get_risk_free_rate(as_of.isoformat())
            except Exception as e:
                raise ValueError(f"connector.get_risk_free_rate raised: {e}") from e
            if rf_raw is None or not np.isfinite(float(rf_raw)):
                risk_free_rate = 0.04
            else:
                risk_free_rate = float(rf_raw)
                # Connector documents a DECIMAL return; belt-and-suspenders
                # normalisation guards an unnormalised percent.
                if risk_free_rate > 1.0:
                    risk_free_rate = risk_free_rate / 100.0
                risk_free_rate = max(0.0, min(0.25, risk_free_rate))
        if not (0.0 <= risk_free_rate <= 0.25):
            raise ValueError(f"risk_free_rate {risk_free_rate} outside [0, 0.25]")

        # ---------------- remaining DTE / buyback ----------------
        dte_remaining = (pos.call_expiration_date - as_of).days
        if dte_remaining <= 0:
            # At/past expiry - rolling is moot; caller handles expiry/assignment.
            return pd.DataFrame(columns=_ROLL_COLUMNS)
        T_old = dte_remaining / 365.0
        multiplier = 100  # per-contract; suggest_call_rolls is per-contract by convention

        buyback_value_per_share = black_scholes_price(
            S=current_spot,
            K=float(pos.call_strike),
            T=T_old,
            r=risk_free_rate,
            sigma=current_iv,
            option_type="call",
            q=dividend_yield,
        )
        if buyback_value_per_share <= 0.0:
            # Essentially worthless call - holding wins, no roll can beat it.
            return pd.DataFrame(columns=_ROLL_COLUMNS)

        # Full dollar cost to close the current call: BSM principal plus
        # exit-side transaction costs -- the "total_buyback_cost" key.
        # ("total_cost" alone is txn-costs-only; using it would drop the
        # ~principal from roll_ev and break apples-to-apples vs hold_ev,
        # which subtracts buyback_value_per_share * 100 in full.)
        buyback_costs = calculate_total_exit_cost(
            buyback_price_per_share=buyback_value_per_share,
            bid_ask_spread=buyback_value_per_share * 0.10,
            bid=None,
            ask=None,
            trade_type="option",
        )
        buyback_total_dollars = buyback_costs["total_buyback_cost"]

        # ---------------- forward-distribution cache ----------------
        fwd_cache: dict[int, np.ndarray | None] = {}

        def _fwd_for(horizon: int):
            if forward_log_returns is not None:
                return forward_log_returns
            if horizon in fwd_cache:
                return fwd_cache[horizon]
            arr = None
            if self.connector is not None:
                try:
                    from .forward_distribution import (
                        best_available_forward_distribution,
                    )

                    oh = self.connector.get_ohlcv(ticker)
                    if oh is not None and len(oh) > 0:
                        arr, _ = best_available_forward_distribution(
                            oh,
                            horizon_days=int(horizon),
                            as_of=as_of.isoformat(),
                        )
                except Exception:
                    arr = None
            fwd_cache[horizon] = arr
            return arr

        # ---------------- hold_ev ----------------
        ev_engine = EVEngine()  # no event_gate - this is a management decision
        hold_trade = ShortOptionTrade(
            option_type="call",
            underlying=ticker,
            spot=current_spot,
            strike=float(pos.call_strike),
            premium=buyback_value_per_share,
            dte=int(dte_remaining),
            iv=current_iv,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            contracts=1,
            bid=buyback_value_per_share * 0.95,
            ask=buyback_value_per_share * 1.05,
            open_interest=1000,
            regime_multiplier=1.0,
        )
        hold_result = ev_engine.evaluate(
            hold_trade,
            forward_log_returns=_fwd_for(int(dte_remaining)),
        )
        hold_ev = hold_result.ev_dollars - buyback_value_per_share * multiplier

        # ---------------- enumerate roll candidates ----------------
        rows: list[dict] = []
        for new_dte in target_dtes:
            if new_dte <= 0:
                continue
            T_new = new_dte / 365.0
            new_expiry = as_of + timedelta(days=int(new_dte))
            for tgt_delta in target_deltas:
                new_strike_raw = _solve_call_strike(
                    spot=current_spot,
                    T=T_new,
                    r=risk_free_rate,
                    q=dividend_yield,
                    iv=current_iv,
                    target_delta=tgt_delta,
                )
                if new_strike_raw is None:
                    continue
                new_strike = round(new_strike_raw * 2) / 2  # nearest $0.50
                if new_strike <= current_spot:
                    continue  # a covered call is sold OTM (strike above spot)
                new_premium = black_scholes_price(
                    S=current_spot,
                    K=new_strike,
                    T=T_new,
                    r=risk_free_rate,
                    sigma=current_iv,
                    option_type="call",
                    q=dividend_yield,
                )
                if new_premium < 0.05:
                    continue  # too thin to trade
                net_credit_debit = (new_premium - buyback_value_per_share) * multiplier
                if net_credit_debit < min_net_credit:
                    continue
                new_trade = ShortOptionTrade(
                    option_type="call",
                    underlying=ticker,
                    spot=current_spot,
                    strike=float(new_strike),
                    premium=new_premium,
                    dte=int(new_dte),
                    iv=current_iv,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    contracts=1,
                    bid=new_premium * 0.95,
                    ask=new_premium * 1.05,
                    open_interest=1000,
                    regime_multiplier=1.0,
                )
                new_result = ev_engine.evaluate(
                    new_trade,
                    forward_log_returns=_fwd_for(int(new_dte)),
                )
                roll_ev = new_result.ev_dollars - buyback_total_dollars
                rows.append(
                    {
                        "new_strike": new_strike,
                        "new_expiry": new_expiry,
                        "new_dte": int(new_dte),
                        "target_delta": tgt_delta,
                        "new_premium": round(new_premium, 3),
                        "buyback_cost": round(buyback_value_per_share, 3),
                        "net_credit_debit": round(net_credit_debit, 2),
                        "new_ev_dollars": round(new_result.ev_dollars, 2),
                        "roll_ev": round(roll_ev, 2),
                        "hold_ev": round(hold_ev, 2),
                        "prob_otm": round(new_result.prob_profit, 4),
                        "recommend": bool(roll_ev > hold_ev),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=_ROLL_COLUMNS)
        df = pd.DataFrame(rows, columns=_ROLL_COLUMNS)
        df = df.sort_values("roll_ev", ascending=False).reset_index(drop=True)
        return df
