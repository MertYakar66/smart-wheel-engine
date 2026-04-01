"""
Integration Module

Bridges the Advisor AI Layer with the Options Intelligence Engine.

Provides:
- Automatic conversion of engine outputs to advisor inputs
- Pre-trade evaluation workflow
- Post-trade review system
- Batch evaluation for multiple candidates
"""

from datetime import datetime

from .committee import CommitteeEngine, format_committee_report
from .schema import (
    AdvisorInput,
    CandidateTrade,
    CommitteeOutput,
    MarketContext,
    PortfolioContext,
    Position,
    RegimeType,
    TradeType,
)


class EngineIntegration:
    """
    Integration layer between Options Engine and Advisor System.

    Converts engine outputs into standardized advisor inputs
    and provides evaluation workflows.
    """

    # Mapping from engine trade types to advisor schema
    TRADE_TYPE_MAP = {
        "short_put": TradeType.SHORT_PUT,
        "short_call": TradeType.SHORT_CALL,
        "covered_call": TradeType.COVERED_CALL,
        "cash_secured_put": TradeType.CASH_SECURED_PUT,
        "put_spread": TradeType.PUT_SPREAD,
        "call_spread": TradeType.CALL_SPREAD,
        "iron_condor": TradeType.IRON_CONDOR,
        "strangle": TradeType.STRANGLE,
        "straddle": TradeType.STRADDLE,
    }

    # Mapping from engine regime types
    REGIME_MAP = {
        "low_vol": RegimeType.LOW_VOL,
        "normal": RegimeType.NORMAL,
        "high_vol": RegimeType.HIGH_VOL,
        "crisis": RegimeType.CRISIS,
        "trending_up": RegimeType.TRENDING_UP,
        "trending_down": RegimeType.TRENDING_DOWN,
        "low_volatility": RegimeType.LOW_VOL,
        "high_volatility": RegimeType.HIGH_VOL,
    }

    def __init__(self):
        """Initialize integration with committee engine."""
        self.committee = CommitteeEngine(parallel=True)

    def evaluate_trade(
        self,
        trade_candidate: dict,
        portfolio_state: dict,
        market_state: dict,
        specific_concerns: list[str] | None = None,
    ) -> CommitteeOutput:
        """
        Evaluate a single trade candidate.

        Args:
            trade_candidate: Dict with trade parameters from engine
            portfolio_state: Dict with portfolio state from engine
            market_state: Dict with market conditions

        Returns:
            CommitteeOutput with full evaluation
        """
        # Convert to advisor input format
        advisor_input = self._build_advisor_input(
            trade_candidate,
            portfolio_state,
            market_state,
            specific_concerns or [],
        )

        # Run committee evaluation
        return self.committee.evaluate(advisor_input)

    def evaluate_batch(
        self,
        trade_candidates: list[dict],
        portfolio_state: dict,
        market_state: dict,
    ) -> list[CommitteeOutput]:
        """
        Evaluate multiple trade candidates.

        Args:
            trade_candidates: List of trade dicts from engine
            portfolio_state: Dict with portfolio state
            market_state: Dict with market conditions

        Returns:
            List of CommitteeOutput for each trade
        """
        results = []
        for trade in trade_candidates:
            result = self.evaluate_trade(trade, portfolio_state, market_state)
            results.append(result)
        return results

    def filter_approved(
        self,
        trade_candidates: list[dict],
        portfolio_state: dict,
        market_state: dict,
        min_approval_count: int = 2,
    ) -> list[tuple[dict, CommitteeOutput]]:
        """
        Filter trade candidates to only those approved by committee.

        Args:
            trade_candidates: List of trade dicts
            portfolio_state: Portfolio state dict
            market_state: Market conditions dict
            min_approval_count: Minimum advisor approvals required

        Returns:
            List of (trade, evaluation) tuples for approved trades
        """
        approved = []

        for trade in trade_candidates:
            result = self.evaluate_trade(trade, portfolio_state, market_state)

            if result.approval_count >= min_approval_count:
                approved.append((trade, result))

        # Sort by approval count (descending)
        approved.sort(key=lambda x: x[1].approval_count, reverse=True)

        return approved

    def _build_advisor_input(
        self,
        trade: dict,
        portfolio: dict,
        market: dict,
        concerns: list[str],
    ) -> AdvisorInput:
        """Convert engine dicts to AdvisorInput."""

        # Build CandidateTrade
        candidate_trade = CandidateTrade(
            ticker=trade.get("ticker", ""),
            trade_type=self._map_trade_type(trade.get("trade_type", "short_put")),
            strike=float(trade.get("strike", 0)),
            expiration_date=trade.get("expiration_date", ""),
            dte=int(trade.get("dte", 45)),
            delta=float(trade.get("delta", -0.30)),
            premium=float(trade.get("premium", 0)),
            contracts=int(trade.get("contracts", 1)),
            expected_value=float(trade.get("expected_value", 0)),
            p_otm=float(trade.get("p_otm", 0.65)),
            p_profit=float(trade.get("p_profit", 0.70)),
            iv_rank=float(trade.get("iv_rank", 50)),
            iv_percentile=float(trade.get("iv_percentile", 50)),
            theta=float(trade.get("theta", 0)),
            gamma=float(trade.get("gamma", 0)),
            vega=float(trade.get("vega", 0)),
            underlying_price=float(trade.get("underlying_price", trade.get("strike", 0))),
            earnings_before_expiry=trade.get("earnings_before_expiry", False),
            notes=trade.get("notes", ""),
        )

        # Build positions
        positions = []
        for pos in portfolio.get("positions", []):
            positions.append(Position(
                ticker=pos.get("ticker", ""),
                shares=int(pos.get("shares", 0)),
                avg_cost=float(pos.get("avg_cost", 0)),
                current_price=float(pos.get("current_price", 0)),
                sector=pos.get("sector", "Unknown"),
                market_cap=pos.get("market_cap", "large"),
            ))

        # Build PortfolioContext
        portfolio_context = PortfolioContext(
            positions=positions,
            total_equity=float(portfolio.get("total_equity", 100000)),
            cash_available=float(portfolio.get("cash_available", 50000)),
            buying_power=float(portfolio.get("buying_power", 100000)),
            sector_allocation=portfolio.get("sector_allocation", {}),
            top_5_concentration=float(portfolio.get("top_5_concentration", 50)),
            portfolio_beta=float(portfolio.get("portfolio_beta", 1.0)),
            portfolio_delta=float(portfolio.get("portfolio_delta", 0.5)),
            max_drawdown_30d=float(portfolio.get("max_drawdown_30d", -5)),
            var_95=float(portfolio.get("var_95", 2.5)),
            open_positions_count=int(portfolio.get("open_positions_count", 0)),
            total_premium_at_risk=float(portfolio.get("total_premium_at_risk", 0)),
            total_margin_used=float(portfolio.get("total_margin_used", 0)),
        )

        # Build MarketContext
        market_context = MarketContext(
            regime=self._map_regime(market.get("regime", "normal")),
            vix=float(market.get("vix", 18)),
            vix_percentile=float(market.get("vix_percentile", 50)),
            spy_price=float(market.get("spy_price", 450)),
            spy_50ma=float(market.get("spy_50ma", 445)),
            spy_200ma=float(market.get("spy_200ma", 430)),
            fed_funds_rate=float(market.get("fed_funds_rate", 5.0)),
            treasury_10y=float(market.get("treasury_10y", 4.5)),
            recent_fed_action=market.get("recent_fed_action", ""),
            upcoming_events=market.get("upcoming_events", []),
            as_of=datetime.utcnow(),
        )

        return AdvisorInput(
            candidate_trade=candidate_trade,
            portfolio=portfolio_context,
            market=market_context,
            request_id=trade.get("request_id", f"trade_{datetime.utcnow().timestamp()}"),
            urgency=trade.get("urgency", "normal"),
            specific_concerns=concerns,
        )

    def _map_trade_type(self, trade_type: str) -> TradeType:
        """Map string trade type to enum."""
        return self.TRADE_TYPE_MAP.get(trade_type.lower(), TradeType.CUSTOM)

    def _map_regime(self, regime: str) -> RegimeType:
        """Map string regime to enum."""
        return self.REGIME_MAP.get(regime.lower(), RegimeType.NORMAL)


def quick_evaluate(
    ticker: str,
    strike: float,
    dte: int = 45,
    delta: float = -0.30,
    premium: float = 5.0,
    underlying_price: float | None = None,
    expected_value: float = 2.0,
    p_otm: float = 0.68,
    iv_rank: float = 50.0,
    print_report: bool = True,
) -> CommitteeOutput:
    """
    Quick evaluation helper for simple trades.

    Args:
        ticker: Stock ticker
        strike: Strike price
        dte: Days to expiration
        delta: Option delta
        premium: Premium received
        underlying_price: Current stock price (defaults to strike)
        expected_value: Expected return %
        p_otm: Probability of expiring OTM
        iv_rank: IV rank 0-100
        print_report: Print formatted report

    Returns:
        CommitteeOutput
    """
    if underlying_price is None:
        underlying_price = strike * 1.08  # Assume ~8% OTM

    trade = {
        "ticker": ticker,
        "trade_type": "short_put",
        "strike": strike,
        "expiration_date": "",
        "dte": dte,
        "delta": delta,
        "premium": premium,
        "contracts": 1,
        "expected_value": expected_value,
        "p_otm": p_otm,
        "p_profit": p_otm + 0.05,
        "iv_rank": iv_rank,
        "iv_percentile": iv_rank,
        "theta": 0.3,
        "gamma": 0.02,
        "vega": 0.5,
        "underlying_price": underlying_price,
        "earnings_before_expiry": False,
    }

    portfolio = {
        "positions": [],
        "total_equity": 100000,
        "cash_available": 50000,
        "buying_power": 100000,
        "sector_allocation": {},
        "top_5_concentration": 30,
        "portfolio_beta": 1.0,
        "portfolio_delta": 0.5,
        "max_drawdown_30d": -3,
        "var_95": 2.0,
        "open_positions_count": 2,
        "total_premium_at_risk": 3000,
        "total_margin_used": 20000,
    }

    market = {
        "regime": "normal",
        "vix": 18,
        "vix_percentile": 45,
        "spy_price": 485,
        "spy_50ma": 480,
        "spy_200ma": 460,
        "fed_funds_rate": 4.5,
        "treasury_10y": 4.2,
        "recent_fed_action": "",
        "upcoming_events": [],
    }

    integration = EngineIntegration()
    result = integration.evaluate_trade(trade, portfolio, market)

    if print_report:
        print(format_committee_report(result))

    return result
