"""
Base Advisor Class

Abstract base class for all advisor agents.
Provides common functionality and enforces response schema.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime

from .schema import (
    AdvisorInput,
    AdvisorResponse,
    ConfidenceLevel,
    Judgment,
)


class BaseAdvisor(ABC):
    """
    Abstract base class for advisor agents.

    Each advisor represents a distinct investment philosophy
    and evaluates trades through that lens.

    Subclasses must implement:
    - name: Advisor identifier
    - philosophy: One-line philosophy description
    - evaluate(): Core evaluation logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Advisor name/identifier."""
        pass

    @property
    @abstractmethod
    def philosophy(self) -> str:
        """One-line investment philosophy."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt defining advisor persona.

        This is the core of each advisor's personality.
        It defines:
        - Worldview
        - Priorities
        - What to focus on
        - What to ignore
        - How to challenge ideas
        """
        pass

    @abstractmethod
    def _analyze(self, input_data: AdvisorInput) -> dict:
        """
        Core analysis logic.

        Returns a dict with:
        - judgment: Judgment enum
        - judgment_summary: str
        - key_reasons: list[str]
        - critical_questions: list[str]
        - hidden_risks: list[str]
        - would_approve_if: list[str]
        - would_reject_if: list[str]
        - confidence: ConfidenceLevel
        - confidence_explanation: str
        """
        pass

    def evaluate(self, input_data: AdvisorInput) -> AdvisorResponse:
        """
        Evaluate a trade candidate.

        This method:
        1. Times the analysis
        2. Calls the subclass _analyze method
        3. Wraps result in AdvisorResponse
        4. Validates response completeness

        Args:
            input_data: Standardized advisor input

        Returns:
            AdvisorResponse with structured evaluation
        """
        start_time = time.time()

        # Run analysis
        analysis = self._analyze(input_data)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build response
        response = AdvisorResponse(
            advisor_name=self.name,
            advisor_philosophy=self.philosophy,
            judgment=analysis.get("judgment", Judgment.NEUTRAL),
            judgment_summary=analysis.get("judgment_summary", "No summary provided"),
            key_reasons=analysis.get("key_reasons", []),
            critical_questions=analysis.get("critical_questions", []),
            hidden_risks=analysis.get("hidden_risks", []),
            would_approve_if=analysis.get("would_approve_if", []),
            would_reject_if=analysis.get("would_reject_if", []),
            confidence=analysis.get("confidence", ConfidenceLevel.MEDIUM),
            confidence_explanation=analysis.get("confidence_explanation", ""),
            analysis_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
        )

        # Validate response
        self._validate_response(response)

        return response

    def _validate_response(self, response: AdvisorResponse) -> None:
        """
        Validate response completeness.

        Ensures advisors don't produce empty or incomplete responses.
        """
        if not response.judgment_summary:
            raise ValueError(f"{self.name}: judgment_summary cannot be empty")

        if len(response.key_reasons) < 2:
            raise ValueError(f"{self.name}: must provide at least 2 key_reasons")

        if len(response.critical_questions) < 1:
            raise ValueError(f"{self.name}: must provide at least 1 critical_question")

    # =========================================================================
    # HELPER METHODS FOR SUBCLASSES
    # =========================================================================

    def _assess_position_size(self, input_data: AdvisorInput) -> str:
        """Assess if position size is appropriate."""
        trade = input_data.candidate_trade
        portfolio = input_data.portfolio

        # Calculate notional exposure
        notional = trade.strike * 100 * trade.contracts

        # As percentage of portfolio
        notional_pct = (notional / portfolio.total_equity) * 100

        if notional_pct > 10:
            return f"LARGE: {notional_pct:.1f}% of portfolio"
        elif notional_pct > 5:
            return f"MODERATE: {notional_pct:.1f}% of portfolio"
        else:
            return f"SMALL: {notional_pct:.1f}% of portfolio"

    def _assess_concentration_risk(self, input_data: AdvisorInput) -> dict:
        """Assess concentration risk from this trade."""
        trade = input_data.candidate_trade
        portfolio = input_data.portfolio

        # Find current sector allocation
        # Determine sector from existing positions
        sector = None
        for pos in portfolio.positions:
            if pos.ticker == trade.ticker:
                sector = pos.sector
                break

        # Default sector mapping for common tickers
        sector_map = {
            "NVDA": "Technology",
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "META": "Technology",
            "AMZN": "Consumer",
            "TSLA": "Consumer",
            "JPM": "Financials",
            "BAC": "Financials",
            "XOM": "Energy",
            "CVX": "Energy",
            "JNJ": "Healthcare",
            "UNH": "Healthcare",
            "PFE": "Healthcare",
        }

        if not sector:
            sector = sector_map.get(trade.ticker, "Unknown")

        current_allocation = portfolio.sector_allocation.get(sector, 0)

        return {
            "sector": sector,
            "current_allocation": current_allocation,
            "high_concentration": current_allocation > 30,
            "warning": current_allocation > 40,
        }

    def _assess_earnings_risk(self, input_data: AdvisorInput) -> dict:
        """Assess earnings-related risk."""
        trade = input_data.candidate_trade

        return {
            "has_earnings": trade.earnings_before_expiry,
            "dte": trade.dte,
            "risk_level": "HIGH" if trade.earnings_before_expiry else "LOW",
            "note": "Earnings event within expiration window"
            if trade.earnings_before_expiry
            else "No earnings event",
        }

    def _assess_iv_environment(self, input_data: AdvisorInput) -> dict:
        """Assess implied volatility environment."""
        trade = input_data.candidate_trade
        market = input_data.market

        iv_rank = trade.iv_rank

        if iv_rank > 80:
            level = "VERY_HIGH"
            interpretation = "Premium rich - good for selling"
        elif iv_rank > 60:
            level = "HIGH"
            interpretation = "Elevated premium - favorable for sellers"
        elif iv_rank > 40:
            level = "NORMAL"
            interpretation = "Average premium levels"
        elif iv_rank > 20:
            level = "LOW"
            interpretation = "Compressed premium - less favorable for sellers"
        else:
            level = "VERY_LOW"
            interpretation = "Premium poor - avoid selling strategies"

        return {
            "iv_rank": iv_rank,
            "vix": market.vix,
            "level": level,
            "interpretation": interpretation,
            "regime": market.regime.value,
        }

    def _assess_probability_profile(self, input_data: AdvisorInput) -> dict:
        """Assess probability and expected value."""
        trade = input_data.candidate_trade

        ev = trade.expected_value
        p_otm = trade.p_otm

        if ev > 3.0 and p_otm > 0.70:
            quality = "EXCELLENT"
        elif ev > 2.0 and p_otm > 0.65:
            quality = "GOOD"
        elif ev > 1.0 and p_otm > 0.60:
            quality = "ACCEPTABLE"
        elif ev > 0:
            quality = "MARGINAL"
        else:
            quality = "POOR"

        return {
            "expected_value": ev,
            "p_otm": p_otm,
            "p_profit": trade.p_profit,
            "quality": quality,
        }

    def _format_trade_summary(self, input_data: AdvisorInput) -> str:
        """Format trade into readable summary."""
        trade = input_data.candidate_trade
        return (
            f"{trade.trade_type.value.upper()} on {trade.ticker}: "
            f"${trade.strike} strike, {trade.dte} DTE, "
            f"{abs(trade.delta):.0%} delta, ${trade.premium:.2f} premium"
        )
