"""
Advisor System Schemas

Strict data models for advisor inputs and outputs.
Ensures consistent, structured reasoning across all agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConfidenceLevel(Enum):
    """Advisor confidence in their assessment."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Judgment(Enum):
    """Core trade judgment."""
    STRONG_APPROVE = "strong_approve"
    APPROVE = "approve"
    NEUTRAL = "neutral"
    REJECT = "reject"
    STRONG_REJECT = "strong_reject"


class TradeType(Enum):
    """Option trade types."""
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    IRON_CONDOR = "iron_condor"
    STRANGLE = "strangle"
    STRADDLE = "straddle"
    CUSTOM = "custom"


class RegimeType(Enum):
    """Market regime classification."""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

@dataclass
class Position:
    """Current portfolio position."""
    ticker: str
    shares: int
    avg_cost: float
    current_price: float
    sector: str
    market_cap: str  # small/mid/large/mega

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.current_price - self.avg_cost) / self.avg_cost * 100


@dataclass
class CandidateTrade:
    """Trade candidate to evaluate."""
    ticker: str
    trade_type: TradeType
    strike: float
    expiration_date: str  # YYYY-MM-DD
    dte: int  # days to expiration
    delta: float
    premium: float
    contracts: int

    # Calculated metrics from quant engine
    expected_value: float  # as percentage
    p_otm: float  # probability of expiring OTM
    p_profit: float  # probability of profit
    iv_rank: float  # 0-100
    iv_percentile: float  # 0-100

    # Greeks
    theta: float
    gamma: float
    vega: float

    # Optional context
    underlying_price: float = 0.0
    earnings_before_expiry: bool = False
    notes: str = ""


@dataclass
class PortfolioContext:
    """Current portfolio state for context."""
    positions: list[Position]
    total_equity: float
    cash_available: float
    buying_power: float

    # Concentration metrics
    sector_allocation: dict[str, float]  # sector -> percentage
    top_5_concentration: float  # % of portfolio in top 5 positions

    # Risk metrics
    portfolio_beta: float
    portfolio_delta: float
    max_drawdown_30d: float
    var_95: float  # 95% VaR as percentage

    # Current open options
    open_positions_count: int
    total_premium_at_risk: float
    total_margin_used: float


@dataclass
class MarketContext:
    """Current market environment."""
    regime: RegimeType
    vix: float
    vix_percentile: float  # where current VIX sits vs history

    # Market levels
    spy_price: float
    spy_50ma: float
    spy_200ma: float

    # Rates
    fed_funds_rate: float
    treasury_10y: float

    # Recent events
    recent_fed_action: str = ""
    upcoming_events: list[str] = field(default_factory=list)

    # Timestamp
    as_of: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AdvisorInput:
    """
    Complete input package for advisor evaluation.

    This standardized format ensures consistent analysis
    across all advisor agents.
    """
    candidate_trade: CandidateTrade
    portfolio: PortfolioContext
    market: MarketContext

    # Request context
    request_id: str = ""
    urgency: str = "normal"  # normal, high, low
    specific_concerns: list[str] = field(default_factory=list)


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

@dataclass
class AdvisorResponse:
    """
    Structured response from an advisor agent.

    Every agent MUST produce this exact structure.
    No free-form rambling allowed.
    """
    # Identity
    advisor_name: str
    advisor_philosophy: str

    # Core judgment
    judgment: Judgment
    judgment_summary: str  # 1-2 sentence summary

    # Structured reasoning
    key_reasons: list[str]  # 3-5 bullet points
    critical_questions: list[str]  # questions the trader should answer
    hidden_risks: list[str]  # risks not obvious from metrics

    # What would change the assessment
    would_approve_if: list[str]  # conditions for approval
    would_reject_if: list[str]  # conditions for rejection

    # Confidence
    confidence: ConfidenceLevel
    confidence_explanation: str

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0


@dataclass
class CommitteeOutput:
    """
    Aggregated output from all advisors.

    Synthesizes individual opinions into actionable summary.
    """
    # Request info
    request_id: str
    trade_summary: str

    # Individual responses
    advisor_responses: list[AdvisorResponse]

    # Aggregated analysis
    unanimous_approve: bool
    unanimous_reject: bool
    approval_count: int
    rejection_count: int
    neutral_count: int

    # Consensus analysis
    areas_of_agreement: list[str]
    areas_of_disagreement: list[str]
    unresolved_risks: list[str]

    # Final recommendation
    committee_judgment: Judgment
    committee_reasoning: str

    # Action items
    required_before_trade: list[str]  # must do before trading
    recommended_modifications: list[str]  # suggested changes to trade

    # Confidence
    committee_confidence: ConfidenceLevel

    # Timestamps
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    total_processing_time_ms: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_sample_input() -> AdvisorInput:
    """Create a sample input for testing."""
    position = Position(
        ticker="NVDA",
        shares=200,
        avg_cost=450.0,
        current_price=520.0,
        sector="Technology",
        market_cap="mega"
    )

    trade = CandidateTrade(
        ticker="NVDA",
        trade_type=TradeType.SHORT_PUT,
        strike=480.0,
        expiration_date="2026-05-15",
        dte=45,
        delta=-0.30,
        premium=12.50,
        contracts=1,
        expected_value=2.3,
        p_otm=0.68,
        p_profit=0.72,
        iv_rank=72.0,
        iv_percentile=78.0,
        theta=0.45,
        gamma=0.02,
        vega=0.85,
        underlying_price=520.0,
        earnings_before_expiry=True
    )

    portfolio = PortfolioContext(
        positions=[position],
        total_equity=150000.0,
        cash_available=50000.0,
        buying_power=100000.0,
        sector_allocation={"Technology": 45.0, "Healthcare": 20.0, "Financials": 15.0, "Consumer": 20.0},
        top_5_concentration=65.0,
        portfolio_beta=1.25,
        portfolio_delta=0.8,
        max_drawdown_30d=-8.5,
        var_95=3.2,
        open_positions_count=5,
        total_premium_at_risk=8500.0,
        total_margin_used=25000.0
    )

    market = MarketContext(
        regime=RegimeType.HIGH_VOL,
        vix=22.5,
        vix_percentile=72.0,
        spy_price=485.0,
        spy_50ma=478.0,
        spy_200ma=465.0,
        fed_funds_rate=4.5,
        treasury_10y=4.2,
        recent_fed_action="Held rates steady",
        upcoming_events=["NVDA earnings in 15 days", "FOMC meeting in 20 days"]
    )

    return AdvisorInput(
        candidate_trade=trade,
        portfolio=portfolio,
        market=market,
        request_id="test_001",
        specific_concerns=["Already heavy in semiconductors", "Earnings risk"]
    )
