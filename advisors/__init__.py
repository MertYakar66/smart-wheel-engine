"""
Advisor AI Layer for Options Intelligence Engine

Multi-agent advisory system providing institutional-grade decision critique
using AI agents modeled after distinct investment philosophies.

Core Advisors:
- BuffettAdvisor: Business quality + capital allocation discipline
- MungerAdvisor: Inversion + bias detection + multidisciplinary reasoning
- SimonsAdvisor: Statistical rigor + signal validation + robustness

Usage:
    from advisors import CommitteeEngine, create_sample_input, format_committee_report

    # Create committee
    committee = CommitteeEngine()

    # Evaluate a trade
    input_data = create_sample_input()  # or build your own AdvisorInput
    result = committee.evaluate(input_data)

    # Print formatted report
    print(format_committee_report(result))
"""

# Schemas
# Base
from .base import BaseAdvisor

# Advisors
from .buffett import BuffettAdvisor

# Committee
from .committee import CommitteeEngine, format_committee_report

# Integration
from .integration import EngineIntegration, quick_evaluate
from .munger import MungerAdvisor
from .schema import (
    AdvisorInput,
    # Output models
    AdvisorResponse,
    CandidateTrade,
    ClosedTradeRecord,
    CommitteeOutput,
    # Enums
    ConfidenceLevel,
    Judgment,
    MarketContext,
    PortfolioContext,
    PortfolioReviewInput,
    PortfolioReviewOutput,
    # Input models
    Position,
    PostMortemInput,
    PostMortemOutput,
    RegimeType,
    TradeType,
    # Helpers
    create_sample_input,
)
from .simons import SimonsAdvisor

# Optional advisors (may not be installed yet)
try:
    from .taleb import TalebAdvisor
except ImportError:
    TalebAdvisor = None  # type: ignore[assignment,misc]

__all__ = [
    # Enums
    "ConfidenceLevel",
    "Judgment",
    "TradeType",
    "RegimeType",
    # Input models
    "Position",
    "CandidateTrade",
    "PortfolioContext",
    "MarketContext",
    "AdvisorInput",
    # Output models
    "AdvisorResponse",
    "CommitteeOutput",
    # Helpers
    "create_sample_input",
    # Base
    "BaseAdvisor",
    # Advisors
    "BuffettAdvisor",
    "MungerAdvisor",
    "SimonsAdvisor",
    # Committee
    "CommitteeEngine",
    "format_committee_report",
    # Integration
    "EngineIntegration",
    "quick_evaluate",
    # Taleb (optional)
    "TalebAdvisor",
    # Portfolio review & post-mortem
    "PortfolioReviewInput",
    "PortfolioReviewOutput",
    "PostMortemInput",
    "PostMortemOutput",
    "ClosedTradeRecord",
]
