"""Backtesting module for Wheel Strategy Engine."""

from .simulator import WheelBacktester
from .walk_forward import (
    FoldResult,
    OutOfSampleTracker,
    ParameterStabilityAnalyzer,
    ValidationFold,
    ValidationMethod,
    WalkForwardResult,
    WalkForwardValidator,
    run_anchored_walk_forward,
)

__all__ = [
    "WheelBacktester",
    "WalkForwardValidator",
    "WalkForwardResult",
    "ValidationFold",
    "FoldResult",
    "ValidationMethod",
    "ParameterStabilityAnalyzer",
    "OutOfSampleTracker",
    "run_anchored_walk_forward",
]
