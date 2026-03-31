"""
Feature Engineering Module

Layer 2 of the data architecture - where edge lives.

Modules:
- volatility: RV estimators, IV metrics
- technical: Price-based indicators
- options: Flow features, Greeks
- dynamics: CHANGE-based features (ΔOI, ΔIV) - THE EDGE
- vol_edge: Volatility mispricing (IV vs RV) - THE CORE
- assignment: Assignment risk modeling
- events: Earnings and macro event features
- regime: Market regime detection
- labels: Training labels for ML
"""

from src.features.assignment import AssignmentFeatures
from src.features.dynamics import OptionsDynamics
from src.features.events import EventVolatility
from src.features.labels import LabelGenerator, OptionOutcome
from src.features.options import OptionsFeatures
from src.features.regime import MarketRegime, RegimeDetector, VolRegime
from src.features.technical import TechnicalFeatures
from src.features.vol_edge import VolatilityEdge
from src.features.volatility import VolatilityFeatures

__all__ = [
    # State features (Layer 1 derivatives)
    "VolatilityFeatures",
    "TechnicalFeatures",
    "OptionsFeatures",
    # Change features (THE EDGE)
    "OptionsDynamics",
    "VolatilityEdge",
    # Strategy-specific
    "AssignmentFeatures",
    "EventVolatility",
    "RegimeDetector",
    # Labels
    "LabelGenerator",
    # Enums
    "MarketRegime",
    "VolRegime",
    "OptionOutcome",
]
