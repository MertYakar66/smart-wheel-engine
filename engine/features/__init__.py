"""
Feature Engineering Module

Moved from ``src/features/`` on 2026-09-17 (Track F; DECISIONS.md D2 update).
``technical.py`` and ``volatility.py`` have engine / data-layer consumers; the
other seven modules serve only the research feature pipeline
(``data/feature_pipeline.py``) and its tests.

Layer 2 of the original data architecture - where edge lives.

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

from engine.features.assignment import AssignmentFeatures
from engine.features.dynamics import OptionsDynamics
from engine.features.events import EventVolatility
from engine.features.labels import LabelGenerator, OptionOutcome
from engine.features.options import OptionsFeatures
from engine.features.regime import MarketRegime, RegimeDetector, VolRegime
from engine.features.technical import TechnicalFeatures
from engine.features.vol_edge import VolatilityEdge
from engine.features.volatility import VolatilityFeatures

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
