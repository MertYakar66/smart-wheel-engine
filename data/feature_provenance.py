"""
Feature Provenance and Lag Enforcement

Institutional audit artifact ensuring every generated feature has:
- Documented source data lineage
- Explicit lag specification (how many periods the feature looks back)
- Point-in-time validation (no lookahead)
- Provenance metadata for regulatory compliance

Usage:
    registry = ProvenanceRegistry()
    registry.register_feature(FeatureProvenance(
        name="iv_rank_252d",
        source="options_chain",
        lag_periods=252,
        lag_unit="trading_days",
        description="IV rank over trailing 252 trading days",
    ))

    # Validate all features
    issues = registry.validate_all()
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


class LagUnit(Enum):
    """Time unit for feature lag specification."""
    TRADING_DAYS = "trading_days"
    CALENDAR_DAYS = "calendar_days"
    MINUTES = "minutes"
    TICKS = "ticks"
    NONE = "none"  # Contemporaneous (e.g., current price)


class FeatureStatus(Enum):
    """Feature lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    RETIRED = "retired"


@dataclass
class FeatureProvenance:
    """Complete provenance record for a single feature."""

    name: str
    source: str  # Source data (e.g., "ohlcv", "options_chain", "earnings")
    lag_periods: int  # How many periods of lookback
    lag_unit: LagUnit | str = LagUnit.TRADING_DAYS
    description: str = ""
    computation: str = ""  # Brief formula or method description

    # Audit metadata
    created_date: date | None = None
    created_by: str = ""
    last_validated: date | None = None
    status: FeatureStatus | str = FeatureStatus.ACTIVE

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Other features this depends on

    # Governance
    approved_for_production: bool = False
    approver: str = ""

    def __post_init__(self):
        if isinstance(self.lag_unit, str):
            self.lag_unit = LagUnit(self.lag_unit)
        if isinstance(self.status, str):
            self.status = FeatureStatus(self.status)
        if self.created_date is None:
            self.created_date = date.today()

    @property
    def is_point_in_time_safe(self) -> bool:
        """Feature uses only past data (lag > 0 or is contemporaneous spot)."""
        return self.lag_periods >= 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lag_unit"] = self.lag_unit.value
        d["status"] = self.status.value
        if self.created_date:
            d["created_date"] = self.created_date.isoformat()
        if self.last_validated:
            d["last_validated"] = self.last_validated.isoformat()
        return d


class ProvenanceRegistry:
    """
    Central registry of all feature provenance records.

    Enforces that every feature used in production has documented
    lineage, lag specification, and approval status.
    """

    def __init__(self):
        self._features: dict[str, FeatureProvenance] = {}
        self._audit_log: list[dict] = []

    def register_feature(self, provenance: FeatureProvenance) -> None:
        """Register a feature's provenance."""
        self._features[provenance.name] = provenance
        self._audit_log.append({
            "action": "register",
            "feature": provenance.name,
            "timestamp": datetime.now().isoformat(),
        })

    def get_feature(self, name: str) -> FeatureProvenance | None:
        return self._features.get(name)

    def list_features(
        self,
        status: FeatureStatus | None = None,
        source: str | None = None,
    ) -> list[FeatureProvenance]:
        """List features with optional filters."""
        features = list(self._features.values())
        if status:
            features = [f for f in features if f.status == status]
        if source:
            features = [f for f in features if f.source == source]
        return features

    def validate_all(self) -> list[str]:
        """
        Validate all registered features for institutional compliance.

        Checks:
        1. Every feature has a non-empty source
        2. Every feature has lag_periods >= 0 (no lookahead)
        3. Active features have been validated recently (within 90 days)
        4. Dependencies reference registered features
        5. Production features are approved
        """
        issues = []
        today = date.today()

        for name, prov in self._features.items():
            if not prov.source:
                issues.append(f"{name}: Missing source data specification")

            if prov.lag_periods < 0:
                issues.append(f"{name}: Negative lag ({prov.lag_periods}) indicates lookahead")

            if prov.status == FeatureStatus.ACTIVE:
                if prov.last_validated:
                    days_since = (today - prov.last_validated).days
                    if days_since > 90:
                        issues.append(
                            f"{name}: Last validated {days_since} days ago (>90 day threshold)"
                        )
                else:
                    issues.append(f"{name}: Never validated")

                if not prov.approved_for_production:
                    issues.append(f"{name}: Active but not approved for production")

            for dep in prov.depends_on:
                if dep not in self._features:
                    issues.append(f"{name}: Depends on unregistered feature '{dep}'")

        return issues

    def approve_feature(self, name: str, approver: str) -> bool:
        """Approve a feature for production use."""
        prov = self._features.get(name)
        if prov is None:
            return False

        prov.approved_for_production = True
        prov.approver = approver
        prov.last_validated = date.today()

        self._audit_log.append({
            "action": "approve",
            "feature": name,
            "approver": approver,
            "timestamp": datetime.now().isoformat(),
        })
        return True

    def export_registry(self, path: str) -> None:
        """Export registry to JSON for version control."""
        data = {
            "features": {name: prov.to_dict() for name, prov in self._features.items()},
            "export_date": date.today().isoformat(),
            "feature_count": len(self._features),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_dataframe(self):
        """Export registry as DataFrame for analysis."""
        import pandas as pd
        rows = [prov.to_dict() for prov in self._features.values()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_audit_trail(self) -> list[dict]:
        """Get full audit trail of registry changes."""
        return list(self._audit_log)


# =============================================================================
# Default feature provenance for Smart Wheel Engine
# =============================================================================

def build_default_provenance_registry() -> ProvenanceRegistry:
    """Build provenance registry with all standard wheel engine features."""
    registry = ProvenanceRegistry()

    # Technical features
    for lookback in [5, 10, 20, 50, 200]:
        registry.register_feature(FeatureProvenance(
            name=f"sma_{lookback}",
            source="ohlcv",
            lag_periods=lookback,
            lag_unit=LagUnit.TRADING_DAYS,
            description=f"Simple moving average over {lookback} trading days",
            computation=f"close.rolling({lookback}).mean()",
        ))

    # Volatility features
    for window in [10, 20, 60, 252]:
        registry.register_feature(FeatureProvenance(
            name=f"realized_vol_{window}d",
            source="ohlcv",
            lag_periods=window,
            lag_unit=LagUnit.TRADING_DAYS,
            description=f"Realized volatility over {window} trading days",
            computation=f"log_returns.rolling({window}).std() * sqrt(252)",
        ))

    registry.register_feature(FeatureProvenance(
        name="iv_rank_252d",
        source="options_chain",
        lag_periods=252,
        lag_unit=LagUnit.TRADING_DAYS,
        description="IV rank: percentile of current IV vs trailing 252-day range",
    ))

    registry.register_feature(FeatureProvenance(
        name="iv_percentile_252d",
        source="options_chain",
        lag_periods=252,
        lag_unit=LagUnit.TRADING_DAYS,
        description="IV percentile: % of days with IV below current level",
    ))

    # Event features
    registry.register_feature(FeatureProvenance(
        name="days_to_earnings",
        source="earnings_calendar",
        lag_periods=0,
        lag_unit=LagUnit.NONE,
        description="Days until next earnings announcement",
    ))

    registry.register_feature(FeatureProvenance(
        name="days_to_ex_div",
        source="dividend_calendar",
        lag_periods=0,
        lag_unit=LagUnit.NONE,
        description="Days until next ex-dividend date",
    ))

    # Regime features
    registry.register_feature(FeatureProvenance(
        name="regime_label",
        source="ohlcv",
        lag_periods=60,
        lag_unit=LagUnit.TRADING_DAYS,
        description="Market regime classification (bull/bear/sideways/volatile)",
        depends_on=["realized_vol_20d", "sma_50"],
    ))

    return registry
