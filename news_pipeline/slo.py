"""
News Pipeline Service Level Objectives (SLOs)

Defines and tracks SLOs for the news pipeline:
- Latency targets for each stage
- Availability requirements
- Error budgets
- Quality metrics

Based on professional news trading systems requirements.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """SLO compliance status."""
    MET = "met"
    AT_RISK = "at_risk"  # Within 10% of budget
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class PipelineStage(Enum):
    """News pipeline stages."""
    DISCOVERY = "discovery"
    VERIFICATION = "verification"
    FORMATTING = "formatting"
    EDITORIAL = "editorial"
    PUBLISHING = "publishing"
    END_TO_END = "end_to_end"


class NewsCategory(Enum):
    """News categories with different SLO tiers."""
    FED = "fed"              # Tier 1: Most critical
    EARNINGS = "earnings"    # Tier 1
    MACRO = "macro"          # Tier 2
    GEOPOLITICS = "geopolitics"  # Tier 2
    OIL = "oil"              # Tier 3
    CRYPTO = "crypto"        # Tier 3
    TECH = "tech"            # Tier 3
    OTHER = "other"          # Tier 3


@dataclass
class LatencySLO:
    """Latency SLO definition."""
    stage: PipelineStage
    p50_target_ms: float
    p95_target_ms: float
    p99_target_ms: float
    max_allowed_ms: float

    def check(self, latencies_ms: list[float]) -> SLOStatus:
        """Check if latencies meet SLO."""
        if not latencies_ms:
            return SLOStatus.UNKNOWN

        import numpy as np
        arr = np.array(latencies_ms)

        p50 = np.percentile(arr, 50)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        max_val = arr.max()

        # Check violations
        if max_val > self.max_allowed_ms:
            return SLOStatus.VIOLATED
        if p99 > self.p99_target_ms:
            return SLOStatus.VIOLATED
        if p95 > self.p95_target_ms:
            return SLOStatus.AT_RISK
        if p50 > self.p50_target_ms:
            return SLOStatus.AT_RISK

        return SLOStatus.MET


@dataclass
class AvailabilitySLO:
    """Availability SLO definition."""
    target_percent: float  # e.g., 99.9
    measurement_window: timedelta

    def check(self, uptime_percent: float) -> SLOStatus:
        """Check availability against target."""
        if uptime_percent >= self.target_percent:
            return SLOStatus.MET
        if uptime_percent >= self.target_percent * 0.99:  # Within 1% of target
            return SLOStatus.AT_RISK
        return SLOStatus.VIOLATED


@dataclass
class ErrorBudget:
    """Error budget tracking."""
    name: str
    budget_percent: float  # Allowed error rate
    window: timedelta
    consumed_percent: float = 0.0

    @property
    def remaining_percent(self) -> float:
        return max(0, self.budget_percent - self.consumed_percent)

    @property
    def status(self) -> SLOStatus:
        if self.consumed_percent > self.budget_percent:
            return SLOStatus.VIOLATED
        if self.consumed_percent > self.budget_percent * 0.8:
            return SLOStatus.AT_RISK
        return SLOStatus.MET


@dataclass
class SLOMetrics:
    """Collected SLO metrics for a time period."""
    period_start: datetime
    period_end: datetime
    stage: PipelineStage

    # Latency metrics
    total_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    # Error metrics
    successful_requests: int = 0
    failed_requests: int = 0

    # Quality metrics
    verified_stories: int = 0
    rejected_stories: int = 0

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate

    @property
    def p50_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        import numpy as np
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p95_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        import numpy as np
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        import numpy as np
        return float(np.percentile(self.latencies_ms, 99))


# =============================================================================
# SLO DEFINITIONS
# =============================================================================

# Latency SLOs by stage (milliseconds)
LATENCY_SLOS = {
    PipelineStage.DISCOVERY: LatencySLO(
        stage=PipelineStage.DISCOVERY,
        p50_target_ms=500,
        p95_target_ms=2000,
        p99_target_ms=5000,
        max_allowed_ms=10000,
    ),
    PipelineStage.VERIFICATION: LatencySLO(
        stage=PipelineStage.VERIFICATION,
        p50_target_ms=3000,
        p95_target_ms=10000,
        p99_target_ms=30000,
        max_allowed_ms=60000,
    ),
    PipelineStage.FORMATTING: LatencySLO(
        stage=PipelineStage.FORMATTING,
        p50_target_ms=2000,
        p95_target_ms=8000,
        p99_target_ms=20000,
        max_allowed_ms=45000,
    ),
    PipelineStage.EDITORIAL: LatencySLO(
        stage=PipelineStage.EDITORIAL,
        p50_target_ms=3000,
        p95_target_ms=12000,
        p99_target_ms=30000,
        max_allowed_ms=60000,
    ),
    PipelineStage.PUBLISHING: LatencySLO(
        stage=PipelineStage.PUBLISHING,
        p50_target_ms=100,
        p95_target_ms=500,
        p99_target_ms=1000,
        max_allowed_ms=5000,
    ),
    PipelineStage.END_TO_END: LatencySLO(
        stage=PipelineStage.END_TO_END,
        p50_target_ms=15000,
        p95_target_ms=45000,
        p99_target_ms=90000,
        max_allowed_ms=180000,  # 3 minutes max
    ),
}

# Category-specific latency multipliers (Tier 1 = 1.0, faster required)
CATEGORY_LATENCY_MULTIPLIERS = {
    NewsCategory.FED: 0.5,        # 50% of standard (faster)
    NewsCategory.EARNINGS: 0.5,
    NewsCategory.MACRO: 0.75,
    NewsCategory.GEOPOLITICS: 0.75,
    NewsCategory.OIL: 1.0,
    NewsCategory.CRYPTO: 1.0,
    NewsCategory.TECH: 1.0,
    NewsCategory.OTHER: 1.5,      # 150% of standard (can be slower)
}

# Availability SLOs
AVAILABILITY_SLO = AvailabilitySLO(
    target_percent=99.5,
    measurement_window=timedelta(days=30),
)

# Error budgets
ERROR_BUDGETS = {
    "monthly": ErrorBudget(
        name="Monthly Error Budget",
        budget_percent=0.5,  # 0.5% error rate allowed
        window=timedelta(days=30),
    ),
    "weekly": ErrorBudget(
        name="Weekly Error Budget",
        budget_percent=1.0,  # 1% error rate allowed (tighter weekly)
        window=timedelta(days=7),
    ),
    "daily": ErrorBudget(
        name="Daily Error Budget",
        budget_percent=2.0,  # 2% error rate allowed
        window=timedelta(days=1),
    ),
}


class SLOTracker:
    """
    Tracks SLO metrics and compliance.

    Records latencies, errors, and calculates compliance status.
    """

    def __init__(self, max_history_hours: int = 24 * 30):  # 30 days default
        """
        Initialize SLO tracker.

        Args:
            max_history_hours: Maximum hours of metrics to retain
        """
        self.max_history = timedelta(hours=max_history_hours)
        self._metrics: dict[PipelineStage, list[tuple[datetime, float, bool]]] = {
            stage: [] for stage in PipelineStage
        }
        self._category_metrics: dict[NewsCategory, list[tuple[datetime, float, bool]]] = {
            cat: [] for cat in NewsCategory
        }

    def record_request(
        self,
        stage: PipelineStage,
        latency_ms: float,
        success: bool,
        category: NewsCategory | None = None,
    ) -> None:
        """
        Record a pipeline request.

        Args:
            stage: Pipeline stage
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            category: Optional news category
        """
        now = datetime.now()
        self._metrics[stage].append((now, latency_ms, success))

        if category:
            self._category_metrics[category].append((now, latency_ms, success))

        # Prune old data
        self._prune_old_data()

        # Log if latency exceeds threshold
        slo = LATENCY_SLOS.get(stage)
        if slo and latency_ms > slo.p95_target_ms:
            logger.warning(
                f"High latency in {stage.value}: {latency_ms:.0f}ms "
                f"(p95 target: {slo.p95_target_ms:.0f}ms)"
            )

    def get_metrics(
        self,
        stage: PipelineStage,
        window: timedelta | None = None,
    ) -> SLOMetrics:
        """
        Get SLO metrics for a stage.

        Args:
            stage: Pipeline stage
            window: Time window (default: all available)

        Returns:
            SLOMetrics for the stage
        """
        now = datetime.now()
        if window:
            cutoff = now - window
        else:
            cutoff = datetime.min

        records = [
            (ts, lat, success)
            for ts, lat, success in self._metrics[stage]
            if ts >= cutoff
        ]

        metrics = SLOMetrics(
            period_start=cutoff if window else (records[0][0] if records else now),
            period_end=now,
            stage=stage,
            total_requests=len(records),
            latencies_ms=[lat for _, lat, _ in records],
            successful_requests=sum(1 for _, _, success in records if success),
            failed_requests=sum(1 for _, _, success in records if not success),
        )

        return metrics

    def check_slo_compliance(
        self,
        stage: PipelineStage,
        window: timedelta = timedelta(hours=1),
    ) -> dict[str, Any]:
        """
        Check SLO compliance for a stage.

        Args:
            stage: Pipeline stage
            window: Time window to check

        Returns:
            Compliance report
        """
        metrics = self.get_metrics(stage, window)
        slo = LATENCY_SLOS.get(stage)

        report = {
            "stage": stage.value,
            "window_hours": window.total_seconds() / 3600,
            "total_requests": metrics.total_requests,
            "success_rate": metrics.success_rate,
            "error_rate": metrics.error_rate,
            "latency": {
                "p50_ms": metrics.p50_latency_ms,
                "p95_ms": metrics.p95_latency_ms,
                "p99_ms": metrics.p99_latency_ms,
            },
            "slo": {},
            "status": SLOStatus.UNKNOWN.value,
        }

        if slo and metrics.latencies_ms:
            status = slo.check(metrics.latencies_ms)
            report["slo"] = {
                "p50_target_ms": slo.p50_target_ms,
                "p95_target_ms": slo.p95_target_ms,
                "p99_target_ms": slo.p99_target_ms,
                "max_allowed_ms": slo.max_allowed_ms,
            }
            report["status"] = status.value

        return report

    def get_error_budget_status(
        self,
        budget_name: str = "daily",
    ) -> dict[str, Any]:
        """
        Get error budget status.

        Args:
            budget_name: Name of error budget to check

        Returns:
            Error budget status
        """
        budget = ERROR_BUDGETS.get(budget_name)
        if not budget:
            return {"error": f"Unknown budget: {budget_name}"}

        # Calculate error rate across all stages
        total_requests = 0
        total_errors = 0

        for stage in PipelineStage:
            metrics = self.get_metrics(stage, budget.window)
            total_requests += metrics.total_requests
            total_errors += metrics.failed_requests

        if total_requests > 0:
            error_rate = (total_errors / total_requests) * 100
            budget.consumed_percent = error_rate
        else:
            budget.consumed_percent = 0.0

        return {
            "budget_name": budget.name,
            "window_days": budget.window.days,
            "budget_percent": budget.budget_percent,
            "consumed_percent": round(budget.consumed_percent, 3),
            "remaining_percent": round(budget.remaining_percent, 3),
            "status": budget.status.value,
            "total_requests": total_requests,
            "total_errors": total_errors,
        }

    def generate_slo_report(self) -> dict[str, Any]:
        """Generate comprehensive SLO report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "stages": {},
            "error_budgets": {},
            "overall_status": SLOStatus.MET.value,
        }

        # Check each stage
        any_violated = False
        any_at_risk = False

        for stage in PipelineStage:
            stage_report = self.check_slo_compliance(stage, timedelta(hours=24))
            report["stages"][stage.value] = stage_report

            if stage_report["status"] == SLOStatus.VIOLATED.value:
                any_violated = True
            elif stage_report["status"] == SLOStatus.AT_RISK.value:
                any_at_risk = True

        # Check error budgets
        for budget_name in ERROR_BUDGETS:
            budget_report = self.get_error_budget_status(budget_name)
            report["error_budgets"][budget_name] = budget_report

            if budget_report["status"] == SLOStatus.VIOLATED.value:
                any_violated = True
            elif budget_report["status"] == SLOStatus.AT_RISK.value:
                any_at_risk = True

        # Overall status
        if any_violated:
            report["overall_status"] = SLOStatus.VIOLATED.value
        elif any_at_risk:
            report["overall_status"] = SLOStatus.AT_RISK.value

        return report

    def _prune_old_data(self) -> None:
        """Remove metrics older than max_history."""
        cutoff = datetime.now() - self.max_history

        for stage in PipelineStage:
            self._metrics[stage] = [
                (ts, lat, success)
                for ts, lat, success in self._metrics[stage]
                if ts >= cutoff
            ]

        for category in NewsCategory:
            self._category_metrics[category] = [
                (ts, lat, success)
                for ts, lat, success in self._category_metrics[category]
                if ts >= cutoff
            ]


# Global SLO tracker instance
_slo_tracker: SLOTracker | None = None


def get_slo_tracker() -> SLOTracker:
    """Get or create global SLO tracker."""
    global _slo_tracker
    if _slo_tracker is None:
        _slo_tracker = SLOTracker()
    return _slo_tracker


def record_pipeline_request(
    stage: PipelineStage,
    latency_ms: float,
    success: bool,
    category: NewsCategory | None = None,
) -> None:
    """Convenience function to record a pipeline request."""
    get_slo_tracker().record_request(stage, latency_ms, success, category)
