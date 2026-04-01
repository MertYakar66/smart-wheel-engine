"""
Web Vitals and Performance Monitoring

Tracks Core Web Vitals and performance metrics for the dashboard.
Based on Google's Web Vitals standards for production-grade UX.

Core Web Vitals:
- LCP (Largest Contentful Paint): Loading performance
- FID (First Input Delay): Interactivity
- CLS (Cumulative Layout Shift): Visual stability

Additional Metrics:
- FCP (First Contentful Paint)
- TTFB (Time to First Byte)
- TTI (Time to Interactive)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VitalRating(Enum):
    """Performance rating based on Web Vitals thresholds."""
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class VitalThreshold:
    """Threshold definitions for a vital metric."""
    name: str
    unit: str
    good_max: float
    needs_improvement_max: float

    def rate(self, value: float) -> VitalRating:
        """Rate a value against thresholds."""
        if value <= self.good_max:
            return VitalRating.GOOD
        if value <= self.needs_improvement_max:
            return VitalRating.NEEDS_IMPROVEMENT
        return VitalRating.POOR


# Core Web Vitals Thresholds (Google recommendations)
VITAL_THRESHOLDS = {
    "LCP": VitalThreshold(
        name="Largest Contentful Paint",
        unit="ms",
        good_max=2500,
        needs_improvement_max=4000,
    ),
    "FID": VitalThreshold(
        name="First Input Delay",
        unit="ms",
        good_max=100,
        needs_improvement_max=300,
    ),
    "CLS": VitalThreshold(
        name="Cumulative Layout Shift",
        unit="score",
        good_max=0.1,
        needs_improvement_max=0.25,
    ),
    "FCP": VitalThreshold(
        name="First Contentful Paint",
        unit="ms",
        good_max=1800,
        needs_improvement_max=3000,
    ),
    "TTFB": VitalThreshold(
        name="Time to First Byte",
        unit="ms",
        good_max=800,
        needs_improvement_max=1800,
    ),
    "TTI": VitalThreshold(
        name="Time to Interactive",
        unit="ms",
        good_max=3800,
        needs_improvement_max=7300,
    ),
}

# Page-specific performance budgets
PAGE_BUDGETS = {
    "dashboard": {
        "LCP": 2000,
        "FID": 50,
        "CLS": 0.05,
        "bundle_size_kb": 500,
        "api_calls_max": 10,
    },
    "portfolio": {
        "LCP": 2500,
        "FID": 100,
        "CLS": 0.1,
        "bundle_size_kb": 300,
        "api_calls_max": 5,
    },
    "news": {
        "LCP": 1500,  # News should load fast
        "FID": 50,
        "CLS": 0.05,
        "bundle_size_kb": 200,
        "api_calls_max": 3,
    },
    "options": {
        "LCP": 3000,  # Complex calculations allowed
        "FID": 100,
        "CLS": 0.1,
        "bundle_size_kb": 400,
        "api_calls_max": 8,
    },
}


@dataclass
class VitalMeasurement:
    """Single measurement of a vital metric."""
    metric: str
    value: float
    timestamp: datetime
    page: str
    user_agent: str | None = None
    connection_type: str | None = None  # 4g, 3g, wifi, etc.

    @property
    def rating(self) -> VitalRating:
        threshold = VITAL_THRESHOLDS.get(self.metric)
        if not threshold:
            return VitalRating.GOOD
        return threshold.rate(self.value)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "rating": self.rating.value,
            "timestamp": self.timestamp.isoformat(),
            "page": self.page,
            "user_agent": self.user_agent,
            "connection_type": self.connection_type,
        }


@dataclass
class PageMetrics:
    """Aggregated metrics for a page."""
    page: str
    period_start: datetime
    period_end: datetime

    # Sample counts
    total_measurements: int = 0

    # LCP
    lcp_p75: float | None = None
    lcp_values: list[float] = field(default_factory=list)

    # FID
    fid_p75: float | None = None
    fid_values: list[float] = field(default_factory=list)

    # CLS
    cls_p75: float | None = None
    cls_values: list[float] = field(default_factory=list)

    # Additional
    fcp_p75: float | None = None
    ttfb_p75: float | None = None

    @property
    def lcp_rating(self) -> VitalRating | None:
        if self.lcp_p75 is None:
            return None
        return VITAL_THRESHOLDS["LCP"].rate(self.lcp_p75)

    @property
    def fid_rating(self) -> VitalRating | None:
        if self.fid_p75 is None:
            return None
        return VITAL_THRESHOLDS["FID"].rate(self.fid_p75)

    @property
    def cls_rating(self) -> VitalRating | None:
        if self.cls_p75 is None:
            return None
        return VITAL_THRESHOLDS["CLS"].rate(self.cls_p75)

    @property
    def overall_rating(self) -> VitalRating:
        """Overall rating based on all core vitals."""
        ratings = [self.lcp_rating, self.fid_rating, self.cls_rating]
        ratings = [r for r in ratings if r is not None]

        if not ratings:
            return VitalRating.GOOD

        if any(r == VitalRating.POOR for r in ratings):
            return VitalRating.POOR
        if any(r == VitalRating.NEEDS_IMPROVEMENT for r in ratings):
            return VitalRating.NEEDS_IMPROVEMENT
        return VitalRating.GOOD

    def check_budget(self) -> dict[str, Any]:
        """Check if page meets its performance budget."""
        budget = PAGE_BUDGETS.get(self.page, {})
        violations = []

        if self.lcp_p75 and "LCP" in budget:
            if self.lcp_p75 > budget["LCP"]:
                violations.append({
                    "metric": "LCP",
                    "budget": budget["LCP"],
                    "actual": self.lcp_p75,
                    "exceeded_by": self.lcp_p75 - budget["LCP"],
                })

        if self.fid_p75 and "FID" in budget:
            if self.fid_p75 > budget["FID"]:
                violations.append({
                    "metric": "FID",
                    "budget": budget["FID"],
                    "actual": self.fid_p75,
                    "exceeded_by": self.fid_p75 - budget["FID"],
                })

        if self.cls_p75 and "CLS" in budget:
            if self.cls_p75 > budget["CLS"]:
                violations.append({
                    "metric": "CLS",
                    "budget": budget["CLS"],
                    "actual": self.cls_p75,
                    "exceeded_by": self.cls_p75 - budget["CLS"],
                })

        return {
            "page": self.page,
            "budget_met": len(violations) == 0,
            "violations": violations,
        }

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_measurements": self.total_measurements,
            "core_web_vitals": {
                "LCP": {
                    "p75": self.lcp_p75,
                    "rating": self.lcp_rating.value if self.lcp_rating else None,
                    "threshold_good": VITAL_THRESHOLDS["LCP"].good_max,
                },
                "FID": {
                    "p75": self.fid_p75,
                    "rating": self.fid_rating.value if self.fid_rating else None,
                    "threshold_good": VITAL_THRESHOLDS["FID"].good_max,
                },
                "CLS": {
                    "p75": self.cls_p75,
                    "rating": self.cls_rating.value if self.cls_rating else None,
                    "threshold_good": VITAL_THRESHOLDS["CLS"].good_max,
                },
            },
            "overall_rating": self.overall_rating.value,
            "budget_check": self.check_budget(),
        }


class WebVitalsTracker:
    """
    Tracks and analyzes Web Vitals metrics.

    Features:
    - Records individual measurements
    - Calculates p75 aggregates (Google's standard)
    - Checks against performance budgets
    - Generates performance reports
    """

    def __init__(self, retention_hours: int = 24 * 7):  # 7 days default
        """
        Initialize tracker.

        Args:
            retention_hours: How long to keep measurements
        """
        self.retention = timedelta(hours=retention_hours)
        self._measurements: list[VitalMeasurement] = []

    def record(
        self,
        metric: str,
        value: float,
        page: str,
        user_agent: str | None = None,
        connection_type: str | None = None,
    ) -> None:
        """
        Record a vital measurement.

        Args:
            metric: Metric name (LCP, FID, CLS, etc.)
            value: Measured value
            page: Page where measurement was taken
            user_agent: Optional user agent string
            connection_type: Optional connection type
        """
        measurement = VitalMeasurement(
            metric=metric,
            value=value,
            timestamp=datetime.now(),
            page=page,
            user_agent=user_agent,
            connection_type=connection_type,
        )
        self._measurements.append(measurement)

        # Log warnings for poor ratings
        if measurement.rating == VitalRating.POOR:
            logger.warning(
                f"Poor {metric} on {page}: {value} "
                f"(threshold: {VITAL_THRESHOLDS.get(metric, VitalThreshold('', '', 0, 0)).needs_improvement_max})"
            )

        # Prune old measurements
        self._prune()

    def get_page_metrics(
        self,
        page: str,
        window: timedelta = timedelta(hours=24),
    ) -> PageMetrics:
        """
        Get aggregated metrics for a page.

        Args:
            page: Page name
            window: Time window for aggregation

        Returns:
            PageMetrics with p75 values
        """
        now = datetime.now()
        cutoff = now - window

        # Filter measurements
        page_measurements = [
            m for m in self._measurements
            if m.page == page and m.timestamp >= cutoff
        ]

        metrics = PageMetrics(
            page=page,
            period_start=cutoff,
            period_end=now,
            total_measurements=len(page_measurements),
        )

        # Calculate p75 for each metric
        for metric_name in ["LCP", "FID", "CLS", "FCP", "TTFB"]:
            values = [
                m.value for m in page_measurements
                if m.metric == metric_name
            ]
            if values:
                p75 = float(np.percentile(values, 75))
                if metric_name == "LCP":
                    metrics.lcp_p75 = p75
                    metrics.lcp_values = values
                elif metric_name == "FID":
                    metrics.fid_p75 = p75
                    metrics.fid_values = values
                elif metric_name == "CLS":
                    metrics.cls_p75 = p75
                    metrics.cls_values = values
                elif metric_name == "FCP":
                    metrics.fcp_p75 = p75
                elif metric_name == "TTFB":
                    metrics.ttfb_p75 = p75

        return metrics

    def get_all_pages_metrics(
        self,
        window: timedelta = timedelta(hours=24),
    ) -> dict[str, PageMetrics]:
        """Get metrics for all pages."""
        pages = {m.page for m in self._measurements}
        return {
            page: self.get_page_metrics(page, window)
            for page in pages
        }

    def check_all_budgets(self) -> dict[str, Any]:
        """Check all pages against their budgets."""
        results = {
            "checked_at": datetime.now().isoformat(),
            "pages": {},
            "all_budgets_met": True,
            "violations_count": 0,
        }

        for page in PAGE_BUDGETS:
            metrics = self.get_page_metrics(page)
            budget_check = metrics.check_budget()
            results["pages"][page] = budget_check

            if not budget_check["budget_met"]:
                results["all_budgets_met"] = False
                results["violations_count"] += len(budget_check["violations"])

        return results

    def generate_report(
        self,
        window: timedelta = timedelta(hours=24),
    ) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        all_metrics = self.get_all_pages_metrics(window)
        budget_check = self.check_all_budgets()

        # Count ratings
        rating_counts = {
            VitalRating.GOOD.value: 0,
            VitalRating.NEEDS_IMPROVEMENT.value: 0,
            VitalRating.POOR.value: 0,
        }
        for metrics in all_metrics.values():
            rating_counts[metrics.overall_rating.value] += 1

        report = {
            "generated_at": datetime.now().isoformat(),
            "window_hours": window.total_seconds() / 3600,
            "summary": {
                "total_pages": len(all_metrics),
                "total_measurements": sum(
                    m.total_measurements for m in all_metrics.values()
                ),
                "rating_distribution": rating_counts,
                "all_budgets_met": budget_check["all_budgets_met"],
                "budget_violations": budget_check["violations_count"],
            },
            "pages": {
                page: metrics.to_dict()
                for page, metrics in all_metrics.items()
            },
            "thresholds": {
                name: {
                    "good_max": t.good_max,
                    "needs_improvement_max": t.needs_improvement_max,
                    "unit": t.unit,
                }
                for name, t in VITAL_THRESHOLDS.items()
            },
            "recommendations": self._generate_recommendations(all_metrics),
        }

        return report

    def _generate_recommendations(
        self,
        metrics: dict[str, PageMetrics],
    ) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        for page, m in metrics.items():
            if m.lcp_rating == VitalRating.POOR:
                recommendations.append(
                    f"{page}: LCP is poor ({m.lcp_p75:.0f}ms). "
                    f"Consider lazy loading, image optimization, or CDN."
                )
            elif m.lcp_rating == VitalRating.NEEDS_IMPROVEMENT:
                recommendations.append(
                    f"{page}: LCP needs improvement ({m.lcp_p75:.0f}ms). "
                    f"Target: <{VITAL_THRESHOLDS['LCP'].good_max}ms"
                )

            if m.fid_rating == VitalRating.POOR:
                recommendations.append(
                    f"{page}: FID is poor ({m.fid_p75:.0f}ms). "
                    f"Consider code splitting, reducing JS bundle size."
                )

            if m.cls_rating == VitalRating.POOR:
                recommendations.append(
                    f"{page}: CLS is poor ({m.cls_p75:.3f}). "
                    f"Set explicit dimensions on images/embeds, avoid dynamic content injection."
                )

        return recommendations

    def _prune(self) -> None:
        """Remove measurements older than retention period."""
        cutoff = datetime.now() - self.retention
        self._measurements = [
            m for m in self._measurements
            if m.timestamp >= cutoff
        ]


# Global tracker instance
_vitals_tracker: WebVitalsTracker | None = None


def get_vitals_tracker() -> WebVitalsTracker:
    """Get or create global vitals tracker."""
    global _vitals_tracker
    if _vitals_tracker is None:
        _vitals_tracker = WebVitalsTracker()
    return _vitals_tracker


def record_vital(
    metric: str,
    value: float,
    page: str,
    **kwargs,
) -> None:
    """Convenience function to record a vital."""
    get_vitals_tracker().record(metric, value, page, **kwargs)
