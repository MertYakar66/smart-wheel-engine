"""
Browser Agent Selector Robustness Metrics

Tracks and validates CSS selector reliability across browser agents.
Essential for detecting when target websites change their DOM structure.

Features:
- Selector success rate tracking
- DOM drift detection
- Automatic fallback suggestions
- Nightly replay test support
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SelectorStatus(Enum):
    """Status of a selector."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILING = "failing"
    DEPRECATED = "deprecated"


class DriftSeverity(Enum):
    """Severity of detected DOM drift."""
    NONE = "none"
    MINOR = "minor"      # Selector still works but structure changed
    MODERATE = "moderate"  # Selector works intermittently
    SEVERE = "severe"    # Selector no longer works


@dataclass
class SelectorMetrics:
    """Metrics for a single selector."""
    selector: str
    element_type: str  # e.g., "input", "button", "textarea"
    purpose: str  # e.g., "prompt_input", "send_button", "response_text"

    # Success tracking
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    # Timing
    avg_locate_ms: float = 0.0
    max_locate_ms: float = 0.0

    # DOM fingerprint
    last_dom_hash: str | None = None
    dom_changes_detected: int = 0

    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_success: datetime | None = None
    last_failure: datetime | None = None

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    @property
    def status(self) -> SelectorStatus:
        if self.total_attempts < 5:
            return SelectorStatus.ACTIVE
        if self.success_rate >= 0.95:
            return SelectorStatus.ACTIVE
        if self.success_rate >= 0.80:
            return SelectorStatus.DEGRADED
        return SelectorStatus.FAILING

    @property
    def drift_severity(self) -> DriftSeverity:
        if self.dom_changes_detected == 0:
            return DriftSeverity.NONE
        if self.success_rate >= 0.90:
            return DriftSeverity.MINOR
        if self.success_rate >= 0.50:
            return DriftSeverity.MODERATE
        return DriftSeverity.SEVERE

    def record_attempt(
        self,
        success: bool,
        locate_ms: float,
        dom_hash: str | None = None,
    ) -> None:
        """Record a selector location attempt."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
            self.last_success = datetime.now()
        else:
            self.failed_attempts += 1
            self.last_failure = datetime.now()

        # Update timing
        if self.total_attempts == 1:
            self.avg_locate_ms = locate_ms
        else:
            self.avg_locate_ms = (
                (self.avg_locate_ms * (self.total_attempts - 1) + locate_ms)
                / self.total_attempts
            )
        self.max_locate_ms = max(self.max_locate_ms, locate_ms)

        # Check for DOM drift
        if dom_hash and dom_hash != self.last_dom_hash:
            if self.last_dom_hash is not None:
                self.dom_changes_detected += 1
                logger.info(f"DOM change detected for selector: {self.selector}")
            self.last_dom_hash = dom_hash

    def to_dict(self) -> dict:
        return {
            "selector": self.selector,
            "element_type": self.element_type,
            "purpose": self.purpose,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": round(self.success_rate, 4),
            "status": self.status.value,
            "drift_severity": self.drift_severity.value,
            "avg_locate_ms": round(self.avg_locate_ms, 2),
            "max_locate_ms": round(self.max_locate_ms, 2),
            "dom_changes_detected": self.dom_changes_detected,
            "first_seen": self.first_seen.isoformat(),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
        }


@dataclass
class FallbackSelector:
    """Alternative selector for fallback."""
    selector: str
    priority: int  # Lower = higher priority
    success_rate: float
    notes: str = ""


@dataclass
class AgentMetrics:
    """Metrics for a browser agent (Claude, ChatGPT, Gemini)."""
    agent_name: str
    selectors: dict[str, SelectorMetrics] = field(default_factory=dict)

    # Overall metrics
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0

    # Downtime tracking
    last_healthy: datetime | None = None
    current_downtime_start: datetime | None = None
    total_downtime_minutes: float = 0.0

    @property
    def session_success_rate(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return self.successful_sessions / self.total_sessions

    @property
    def overall_selector_health(self) -> float:
        """Average selector success rate."""
        if not self.selectors:
            return 1.0
        rates = [s.success_rate for s in self.selectors.values()]
        return sum(rates) / len(rates)

    @property
    def failing_selectors(self) -> list[str]:
        """List of selectors with FAILING status."""
        return [
            name for name, sel in self.selectors.items()
            if sel.status == SelectorStatus.FAILING
        ]

    @property
    def degraded_selectors(self) -> list[str]:
        """List of selectors with DEGRADED status."""
        return [
            name for name, sel in self.selectors.items()
            if sel.status == SelectorStatus.DEGRADED
        ]

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "session_success_rate": round(self.session_success_rate, 4),
            "overall_selector_health": round(self.overall_selector_health, 4),
            "failing_selectors": self.failing_selectors,
            "degraded_selectors": self.degraded_selectors,
            "total_downtime_minutes": round(self.total_downtime_minutes, 2),
            "selectors": {
                name: sel.to_dict()
                for name, sel in self.selectors.items()
            },
        }


class RobustnessTracker:
    """
    Tracks browser agent selector robustness.

    Features:
    - Records selector success/failure
    - Detects DOM changes
    - Suggests fallbacks
    - Generates robustness reports
    """

    # Selector definitions by agent
    AGENT_SELECTORS = {
        "claude": {
            "prompt_input": ("div.ProseMirror[contenteditable='true']", "div"),
            "send_button": ("button[aria-label='Send Message']", "button"),
            "response_text": ("div.prose", "div"),
            "new_chat_button": ("button[aria-label='Start new chat']", "button"),
            "model_selector": ("button[data-testid='model-selector']", "button"),
        },
        "chatgpt": {
            "prompt_input": ("textarea#prompt-textarea", "textarea"),
            "send_button": ("button[data-testid='send-button']", "button"),
            "response_text": ("div.markdown", "div"),
            "new_chat_button": ("nav a", "a"),
            "model_selector": ("button[data-testid='model-switcher']", "button"),
        },
        "gemini": {
            "prompt_input": ("textarea[aria-label='Enter a prompt']", "textarea"),
            "send_button": ("button[aria-label='Send message']", "button"),
            "response_text": ("div.response-content", "div"),
            "new_chat_button": ("button[aria-label='New chat']", "button"),
        },
    }

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize robustness tracker.

        Args:
            storage_path: Path to store metrics. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.agents: dict[str, AgentMetrics] = {}

        # Initialize agent metrics with known selectors
        for agent_name, selectors in self.AGENT_SELECTORS.items():
            self.agents[agent_name] = AgentMetrics(agent_name=agent_name)
            for purpose, (selector, elem_type) in selectors.items():
                self.agents[agent_name].selectors[purpose] = SelectorMetrics(
                    selector=selector,
                    element_type=elem_type,
                    purpose=purpose,
                )

        if self.storage_path and self.storage_path.exists():
            self._load_metrics()

    def record_selector_attempt(
        self,
        agent_name: str,
        purpose: str,
        success: bool,
        locate_ms: float,
        dom_snippet: str | None = None,
    ) -> None:
        """
        Record a selector location attempt.

        Args:
            agent_name: Name of the agent (claude, chatgpt, gemini)
            purpose: Purpose of selector (prompt_input, send_button, etc.)
            success: Whether element was found
            locate_ms: Time to locate in milliseconds
            dom_snippet: Optional DOM snippet for change detection
        """
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentMetrics(agent_name=agent_name)

        agent = self.agents[agent_name]

        if purpose not in agent.selectors:
            # Create new selector entry
            selector, elem_type = self.AGENT_SELECTORS.get(agent_name, {}).get(
                purpose, ("unknown", "unknown")
            )
            agent.selectors[purpose] = SelectorMetrics(
                selector=selector,
                element_type=elem_type,
                purpose=purpose,
            )

        # Calculate DOM hash if snippet provided
        dom_hash = None
        if dom_snippet:
            dom_hash = hashlib.md5(dom_snippet.encode()).hexdigest()[:16]

        agent.selectors[purpose].record_attempt(success, locate_ms, dom_hash)

        # Log warnings for failures
        if not success:
            selector_metrics = agent.selectors[purpose]
            if selector_metrics.failed_attempts >= 3:
                logger.warning(
                    f"Selector '{purpose}' for {agent_name} has "
                    f"{selector_metrics.failed_attempts} failures "
                    f"(success rate: {selector_metrics.success_rate:.1%})"
                )

        # Save periodically
        if self.storage_path and agent.selectors[purpose].total_attempts % 10 == 0:
            self._save_metrics()

    def record_session(
        self,
        agent_name: str,
        success: bool,
    ) -> None:
        """Record a session attempt."""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentMetrics(agent_name=agent_name)

        agent = self.agents[agent_name]
        agent.total_sessions += 1

        if success:
            agent.successful_sessions += 1
            if agent.current_downtime_start:
                # Calculate downtime
                downtime = datetime.now() - agent.current_downtime_start
                agent.total_downtime_minutes += downtime.total_seconds() / 60
                agent.current_downtime_start = None
            agent.last_healthy = datetime.now()
        else:
            agent.failed_sessions += 1
            if not agent.current_downtime_start:
                agent.current_downtime_start = datetime.now()

    def get_agent_metrics(self, agent_name: str) -> AgentMetrics | None:
        """Get metrics for an agent."""
        return self.agents.get(agent_name)

    def get_failing_selectors(self) -> dict[str, list[str]]:
        """Get all failing selectors by agent."""
        result = {}
        for agent_name, agent in self.agents.items():
            failing = agent.failing_selectors
            if failing:
                result[agent_name] = failing
        return result

    def suggest_fallbacks(
        self,
        agent_name: str,
        purpose: str,
    ) -> list[FallbackSelector]:
        """
        Suggest fallback selectors for a failing selector.

        Returns list of alternative selectors to try.
        """
        # Common fallback strategies
        fallbacks = []

        original = self.agents.get(agent_name, AgentMetrics(agent_name=agent_name)).selectors.get(purpose)
        if not original:
            return fallbacks

        # Strategy 1: More generic selectors
        if "data-testid" in original.selector:
            # Try without data-testid
            generic = original.selector.split("[data-testid")[0]
            fallbacks.append(FallbackSelector(
                selector=generic,
                priority=1,
                success_rate=0.0,
                notes="Generic selector without data-testid",
            ))

        # Strategy 2: Different attribute selectors
        elem_type = original.element_type
        if purpose == "prompt_input":
            fallbacks.extend([
                FallbackSelector(f"{elem_type}[contenteditable='true']", 2, 0.0, "Contenteditable"),
                FallbackSelector(f"{elem_type}[role='textbox']", 3, 0.0, "Role-based"),
                FallbackSelector(f"{elem_type}.input", 4, 0.0, "Class-based"),
            ])
        elif purpose == "send_button":
            fallbacks.extend([
                FallbackSelector("button[type='submit']", 2, 0.0, "Submit button"),
                FallbackSelector("button svg[data-icon='send']", 3, 0.0, "SVG icon"),
                FallbackSelector("button:has-text('Send')", 4, 0.0, "Text-based"),
            ])

        return fallbacks

    def generate_robustness_report(self) -> dict[str, Any]:
        """Generate comprehensive robustness report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_agents": len(self.agents),
                "healthy_agents": 0,
                "degraded_agents": 0,
                "failing_agents": 0,
            },
            "agents": {},
            "alerts": [],
            "recommendations": [],
        }

        for agent_name, agent in self.agents.items():
            agent_report = agent.to_dict()
            report["agents"][agent_name] = agent_report

            # Categorize agent health
            if agent.overall_selector_health >= 0.95:
                report["summary"]["healthy_agents"] += 1
            elif agent.overall_selector_health >= 0.80:
                report["summary"]["degraded_agents"] += 1
            else:
                report["summary"]["failing_agents"] += 1

            # Generate alerts
            for selector_name in agent.failing_selectors:
                report["alerts"].append({
                    "severity": "high",
                    "agent": agent_name,
                    "selector": selector_name,
                    "message": f"Selector '{selector_name}' is failing",
                })

            for selector_name in agent.degraded_selectors:
                report["alerts"].append({
                    "severity": "medium",
                    "agent": agent_name,
                    "selector": selector_name,
                    "message": f"Selector '{selector_name}' is degraded",
                })

            # Generate recommendations
            if agent.failing_selectors:
                fallbacks = self.suggest_fallbacks(agent_name, agent.failing_selectors[0])
                if fallbacks:
                    report["recommendations"].append({
                        "agent": agent_name,
                        "selector": agent.failing_selectors[0],
                        "fallbacks": [
                            {"selector": f.selector, "notes": f.notes}
                            for f in fallbacks[:3]
                        ],
                    })

        return report

    def _save_metrics(self) -> None:
        """Save metrics to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        metrics_file = self.storage_path / "robustness_metrics.json"

        data = {
            agent_name: agent.to_dict()
            for agent_name, agent in self.agents.items()
        }

        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_metrics(self) -> None:
        """Load metrics from storage."""
        if not self.storage_path:
            return

        metrics_file = self.storage_path / "robustness_metrics.json"
        if not metrics_file.exists():
            return

        try:
            with open(metrics_file) as f:
                data = json.load(f)

            # Reconstruct metrics objects
            for agent_name, agent_data in data.items():
                if agent_name not in self.agents:
                    self.agents[agent_name] = AgentMetrics(agent_name=agent_name)

                agent = self.agents[agent_name]
                agent.total_sessions = agent_data.get("total_sessions", 0)
                agent.successful_sessions = agent_data.get("successful_sessions", 0)
                agent.failed_sessions = agent_data.get("failed_sessions", 0)
                agent.total_downtime_minutes = agent_data.get("total_downtime_minutes", 0.0)

                for sel_name, sel_data in agent_data.get("selectors", {}).items():
                    if sel_name not in agent.selectors:
                        agent.selectors[sel_name] = SelectorMetrics(
                            selector=sel_data.get("selector", ""),
                            element_type=sel_data.get("element_type", ""),
                            purpose=sel_name,
                        )
                    sel = agent.selectors[sel_name]
                    sel.total_attempts = sel_data.get("total_attempts", 0)
                    sel.successful_attempts = sel_data.get("successful_attempts", 0)
                    sel.failed_attempts = sel_data.get("failed_attempts", 0)
                    sel.avg_locate_ms = sel_data.get("avg_locate_ms", 0.0)
                    sel.max_locate_ms = sel_data.get("max_locate_ms", 0.0)
                    sel.dom_changes_detected = sel_data.get("dom_changes_detected", 0)

        except Exception as e:
            logger.error(f"Failed to load robustness metrics: {e}")


# Global tracker instance
_robustness_tracker: RobustnessTracker | None = None


def get_robustness_tracker() -> RobustnessTracker:
    """Get or create global robustness tracker."""
    global _robustness_tracker
    if _robustness_tracker is None:
        _robustness_tracker = RobustnessTracker()
    return _robustness_tracker
