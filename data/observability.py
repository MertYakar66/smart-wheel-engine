"""
Observability - Logging, metrics, and tracing for production monitoring.

Provides:
- Structured logging with context
- Metrics collection (counters, gauges, histograms)
- Execution tracing with spans
- Alerting on thresholds
- Dashboard-ready exports

Usage:
    from data.observability import metrics, logger, trace

    # Log with context
    logger.info("Processing ticker", ticker="AAPL", rows=1000)

    # Record metrics
    metrics.increment("pipeline.tasks_completed", tags={"ticker": "AAPL"})
    metrics.timing("pipeline.duration_ms", duration)

    # Trace execution
    with trace.span("compute_features") as span:
        span.set_attribute("ticker", "AAPL")
        # ... do work
"""

import json
import logging
import sys
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import functools

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextLogger:
    """Logger with automatic context injection."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        self._local = threading.local()

    def _get_context(self) -> dict:
        thread_ctx = getattr(self._local, "context", {})
        return {**self._context, **thread_ctx}

    def _log(self, level: int, msg: str, **kwargs):
        extra = {"extra": {**self._get_context(), **kwargs}}
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

    def set_context(self, **kwargs):
        """Set persistent context for all logs."""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear persistent context."""
        self._context.clear()

    @contextmanager
    def context(self, **kwargs):
        """Temporary context for a block."""
        old_context = getattr(self._local, "context", {})
        self._local.context = {**old_context, **kwargs}
        try:
            yield
        finally:
            self._local.context = old_context


def setup_logging(
    level: int = logging.INFO,
    structured: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """Configure logging for the application."""
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)

    if structured:
        console.setFormatter(StructuredFormatter())
    else:
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

    root.addHandler(console)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)


# Metrics collection
class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class MetricValue:
    """A single metric value."""
    name: str
    type: MetricType
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """Thread-safe metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._tags: Dict[str, Dict[str, str]] = {}
        self._history: List[MetricValue] = []
        self._max_history = 10000

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}:{tag_str}"
        return name

    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._tags[key] = tags or {}
            self._record(name, MetricType.COUNTER, self._counters[key], tags)

    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
            self._tags[key] = tags or {}
            self._record(name, MetricType.GAUGE, value, tags)

    def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
            self._tags[key] = tags or {}
            self._record(name, MetricType.HISTOGRAM, value, tags)

    def timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._timings[key].append(duration_ms)
            self._tags[key] = tags or {}
            self._record(name, MetricType.TIMING, duration_ms, tags)

    def _record(
        self,
        name: str,
        type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]],
    ) -> None:
        """Record to history."""
        metric = MetricValue(name=name, type=type, value=value, tags=tags or {})
        self._history.append(metric)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing."""
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.timing(name, duration_ms, tags)

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key)

    def get_histogram_stats(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[dict]:
        """Get histogram statistics."""
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])

        if not values:
            return None

        import numpy as np
        arr = np.array(values)

        return {
            "count": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_timing_stats(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[dict]:
        """Get timing statistics."""
        key = self._make_key(name, tags)
        values = self._timings.get(key, [])

        if not values:
            return None

        import numpy as np
        arr = np.array(values)

        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "total_ms": float(arr.sum()),
        }

    def get_all_metrics(self) -> dict:
        """Get all current metrics."""
        with self._lock:
            result = {
                "counters": {},
                "gauges": {},
                "histograms": {},
                "timings": {},
            }

            for key, value in self._counters.items():
                result["counters"][key] = {"value": value, "tags": self._tags.get(key, {})}

            for key, value in self._gauges.items():
                result["gauges"][key] = {"value": value, "tags": self._tags.get(key, {})}

            for key, values in self._histograms.items():
                if values:
                    import numpy as np
                    arr = np.array(values)
                    result["histograms"][key] = {
                        "count": len(arr),
                        "mean": float(arr.mean()),
                        "p95": float(np.percentile(arr, 95)),
                        "tags": self._tags.get(key, {}),
                    }

            for key, values in self._timings.items():
                if values:
                    import numpy as np
                    arr = np.array(values)
                    result["timings"][key] = {
                        "count": len(arr),
                        "mean_ms": float(arr.mean()),
                        "p95_ms": float(np.percentile(arr, 95)),
                        "tags": self._tags.get(key, {}),
                    }

            return result

    def get_history(
        self,
        name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[MetricValue]:
        """Get metric history."""
        history = self._history

        if name:
            history = [m for m in history if m.name == name]

        if since:
            since_str = since.isoformat()
            history = [m for m in history if m.timestamp >= since_str]

        return history[-limit:]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timings.clear()
            self._tags.clear()
            self._history.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for key, value in self._counters.items():
            name = key.split(":")[0].replace(".", "_")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        for key, value in self._gauges.items():
            name = key.split(":")[0].replace(".", "_")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines)


# Tracing
@dataclass
class Span:
    """A trace span representing an operation."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attributes) -> None:
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes,
        })

    def set_error(self, error: str) -> None:
        self.status = "ERROR"
        self.error = error

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
        }


class Tracer:
    """Distributed tracing implementation."""

    def __init__(self):
        self._local = threading.local()
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._max_spans = 10000

    def _generate_id(self) -> str:
        import uuid
        return uuid.uuid4().hex[:16]

    def _get_current_span(self) -> Optional[Span]:
        stack = getattr(self._local, "span_stack", [])
        return stack[-1] if stack else None

    @contextmanager
    def span(self, name: str, **attributes):
        """Create a new span."""
        current = self._get_current_span()

        span = Span(
            trace_id=current.trace_id if current else self._generate_id(),
            span_id=self._generate_id(),
            parent_id=current.span_id if current else None,
            name=name,
            start_time=datetime.utcnow(),
            attributes=attributes,
        )

        # Push to stack
        stack = getattr(self._local, "span_stack", [])
        stack.append(span)
        self._local.span_stack = stack

        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            span.end_time = datetime.utcnow()

            # Pop from stack
            stack.pop()
            self._local.span_stack = stack

            # Record span
            with self._lock:
                self._spans.append(span)
                if len(self._spans) > self._max_spans:
                    self._spans = self._spans[-self._max_spans:]

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return [s for s in self._spans if s.trace_id == trace_id]

    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans."""
        return self._spans[-limit:]

    def get_slow_spans(
        self,
        threshold_ms: float = 1000,
        limit: int = 100,
    ) -> List[Span]:
        """Get spans slower than threshold."""
        slow = [
            s for s in self._spans
            if s.duration_ms and s.duration_ms > threshold_ms
        ]
        return sorted(slow, key=lambda s: s.duration_ms or 0, reverse=True)[:limit]

    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self._spans.clear()


# Alerting
class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert triggered by a threshold."""
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    """Manage alert rules and firing."""

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self._rules: List[dict] = []
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []

    def add_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "gt",  # gt, lt, gte, lte, eq
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: Optional[str] = None,
    ) -> None:
        """Add an alert rule."""
        self._rules.append({
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "message": message or f"{metric_name} {comparison} {threshold}",
        })

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)

    def check_rules(self) -> List[Alert]:
        """Check all rules and fire alerts."""
        new_alerts = []

        for rule in self._rules:
            value = self.metrics.get_gauge(rule["metric_name"])
            if value is None:
                value = self.metrics.get_counter(rule["metric_name"])

            if value is None:
                continue

            triggered = False
            comparison = rule["comparison"]
            threshold = rule["threshold"]

            if comparison == "gt" and value > threshold:
                triggered = True
            elif comparison == "lt" and value < threshold:
                triggered = True
            elif comparison == "gte" and value >= threshold:
                triggered = True
            elif comparison == "lte" and value <= threshold:
                triggered = True
            elif comparison == "eq" and value == threshold:
                triggered = True

            if triggered:
                alert = Alert(
                    name=rule["name"],
                    severity=rule["severity"],
                    message=rule["message"],
                    metric_name=rule["metric_name"],
                    metric_value=value,
                    threshold=threshold,
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

                # Call handlers
                for handler in self._handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logging.error(f"Alert handler failed: {e}")

        return new_alerts

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alerts with optional filters."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if since:
            since_str = since.isoformat()
            alerts = [a for a in alerts if a.timestamp >= since_str]

        return alerts


# Global instances
metrics = MetricsCollector()
trace = Tracer()
logger = ContextLogger("smart_wheel")


def timed(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with metrics.timer(metric_name, tags):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def traced(name: Optional[str] = None):
    """Decorator for tracing function execution."""
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace.span(span_name) as span:
                return func(*args, **kwargs)

        return wrapper
    return decorator
