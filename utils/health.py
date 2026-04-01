"""
Health Check Endpoints and System Status

Provides production-grade health checks for monitoring and orchestration.
Follows Kubernetes health check patterns (liveness, readiness, startup).

Usage:
    from utils.health import HealthChecker, get_health_status

    health = HealthChecker()
    health.register_check("database", check_database_connection)

    # In your API:
    @app.get("/health")
    def health_endpoint():
        return health.get_status()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Type of health check."""
    LIVENESS = "liveness"    # Is the service alive?
    READINESS = "readiness"  # Is the service ready to accept traffic?
    STARTUP = "startup"      # Has the service finished starting?
    DEPENDENCY = "dependency"  # Is an external dependency healthy?


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    check_type: CheckType
    message: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "type": self.check_type.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: list[CheckResult]
    uptime_seconds: float
    version: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


# Type aliases for check functions
SyncCheckFn = Callable[[], tuple[HealthStatus, str, dict]]
AsyncCheckFn = Callable[[], tuple[HealthStatus, str, dict]]


class HealthChecker:
    """
    Manages health checks for the application.

    Features:
    - Register sync and async health checks
    - Configurable check timeouts
    - Caching to prevent excessive checks
    - Aggregate status calculation
    """

    def __init__(
        self,
        version: str = "unknown",
        cache_ttl_seconds: float = 5.0,
        default_timeout_seconds: float = 10.0,
    ):
        """
        Initialize health checker.

        Args:
            version: Application version string
            cache_ttl_seconds: How long to cache check results
            default_timeout_seconds: Default timeout for checks
        """
        self.version = version
        self.cache_ttl = cache_ttl_seconds
        self.default_timeout = default_timeout_seconds
        self.start_time = time.time()

        self._checks: dict[str, tuple[CheckType, SyncCheckFn | AsyncCheckFn, bool]] = {}
        self._cache: dict[str, tuple[CheckResult, float]] = {}

    def register_check(
        self,
        name: str,
        check_fn: SyncCheckFn | AsyncCheckFn,
        check_type: CheckType = CheckType.DEPENDENCY,
        is_async: bool = False,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Unique name for the check
            check_fn: Function that returns (HealthStatus, message, details)
            check_type: Type of check
            is_async: Whether check_fn is async
        """
        self._checks[name] = (check_type, check_fn, is_async)
        logger.info(f"Registered health check: {name} ({check_type.value})")

    def unregister_check(self, name: str) -> None:
        """Remove a health check."""
        if name in self._checks:
            del self._checks[name]
            if name in self._cache:
                del self._cache[name]
            logger.info(f"Unregistered health check: {name}")

    def get_status(self, force_refresh: bool = False) -> SystemHealth:
        """
        Get current system health status (sync version).

        Args:
            force_refresh: If True, bypass cache

        Returns:
            SystemHealth object
        """
        results = []

        for name, (check_type, check_fn, is_async) in self._checks.items():
            # Check cache
            if not force_refresh and name in self._cache:
                cached_result, cache_time = self._cache[name]
                if time.time() - cache_time < self.cache_ttl:
                    results.append(cached_result)
                    continue

            # Run check
            start = time.time()
            try:
                if is_async:
                    # Run async check in new event loop
                    loop = asyncio.new_event_loop()
                    try:
                        status, message, details = loop.run_until_complete(
                            asyncio.wait_for(check_fn(), timeout=self.default_timeout)
                        )
                    finally:
                        loop.close()
                else:
                    status, message, details = check_fn()
            except asyncio.TimeoutError:
                status = HealthStatus.UNHEALTHY
                message = f"Check timed out after {self.default_timeout}s"
                details = {}
            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"Check failed: {str(e)}"
                details = {"error": str(e)}
                logger.exception(f"Health check {name} failed")

            latency = (time.time() - start) * 1000

            result = CheckResult(
                name=name,
                status=status,
                check_type=check_type,
                message=message,
                latency_ms=latency,
                details=details,
            )
            results.append(result)
            self._cache[name] = (result, time.time())

        # Calculate aggregate status
        aggregate_status = self._calculate_aggregate_status(results)

        return SystemHealth(
            status=aggregate_status,
            checks=results,
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
        )

    async def get_status_async(self, force_refresh: bool = False) -> SystemHealth:
        """
        Get current system health status (async version).

        Args:
            force_refresh: If True, bypass cache

        Returns:
            SystemHealth object
        """
        results = []

        for name, (check_type, check_fn, is_async) in self._checks.items():
            # Check cache
            if not force_refresh and name in self._cache:
                cached_result, cache_time = self._cache[name]
                if time.time() - cache_time < self.cache_ttl:
                    results.append(cached_result)
                    continue

            # Run check
            start = time.time()
            try:
                if is_async:
                    status, message, details = await asyncio.wait_for(
                        check_fn(), timeout=self.default_timeout
                    )
                else:
                    status, message, details = check_fn()
            except asyncio.TimeoutError:
                status = HealthStatus.UNHEALTHY
                message = f"Check timed out after {self.default_timeout}s"
                details = {}
            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"Check failed: {str(e)}"
                details = {"error": str(e)}
                logger.exception(f"Health check {name} failed")

            latency = (time.time() - start) * 1000

            result = CheckResult(
                name=name,
                status=status,
                check_type=check_type,
                message=message,
                latency_ms=latency,
                details=details,
            )
            results.append(result)
            self._cache[name] = (result, time.time())

        aggregate_status = self._calculate_aggregate_status(results)

        return SystemHealth(
            status=aggregate_status,
            checks=results,
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
        )

    def _calculate_aggregate_status(self, results: list[CheckResult]) -> HealthStatus:
        """Calculate aggregate health status from individual checks."""
        if not results:
            return HealthStatus.UNKNOWN

        # Critical checks (liveness, startup) must all pass
        critical_checks = [
            r for r in results
            if r.check_type in (CheckType.LIVENESS, CheckType.STARTUP)
        ]
        if any(r.status == HealthStatus.UNHEALTHY for r in critical_checks):
            return HealthStatus.UNHEALTHY

        # Count statuses
        statuses = [r.status for r in results]
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)

        # Determine overall status
        if unhealthy_count > 0:
            # If any non-critical check is unhealthy, we're degraded
            return HealthStatus.DEGRADED
        if degraded_count > 0:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# =============================================================================
# COMMON HEALTH CHECKS
# =============================================================================

def check_disk_space(
    path: str = "/",
    min_free_gb: float = 1.0
) -> tuple[HealthStatus, str, dict]:
    """Check if disk has enough free space."""
    import shutil

    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        used_percent = (used / total) * 100

        details = {
            "path": path,
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_percent, 1),
        }

        if free_gb < min_free_gb:
            return (
                HealthStatus.UNHEALTHY,
                f"Low disk space: {free_gb:.1f}GB free",
                details
            )
        if free_gb < min_free_gb * 2:
            return (
                HealthStatus.DEGRADED,
                f"Disk space warning: {free_gb:.1f}GB free",
                details
            )
        return (
            HealthStatus.HEALTHY,
            f"Disk OK: {free_gb:.1f}GB free",
            details
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"Could not check disk: {e}",
            {}
        )


def check_memory(
    max_percent: float = 90.0
) -> tuple[HealthStatus, str, dict]:
    """Check memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()

        details = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_percent": round(mem.percent, 1),
        }

        if mem.percent > max_percent:
            return (
                HealthStatus.UNHEALTHY,
                f"High memory usage: {mem.percent:.1f}%",
                details
            )
        if mem.percent > max_percent * 0.9:
            return (
                HealthStatus.DEGRADED,
                f"Memory warning: {mem.percent:.1f}%",
                details
            )
        return (
            HealthStatus.HEALTHY,
            f"Memory OK: {mem.percent:.1f}% used",
            details
        )
    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not installed",
            {}
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"Could not check memory: {e}",
            {}
        )


async def check_url(
    url: str,
    timeout_seconds: float = 5.0,
    expected_status: int = 200,
) -> tuple[HealthStatus, str, dict]:
    """Check if a URL is reachable."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as resp:
                latency = (time.time() - start) * 1000
                details = {
                    "url": url,
                    "status_code": resp.status,
                    "latency_ms": round(latency, 2),
                }

                if resp.status == expected_status:
                    return (
                        HealthStatus.HEALTHY,
                        f"URL OK: {resp.status}",
                        details
                    )
                return (
                    HealthStatus.DEGRADED,
                    f"Unexpected status: {resp.status}",
                    details
                )
    except asyncio.TimeoutError:
        return (
            HealthStatus.UNHEALTHY,
            f"URL timeout after {timeout_seconds}s",
            {"url": url}
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"URL check failed: {e}",
            {"url": url, "error": str(e)}
        )


# =============================================================================
# GLOBAL HEALTH CHECKER INSTANCE
# =============================================================================

_health_checker: HealthChecker | None = None


def get_health_checker(version: str = "unknown") -> HealthChecker:
    """Get or create global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(version=version)

        # Register default checks
        _health_checker.register_check(
            "disk_space",
            lambda: check_disk_space("/", min_free_gb=1.0),
            check_type=CheckType.LIVENESS,
        )

    return _health_checker


def get_health_status() -> dict:
    """Convenience function to get health status as dict."""
    checker = get_health_checker()
    return checker.get_status().to_dict()
