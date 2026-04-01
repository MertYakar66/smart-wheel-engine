"""
Provider Health Monitor

Pre-run health checks and continuous monitoring of:
- Browser session connectivity
- Model provider availability
- Local LLM (Ollama) status
- Rate limit tracking
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import aiohttp

from news_pipeline.browser_agents.base import ModelType, SessionStatus

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Working but slow/rate-limited
    UNHEALTHY = "unhealthy"  # Not responding
    UNKNOWN = "unknown"  # Not yet checked


@dataclass
class ProviderHealth:
    """Health status for a single provider."""

    provider: ModelType
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: datetime | None = None
    last_success: datetime | None = None
    response_time_ms: int | None = None
    error_message: str | None = None

    # Rate limiting
    rate_limited_until: datetime | None = None
    consecutive_failures: int = 0

    # Metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    @property
    def is_available(self) -> bool:
        """Check if provider is available for use."""
        if self.status == HealthStatus.UNHEALTHY:
            return False
        if self.rate_limited_until and datetime.utcnow() < self.rate_limited_until:
            return False
        return True

    @property
    def is_rate_limited(self) -> bool:
        """Check if provider is currently rate limited."""
        if self.rate_limited_until is None:
            return False
        return datetime.utcnow() < self.rate_limited_until

    @property
    def success_rate(self) -> float:
        """Get success rate (0-1)."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "provider": self.provider.value,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "is_available": self.is_available,
            "is_rate_limited": self.is_rate_limited,
            "success_rate": self.success_rate,
            "consecutive_failures": self.consecutive_failures,
        }


class ProviderHealthMonitor:
    """
    Monitors health of all providers.

    Pre-run checks:
    - Test browser session connectivity
    - Verify authentication
    - Check local LLM availability

    Continuous monitoring:
    - Track response times
    - Detect rate limiting
    - Update availability status
    """

    # Health check intervals
    CHECK_INTERVAL = timedelta(minutes=5)
    DEGRADED_THRESHOLD_MS = 10000  # 10 seconds
    RATE_LIMIT_COOLDOWN = timedelta(minutes=15)

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize health monitor.

        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self._health: dict[ModelType, ProviderHealth] = {
            model: ProviderHealth(provider=model)
            for model in ModelType
        }
        self._session_manager = None

    def get_health(self, provider: ModelType) -> ProviderHealth:
        """Get health status for a provider."""
        return self._health[provider]

    def get_all_health(self) -> dict[ModelType, ProviderHealth]:
        """Get health status for all providers."""
        return dict(self._health)

    def get_available_providers(self) -> list[ModelType]:
        """Get list of currently available providers."""
        return [
            model for model, health in self._health.items()
            if health.is_available
        ]

    async def run_health_checks(self) -> dict[ModelType, ProviderHealth]:
        """
        Run health checks on all providers.

        Returns:
            Dict of provider -> health status
        """
        logger.info("[Health] Running pre-run health checks...")

        # Check all providers in parallel
        checks = [
            self._check_local_llm(),
            self._check_browser_providers(),
        ]

        await asyncio.gather(*checks)

        # Log summary
        available = self.get_available_providers()
        logger.info(
            f"[Health] Check complete: {len(available)}/{len(ModelType)} providers available"
        )

        for provider in ModelType:
            health = self._health[provider]
            logger.debug(f"[Health] {provider.value}: {health.status.value}")

        return self.get_all_health()

    async def _check_local_llm(self) -> None:
        """Check local LLM (Ollama) availability."""
        health = self._health[ModelType.LOCAL]
        health.last_check = datetime.utcnow()

        try:
            start = datetime.utcnow()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    elapsed_ms = int(
                        (datetime.utcnow() - start).total_seconds() * 1000
                    )
                    health.response_time_ms = elapsed_ms

                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        if models:
                            health.status = HealthStatus.HEALTHY
                            health.last_success = datetime.utcnow()
                            health.error_message = None
                            health.consecutive_failures = 0
                        else:
                            health.status = HealthStatus.DEGRADED
                            health.error_message = "No models installed"
                    else:
                        health.status = HealthStatus.UNHEALTHY
                        health.error_message = f"HTTP {response.status}"

        except asyncio.TimeoutError:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = "Connection timeout"
            health.consecutive_failures += 1

        except aiohttp.ClientError as e:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = str(e)
            health.consecutive_failures += 1

        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Unexpected error: {e}"
            health.consecutive_failures += 1

    async def _check_browser_providers(self) -> None:
        """Check browser-based provider availability."""
        from news_pipeline.browser_agents.base import get_session_manager

        self._session_manager = get_session_manager()

        browser_providers = [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.GEMINI]

        for provider in browser_providers:
            health = self._health[provider]
            health.last_check = datetime.utcnow()

            try:
                # Check if we have an existing session
                if provider in self._session_manager.sessions:
                    session = self._session_manager.sessions[provider]

                    if session.state.status == SessionStatus.READY:
                        health.status = HealthStatus.HEALTHY
                        health.last_success = datetime.utcnow()
                        health.error_message = None
                    elif session.state.status == SessionStatus.RATE_LIMITED:
                        health.status = HealthStatus.DEGRADED
                        health.rate_limited_until = (
                            datetime.utcnow() + self.RATE_LIMIT_COOLDOWN
                        )
                        health.error_message = "Rate limited"
                    elif session.state.status == SessionStatus.AUTHENTICATED:
                        health.status = HealthStatus.HEALTHY
                        health.error_message = "Authenticated, not yet ready"
                    else:
                        health.status = HealthStatus.UNKNOWN
                        health.error_message = f"Status: {session.state.status.value}"
                else:
                    # No session - mark as unknown, will be initialized when needed
                    health.status = HealthStatus.UNKNOWN
                    health.error_message = "No active session"

            except Exception as e:
                health.status = HealthStatus.UNHEALTHY
                health.error_message = str(e)
                health.consecutive_failures += 1

    def record_request(
        self,
        provider: ModelType,
        success: bool,
        response_time_ms: int | None = None,
        rate_limited: bool = False,
    ) -> None:
        """
        Record a request result for tracking.

        Args:
            provider: Provider used
            success: Whether request succeeded
            response_time_ms: Response time in ms
            rate_limited: Whether rate limit was hit
        """
        health = self._health[provider]
        health.total_requests += 1

        if success:
            health.successful_requests += 1
            health.consecutive_failures = 0
            health.last_success = datetime.utcnow()

            if response_time_ms:
                health.response_time_ms = response_time_ms
                if response_time_ms > self.DEGRADED_THRESHOLD_MS:
                    health.status = HealthStatus.DEGRADED
                else:
                    health.status = HealthStatus.HEALTHY
        else:
            health.failed_requests += 1
            health.consecutive_failures += 1

            if rate_limited:
                health.rate_limited_until = datetime.utcnow() + self.RATE_LIMIT_COOLDOWN
                health.status = HealthStatus.DEGRADED
            elif health.consecutive_failures >= 3:
                health.status = HealthStatus.UNHEALTHY

    def mark_rate_limited(
        self,
        provider: ModelType,
        cooldown_minutes: int = 15,
    ) -> None:
        """Mark a provider as rate limited."""
        health = self._health[provider]
        health.status = HealthStatus.DEGRADED
        health.rate_limited_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
        health.error_message = f"Rate limited for {cooldown_minutes} minutes"

        logger.warning(f"[Health] {provider.value} rate limited for {cooldown_minutes}m")

    def mark_healthy(self, provider: ModelType) -> None:
        """Mark a provider as healthy."""
        health = self._health[provider]
        health.status = HealthStatus.HEALTHY
        health.last_success = datetime.utcnow()
        health.consecutive_failures = 0
        health.rate_limited_until = None
        health.error_message = None

    def mark_unhealthy(self, provider: ModelType, error: str) -> None:
        """Mark a provider as unhealthy."""
        health = self._health[provider]
        health.status = HealthStatus.UNHEALTHY
        health.error_message = error
        health.consecutive_failures += 1

        logger.warning(f"[Health] {provider.value} marked unhealthy: {error}")

    def get_best_provider(
        self,
        preferred: list[ModelType] | None = None,
        exclude: list[ModelType] | None = None,
    ) -> ModelType | None:
        """
        Get the best available provider.

        Args:
            preferred: Preferred providers (in order)
            exclude: Providers to exclude

        Returns:
            Best available provider or None
        """
        available = self.get_available_providers()
        exclude_set = set(exclude or [])
        candidates = [p for p in available if p not in exclude_set]

        if not candidates:
            return None

        # Check preferred first
        if preferred:
            for p in preferred:
                if p in candidates:
                    return p

        # Return first available, prioritizing by success rate
        candidates.sort(
            key=lambda p: (
                self._health[p].success_rate,
                -(self._health[p].response_time_ms or 0),
            ),
            reverse=True,
        )

        return candidates[0] if candidates else None

    def get_status_summary(self) -> dict:
        """Get a summary of all provider health."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available_count": len(self.get_available_providers()),
            "total_providers": len(ModelType),
            "providers": {
                provider.value: health.to_dict()
                for provider, health in self._health.items()
            },
        }


# Module-level instance
_health_monitor: ProviderHealthMonitor | None = None


def get_health_monitor() -> ProviderHealthMonitor:
    """Get the default health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ProviderHealthMonitor()
    return _health_monitor
