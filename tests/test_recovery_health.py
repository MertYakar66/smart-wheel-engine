"""Tests for news_pipeline/recovery/health.py — sync paths only."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from news_pipeline.browser_agents.types import ModelType
from news_pipeline.recovery.health import (
    HealthStatus,
    ProviderHealth,
    ProviderHealthMonitor,
    get_health_monitor,
)


class TestProviderHealth:
    def test_default_unknown(self):
        h = ProviderHealth(provider=ModelType.LOCAL)
        assert h.status == HealthStatus.UNKNOWN
        assert h.consecutive_failures == 0

    def test_is_available_when_healthy(self):
        h = ProviderHealth(provider=ModelType.LOCAL, status=HealthStatus.HEALTHY)
        assert h.is_available is True

    def test_is_available_false_when_unhealthy(self):
        h = ProviderHealth(provider=ModelType.LOCAL, status=HealthStatus.UNHEALTHY)
        assert h.is_available is False

    def test_is_available_false_when_rate_limited(self):
        future = datetime.utcnow() + timedelta(minutes=10)
        h = ProviderHealth(
            provider=ModelType.LOCAL, status=HealthStatus.HEALTHY,
            rate_limited_until=future,
        )
        assert h.is_available is False
        assert h.is_rate_limited is True

    def test_rate_limit_expired_is_available(self):
        past = datetime.utcnow() - timedelta(minutes=10)
        h = ProviderHealth(
            provider=ModelType.LOCAL, status=HealthStatus.HEALTHY,
            rate_limited_until=past,
        )
        assert h.is_rate_limited is False
        assert h.is_available is True

    def test_success_rate_no_requests(self):
        h = ProviderHealth(provider=ModelType.LOCAL)
        assert h.success_rate == 1.0

    def test_success_rate_partial(self):
        h = ProviderHealth(
            provider=ModelType.LOCAL,
            total_requests=10, successful_requests=7,
        )
        assert h.success_rate == 0.7

    def test_to_dict_complete(self):
        ts = datetime.utcnow()
        h = ProviderHealth(
            provider=ModelType.LOCAL, status=HealthStatus.HEALTHY,
            last_check=ts, last_success=ts, response_time_ms=100,
        )
        d = h.to_dict()
        assert d["provider"] == "local"
        assert d["status"] == "healthy"
        assert d["response_time_ms"] == 100
        assert "is_available" in d
        assert "success_rate" in d


class TestProviderHealthMonitor:
    @pytest.fixture
    def mon(self) -> ProviderHealthMonitor:
        return ProviderHealthMonitor()

    def test_init_creates_health_for_all_providers(self, mon: ProviderHealthMonitor):
        for model in ModelType:
            assert model in mon._health
            assert mon.get_health(model).provider == model

    def test_get_all_health_returns_dict(self, mon: ProviderHealthMonitor):
        all_health = mon.get_all_health()
        assert len(all_health) == len(ModelType)

    def test_get_available_providers_starts_with_all(self, mon: ProviderHealthMonitor):
        # All providers default to UNKNOWN status which is_available=True
        available = mon.get_available_providers()
        assert len(available) == len(ModelType)

    def test_record_request_success(self, mon: ProviderHealthMonitor):
        mon.record_request(ModelType.LOCAL, success=True, response_time_ms=50)
        h = mon.get_health(ModelType.LOCAL)
        assert h.total_requests == 1
        assert h.successful_requests == 1
        assert h.consecutive_failures == 0
        assert h.status == HealthStatus.HEALTHY

    def test_record_request_slow_response_marks_degraded(self, mon: ProviderHealthMonitor):
        # Threshold is 10000ms
        mon.record_request(ModelType.LOCAL, success=True, response_time_ms=15000)
        assert mon.get_health(ModelType.LOCAL).status == HealthStatus.DEGRADED

    def test_record_request_failure_increments_counter(self, mon: ProviderHealthMonitor):
        mon.record_request(ModelType.LOCAL, success=False)
        h = mon.get_health(ModelType.LOCAL)
        assert h.failed_requests == 1
        assert h.consecutive_failures == 1

    def test_three_consecutive_failures_marks_unhealthy(self, mon: ProviderHealthMonitor):
        for _ in range(3):
            mon.record_request(ModelType.LOCAL, success=False)
        assert mon.get_health(ModelType.LOCAL).status == HealthStatus.UNHEALTHY

    def test_rate_limited_request_marks_degraded(self, mon: ProviderHealthMonitor):
        mon.record_request(ModelType.CHATGPT, success=False, rate_limited=True)
        h = mon.get_health(ModelType.CHATGPT)
        assert h.status == HealthStatus.DEGRADED
        assert h.rate_limited_until is not None

    def test_mark_rate_limited(self, mon: ProviderHealthMonitor):
        mon.mark_rate_limited(ModelType.CHATGPT, cooldown_minutes=5)
        h = mon.get_health(ModelType.CHATGPT)
        assert h.status == HealthStatus.DEGRADED
        assert h.is_rate_limited is True

    def test_mark_healthy_resets_counters(self, mon: ProviderHealthMonitor):
        mon.record_request(ModelType.LOCAL, success=False)
        mon.record_request(ModelType.LOCAL, success=False)
        mon.mark_healthy(ModelType.LOCAL)
        h = mon.get_health(ModelType.LOCAL)
        assert h.status == HealthStatus.HEALTHY
        assert h.consecutive_failures == 0
        assert h.rate_limited_until is None

    def test_mark_unhealthy(self, mon: ProviderHealthMonitor):
        mon.mark_unhealthy(ModelType.CLAUDE, "test error")
        h = mon.get_health(ModelType.CLAUDE)
        assert h.status == HealthStatus.UNHEALTHY
        assert h.error_message == "test error"

    def test_get_best_provider_with_preferred(self, mon: ProviderHealthMonitor):
        for m in ModelType:
            mon.mark_healthy(m)
        best = mon.get_best_provider(preferred=[ModelType.CLAUDE, ModelType.CHATGPT])
        assert best == ModelType.CLAUDE

    def test_get_best_provider_skip_excluded(self, mon: ProviderHealthMonitor):
        for m in ModelType:
            mon.mark_healthy(m)
        best = mon.get_best_provider(
            preferred=[ModelType.CLAUDE, ModelType.CHATGPT],
            exclude=[ModelType.CLAUDE],
        )
        assert best == ModelType.CHATGPT

    def test_get_best_provider_none_when_all_unavailable(self, mon: ProviderHealthMonitor):
        for m in ModelType:
            mon.mark_unhealthy(m, "down")
        assert mon.get_best_provider() is None

    def test_get_best_provider_falls_through_to_success_rate_sort(self, mon: ProviderHealthMonitor):
        for m in ModelType:
            mon.mark_healthy(m)
        # No preferred, no exclude → returns the first by success-rate sort
        best = mon.get_best_provider()
        assert best is not None

    def test_get_status_summary(self, mon: ProviderHealthMonitor):
        summary = mon.get_status_summary()
        assert "timestamp" in summary
        assert "available_count" in summary
        assert "providers" in summary
        assert len(summary["providers"]) == len(ModelType)


class TestGetHealthMonitor:
    def test_returns_singleton(self):
        m1 = get_health_monitor()
        m2 = get_health_monitor()
        assert m1 is m2
