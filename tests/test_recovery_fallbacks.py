"""Tests for news_pipeline/recovery/fallbacks.py."""

from __future__ import annotations

import asyncio

import pytest

from news_pipeline.browser_agents.types import ModelType
from news_pipeline.recovery.checkpoints import PipelineStage
from news_pipeline.recovery.fallbacks import (
    DegradedMode,
    DegradedModeConfig,
    FallbackHandler,
    FallbackResult,
    get_fallback_handler,
)
from news_pipeline.recovery.health import ProviderHealthMonitor


class TestDegradedModeConfig:
    def test_default_normal(self):
        cfg = DegradedModeConfig()
        assert cfg.mode == DegradedMode.NORMAL
        assert cfg.skip_verification is False
        assert cfg.use_local_verification is True

    def test_to_dict(self):
        cfg = DegradedModeConfig(
            mode=DegradedMode.PARTIAL,
            skip_verification=True,
            manual_review_stages=[PipelineStage.PUBLISH],
        )
        d = cfg.to_dict()
        assert d["mode"] == "partial"
        assert d["skip_verification"] is True
        assert "publish" in d["manual_review_stages"]


class TestFallbackResult:
    def test_default_construction(self):
        r = FallbackResult(success=True, provider_used=ModelType.CLAUDE)
        assert r.success is True
        assert r.providers_tried == []
        assert r.degraded_mode == DegradedMode.NORMAL


class TestFallbackHandler:
    @pytest.fixture
    def handler(self) -> FallbackHandler:
        # Fresh health monitor per test (avoid singleton state leak)
        return FallbackHandler(health_monitor=ProviderHealthMonitor())

    def test_init_default_mode_is_normal(self, handler: FallbackHandler):
        assert handler.current_mode == DegradedMode.NORMAL

    def test_evaluate_mode_normal_when_all_available(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)
        mode = handler.evaluate_mode()
        assert mode == DegradedMode.NORMAL

    def test_evaluate_mode_local_only(self, handler: FallbackHandler):
        # Only LOCAL is healthy
        for m in ModelType:
            handler.health_monitor.mark_unhealthy(m, "down")
        handler.health_monitor.mark_healthy(ModelType.LOCAL)
        mode = handler.evaluate_mode()
        assert mode == DegradedMode.LOCAL_ONLY

    def test_evaluate_mode_offline(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_unhealthy(m, "down")
        mode = handler.evaluate_mode()
        assert mode == DegradedMode.OFFLINE

    def test_evaluate_mode_partial(self, handler: FallbackHandler):
        # Some unhealthy, but more than just LOCAL is healthy
        handler.health_monitor.mark_unhealthy(ModelType.CHATGPT, "down")
        # Others remain HEALTHY (UNKNOWN counts as available)
        mode = handler.evaluate_mode()
        # Could be PARTIAL or NORMAL depending on remaining count
        assert mode in (DegradedMode.PARTIAL, DegradedMode.NORMAL)

    def test_configure_degraded_mode_sets_skip_flags(self, handler: FallbackHandler):
        cfg = handler.configure_degraded_mode(
            skip_verification=True,
            skip_editorial=True,
            manual_stages=[PipelineStage.PUBLISH],
        )
        assert cfg.skip_verification is True
        assert cfg.skip_editorial is True
        assert PipelineStage.PUBLISH in cfg.manual_review_stages

    def test_configure_degraded_mode_local_only_auto_locals(self, handler: FallbackHandler):
        handler.configure_degraded_mode(mode=DegradedMode.LOCAL_ONLY)
        assert handler.config.use_local_verification is True
        assert handler.config.use_local_formatting is True

    def test_configure_degraded_mode_offline_skips(self, handler: FallbackHandler):
        handler.configure_degraded_mode(mode=DegradedMode.OFFLINE)
        assert handler.config.skip_verification is True
        assert handler.config.skip_editorial is True

    def test_get_fallback_chain_default(self, handler: FallbackHandler):
        chain = handler.get_fallback_chain("verification")
        assert ModelType.CLAUDE in chain
        assert ModelType.LOCAL in chain  # always last fallback

    def test_get_fallback_chain_unknown_task_uses_default(self, handler: FallbackHandler):
        chain = handler.get_fallback_chain("nonexistent_task")
        # Falls back to a default chain (not empty)
        assert len(chain) > 0

    def test_get_fallback_chain_custom(self, handler: FallbackHandler):
        handler.config.fallback_chains["custom"] = [ModelType.GEMINI, ModelType.LOCAL]
        chain = handler.get_fallback_chain("custom")
        assert chain == [ModelType.GEMINI, ModelType.LOCAL]

    def test_get_next_provider_picks_first_available(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)
        nxt = handler.get_next_provider("verification")
        assert nxt is not None

    def test_get_next_provider_skips_tried(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)
        nxt = handler.get_next_provider("verification", tried=[ModelType.CLAUDE])
        assert nxt != ModelType.CLAUDE

    def test_get_next_provider_none_when_all_tried(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)
        all_models = list(ModelType)
        nxt = handler.get_next_provider("verification", tried=all_models)
        assert nxt is None

    def test_should_skip_stage_offline_skips_ai_stages(self, handler: FallbackHandler):
        handler.configure_degraded_mode(mode=DegradedMode.OFFLINE)
        assert handler.should_skip_stage(PipelineStage.VERIFY) is True
        assert handler.should_skip_stage(PipelineStage.FORMAT) is True
        assert handler.should_skip_stage(PipelineStage.EDITORIAL) is True
        # Non-AI stages still run
        assert handler.should_skip_stage(PipelineStage.SCRAPE) is False

    def test_should_skip_stage_individual_flags(self, handler: FallbackHandler):
        handler.configure_degraded_mode(skip_verification=True)
        assert handler.should_skip_stage(PipelineStage.VERIFY) is True
        assert handler.should_skip_stage(PipelineStage.FORMAT) is False

    def test_requires_manual_review(self, handler: FallbackHandler):
        handler.configure_degraded_mode(manual_stages=[PipelineStage.PUBLISH])
        assert handler.requires_manual_review(PipelineStage.PUBLISH) is True
        assert handler.requires_manual_review(PipelineStage.SCRAPE) is False

    def test_manual_queue_lifecycle(self, handler: FallbackHandler):
        handler.queue_for_manual_review(PipelineStage.VERIFY, {"x": 1}, "needs review")
        q = handler.get_manual_queue()
        assert len(q) == 1
        assert q[0]["stage"] == "verify"
        assert q[0]["item"] == {"x": 1}
        cleared = handler.clear_manual_queue()
        assert cleared == 1
        assert handler.get_manual_queue() == []

    def test_enter_exit_manual_mode(self, handler: FallbackHandler):
        handler.enter_manual_mode("test")
        assert handler.config.mode == DegradedMode.MANUAL
        assert handler.config.await_manual_input is True
        handler.exit_manual_mode()
        # Mode re-evaluated; await_manual_input cleared
        assert handler.config.await_manual_input is False

    def test_partial_completion_plan(self, handler: FallbackHandler):
        plan = handler.get_partial_completion_plan(PipelineStage.VERIFY)
        assert plan["failed_stage"] == "verify"
        assert "can_publish_raw" in plan
        assert "recommendations" in plan

    def test_partial_completion_plan_includes_local_when_available(self, handler: FallbackHandler):
        handler.health_monitor.mark_healthy(ModelType.LOCAL)
        plan = handler.get_partial_completion_plan(PipelineStage.FORMAT)
        # Should include local-LLM recommendation
        assert any("local" in r.lower() for r in plan["recommendations"])

    def test_execute_with_fallback_success_first_try(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)

        async def fn(provider):
            return {"data": "ok", "by": provider.value}

        result = asyncio.run(handler.execute_with_fallback("verification", fn))
        assert result.success is True
        assert result.result["data"] == "ok"
        assert len(result.providers_tried) == 1

    def test_execute_with_fallback_all_fail(self, handler: FallbackHandler):
        for m in ModelType:
            handler.health_monitor.mark_healthy(m)

        async def fn(provider):
            raise RuntimeError("always fails")

        result = asyncio.run(
            handler.execute_with_fallback(
                "verification",
                fn,
                max_retries=1,
            )
        )
        assert result.success is False
        assert result.error is not None
        # Should have tried multiple providers
        assert len(result.providers_tried) > 0


class TestGetFallbackHandler:
    def test_returns_singleton(self):
        h1 = get_fallback_handler()
        h2 = get_fallback_handler()
        assert h1 is h2
