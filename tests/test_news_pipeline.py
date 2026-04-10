"""
Tests for the browser-based news pipeline.

Tests cover:
- Data models and schema
- Browser agent interfaces
- Orchestrator logic
- Publisher functionality
- Security and recovery modules
"""

import asyncio
import importlib.util
import json
from datetime import datetime

import pytest

from news_pipeline.models import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    PipelineResult,
    PipelineStage,
    StoryCategory,
    VerificationResult,
    VerificationStatus,
)


class TestDataModels:
    """Tests for pipeline data models."""

    def test_discovery_request_immutable(self):
        """DiscoveryRequest should be immutable."""
        request = DiscoveryRequest(
            tickers=["AAPL", "NVDA"],
            categories=["fed", "earnings"],
            time_window="overnight",
        )

        assert request.tickers == ("AAPL", "NVDA")
        assert request.categories == ("fed", "earnings")
        assert request.time_window == "overnight"

        # Should raise error on mutation attempt
        with pytest.raises(AttributeError):
            request.time_window = "today"

    def test_discovery_request_id_generation(self):
        """DiscoveryRequest should generate consistent IDs."""
        request1 = DiscoveryRequest(tickers=["AAPL"], categories=["fed"])
        request2 = DiscoveryRequest(tickers=["AAPL"], categories=["fed"])
        request3 = DiscoveryRequest(tickers=["NVDA"], categories=["fed"])

        assert request1.request_id == request2.request_id
        assert request1.request_id != request3.request_id

    def test_candidate_story_serialization(self):
        """CandidateStory should serialize to dict."""
        story = CandidateStory(
            story_id="abc123",
            headline="Fed raises rates",
            source_name="Reuters",
            source_url="https://reuters.com/fed",
            snippet="The Federal Reserve...",
            discovered_at=datetime(2024, 1, 15, 10, 30),
            tickers=["SPY", "TLT"],
            category="fed",
        )

        data = story.to_dict()

        assert data["story_id"] == "abc123"
        assert data["headline"] == "Fed raises rates"
        assert data["tickers"] == ["SPY", "TLT"]
        assert "2024-01-15" in data["discovered_at"]

    def test_candidate_story_from_dict(self):
        """CandidateStory should deserialize from dict."""
        data = {
            "story_id": "abc123",
            "headline": "Fed raises rates",
            "source_name": "Reuters",
            "source_url": "https://reuters.com/fed",
            "snippet": "The Federal Reserve...",
            "discovered_at": "2024-01-15T10:30:00",
            "tickers": ["SPY", "TLT"],
            "category": "fed",
        }

        story = CandidateStory.from_dict(data)

        assert story.story_id == "abc123"
        assert story.headline == "Fed raises rates"
        assert isinstance(story.discovered_at, datetime)

    def test_verification_result_publishable(self):
        """VerificationResult should correctly determine publishability."""
        candidate = CandidateStory(
            story_id="test",
            headline="Test",
            source_name="Test",
            source_url="http://test.com",
            snippet="Test",
            discovered_at=datetime.utcnow(),
        )

        # High confidence, verified - publishable
        result1 = VerificationResult(
            story_id="test",
            candidate=candidate,
            status=VerificationStatus.VERIFIED,
            confidence=8,
        )
        assert result1.is_publishable is True

        # Low confidence - not publishable
        result2 = VerificationResult(
            story_id="test",
            candidate=candidate,
            status=VerificationStatus.VERIFIED,
            confidence=4,
        )
        assert result2.is_publishable is False

        # Contradicted - not publishable
        result3 = VerificationResult(
            story_id="test",
            candidate=candidate,
            status=VerificationStatus.CONTRADICTED,
            confidence=8,
        )
        assert result3.is_publishable is False

    def test_finalized_story_json_serialization(self):
        """FinalizedStory should serialize to JSON."""
        story = FinalizedStory(
            story_id="xyz789",
            title="Major Fed Decision",
            what_happened="The Fed announced a 25bps rate cut.",
            why_it_matters="This signals a pivot in monetary policy.",
            bullet_points=["Rate cut of 25bps", "More cuts expected"],
            affected_assets=["SPY", "QQQ", "TLT"],
            category="fed",
            verification_confidence=9,
            is_breaking=True,
            priority=9,
        )

        json_str = story.to_json()
        data = json.loads(json_str)

        assert data["story_id"] == "xyz789"
        assert data["is_breaking"] is True
        assert data["priority"] == 9
        assert len(data["bullet_points"]) == 2

    def test_finalized_story_from_dict(self):
        """FinalizedStory should deserialize from dict."""
        data = {
            "story_id": "xyz789",
            "title": "Major Fed Decision",
            "what_happened": "The Fed announced a rate cut.",
            "why_it_matters": "Policy pivot.",
            "bullet_points": ["Point 1"],
            "affected_assets": ["SPY"],
            "category": "fed",
            "verification_confidence": 9,
            "finalized_at": "2024-01-15T10:30:00",
            "is_breaking": True,
            "priority": 9,
        }

        story = FinalizedStory.from_dict(data)

        assert story.story_id == "xyz789"
        assert story.is_breaking is True
        assert isinstance(story.finalized_at, datetime)

    def test_pipeline_result_metrics(self):
        """PipelineResult should calculate metrics correctly."""
        result = PipelineResult(
            run_id="test123",
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            completed_at=datetime(2024, 1, 15, 10, 1, 30),
            discovered_count=50,
            verified_count=30,
            published_count=25,
        )

        assert result.duration_seconds == 90.0
        assert result.success_rate == 0.5  # 25/50


# Check if playwright is available for browser-based tests
HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
class TestOrchestratorConfig:
    """Tests for orchestrator configuration (requires playwright)."""

    def test_orchestrator_config_defaults(self):
        """OrchestratorConfig should have sensible defaults."""
        from news_pipeline.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.min_confidence == 6
        assert config.max_stories_per_run == 20
        assert config.headless is True
        assert config.use_local_llm is True
        assert config.enable_checkpoints is True
        assert config.enable_health_checks is True

    def test_orchestrator_config_security_settings(self):
        """OrchestratorConfig should have security settings."""
        from news_pipeline.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_sensitivity_check is True
        assert config.strict_security_mode is True
        assert config.auto_sanitize is True

    def test_orchestrator_config_recovery_settings(self):
        """OrchestratorConfig should have recovery settings."""
        from news_pipeline.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.auto_resume is True
        assert config.max_retries_per_stage == 3
        assert config.allow_degraded_mode is True


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
class TestBrowserAgents:
    """Tests for browser-based agents (requires playwright)."""

    def test_model_type_enum(self):
        """ModelType should have correct values."""
        from news_pipeline.browser_agents import ModelType

        assert ModelType.CLAUDE.value == "claude"
        assert ModelType.CHATGPT.value == "chatgpt"
        assert ModelType.GEMINI.value == "gemini"
        assert ModelType.LOCAL.value == "local"

    def test_session_status_enum(self):
        """SessionStatus should have correct values."""
        from news_pipeline.browser_agents.base import SessionStatus

        assert SessionStatus.UNINITIALIZED.value == "uninitialized"
        assert SessionStatus.AUTHENTICATED.value == "authenticated"
        assert SessionStatus.READY.value == "ready"
        assert SessionStatus.RATE_LIMITED.value == "rate_limited"
        assert SessionStatus.ERROR.value == "error"

    def test_model_response_dataclass(self):
        """ModelResponse should store response data."""
        from news_pipeline.browser_agents.base import ModelResponse

        response = ModelResponse(
            success=True,
            content="Test response",
            response_time_ms=150,
        )

        assert response.success is True
        assert response.content == "Test response"
        assert response.response_time_ms == 150


class TestSecurityModule:
    """Tests for security classification and sanitization."""

    def test_data_sensitivity_tiers(self):
        """DataSensitivity should have three tiers."""
        from news_pipeline.security import DataSensitivity

        assert DataSensitivity.TIER_A.value == "public"
        assert DataSensitivity.TIER_B.value == "sanitizable"
        assert DataSensitivity.TIER_C.value == "private"

    def test_classifier_public_news(self):
        """Classifier should identify public news as Tier A."""
        from news_pipeline.security import SensitivityClassifier

        classifier = SensitivityClassifier()

        result = classifier.classify(
            "Fed announces 25bps rate cut, Powell signals more easing ahead"
        )

        assert result.tier.value == "public"
        assert result.is_safe_external is True

    def test_classifier_sensitive_data(self):
        """Classifier should identify sensitive data as Tier C."""
        from news_pipeline.security import SensitivityClassifier

        classifier = SensitivityClassifier()

        result = classifier.classify("API key: sk-1234567890abcdef password: secret123")

        assert result.tier.value == "private"
        assert result.is_private is True

    def test_sanitizer_removes_email(self):
        """Sanitizer should redact email addresses."""
        from news_pipeline.security import Sanitizer

        sanitizer = Sanitizer()

        result = sanitizer.sanitize("Contact john.doe@company.com for details")

        assert "john.doe@company.com" not in result.sanitized
        assert "[email]" in result.sanitized
        assert result.was_modified is True

    def test_sanitizer_removes_phone(self):
        """Sanitizer should redact phone numbers."""
        from news_pipeline.security import Sanitizer

        sanitizer = Sanitizer()

        result = sanitizer.sanitize("Call us at 555-123-4567 for support")

        assert "555-123-4567" not in result.sanitized
        assert "[phone]" in result.sanitized

    def test_routing_policy_external_ok(self):
        """RoutingPolicy should allow external routing for public data."""
        from news_pipeline.security import RoutingDecision, RoutingPolicy

        policy = RoutingPolicy()

        result = policy.evaluate(
            "Reuters: Fed raises rates by 25 basis points",
            task="verification",
        )

        assert result.decision == RoutingDecision.EXTERNAL_OK
        assert result.can_use_external is True


class TestRecoveryModule:
    """Tests for checkpointing and recovery."""

    def test_pipeline_stage_order(self):
        """PipelineStage should have correct ordering."""
        from news_pipeline.recovery import PipelineStage

        assert PipelineStage.INIT.order == 0
        assert PipelineStage.SCRAPE.order == 1
        assert PipelineStage.PREPROCESS.order == 2
        assert PipelineStage.VERIFY.order == 3
        assert PipelineStage.FORMAT.order == 4
        assert PipelineStage.EDITORIAL.order == 5
        assert PipelineStage.PUBLISH.order == 6
        assert PipelineStage.COMPLETE.order == 7

    def test_checkpoint_serialization(self):
        """Checkpoint should serialize to dict."""
        from news_pipeline.recovery import Checkpoint, PipelineStage

        checkpoint = Checkpoint(
            stage=PipelineStage.VERIFY,
            timestamp=datetime(2024, 1, 15, 10, 30),
            run_id="test123",
            data={"items": ["a", "b"]},
            items_processed=2,
            items_total=5,
        )

        data = checkpoint.to_dict()

        assert data["stage"] == "verify"
        assert data["run_id"] == "test123"
        assert data["items_processed"] == 2
        assert checkpoint.progress == 0.4

    def test_health_status_enum(self):
        """HealthStatus should have correct values."""
        from news_pipeline.recovery import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_provider_health_availability(self):
        """ProviderHealth should track availability."""
        from news_pipeline.browser_agents import ModelType
        from news_pipeline.recovery import HealthStatus, ProviderHealth

        health = ProviderHealth(
            provider=ModelType.CLAUDE,
            status=HealthStatus.HEALTHY,
        )

        assert health.is_available is True

        health.status = HealthStatus.UNHEALTHY
        assert health.is_available is False


class TestPublisher:
    """Tests for news publisher."""

    def test_publisher_database_mode(self, tmp_path):
        """Publisher should work in database mode."""
        from news_pipeline.publisher import NewsPublisher

        async def run_test():
            db_path = str(tmp_path / "test.db")
            publisher = NewsPublisher(mode="database", db_path=db_path)

            await publisher.initialize()

            story = FinalizedStory(
                story_id="pub_test",
                title="Test Story",
                what_happened="Testing happened.",
                why_it_matters="Testing matters.",
                bullet_points=["Point 1"],
                affected_assets=["TEST"],
                category="test",
                verification_confidence=8,
            )

            result = await publisher.publish(story)

            assert result is not None
            assert result.story_id == "pub_test"

            # Verify retrieval
            stories = await publisher.get_recent_stories(limit=10)
            assert len(stories) == 1
            assert stories[0]["title"] == "Test Story"

            publisher.close()

        asyncio.run(run_test())

    def test_publisher_file_mode(self, tmp_path):
        """Publisher should work in file mode."""
        from news_pipeline.publisher import NewsPublisher

        async def run_test():
            output_dir = str(tmp_path / "output")
            publisher = NewsPublisher(mode="file", output_dir=output_dir)

            await publisher.initialize()

            story = FinalizedStory(
                story_id="file_test",
                title="File Test Story",
                what_happened="File testing happened.",
                why_it_matters="File testing matters.",
                bullet_points=["Point 1"],
                affected_assets=["FILE"],
                category="test",
                verification_confidence=7,
            )

            result = await publisher.publish(story)

            assert result is not None
            assert "file://" in result.feed_url

            # Verify file was created
            import os

            files = os.listdir(output_dir)
            assert len(files) == 1
            assert "file_test" in files[0]

        asyncio.run(run_test())


class TestEnums:
    """Tests for enum types."""

    def test_pipeline_stage_values(self):
        """PipelineStage should have correct values."""
        assert PipelineStage.DISCOVERY.value == "discovery"
        assert PipelineStage.VERIFICATION.value == "verification"
        assert PipelineStage.FORMATTING.value == "formatting"
        assert PipelineStage.EDITORIAL.value == "editorial"
        assert PipelineStage.PUBLISHING.value == "publishing"

    def test_story_category_values(self):
        """StoryCategory should include key financial categories."""
        categories = [c.value for c in StoryCategory]

        assert "fed" in categories
        assert "earnings" in categories
        assert "oil" in categories
        assert "geopolitics" in categories

    def test_verification_status_values(self):
        """VerificationStatus should have correct values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.PARTIAL.value == "partial"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"


class TestScrapers:
    """Tests for news scrapers."""

    def test_news_category_enum(self):
        """NewsCategory should have correct values."""
        from news_pipeline.scrapers.base import NewsCategory

        assert NewsCategory.FED.value == "fed"
        assert NewsCategory.EARNINGS.value == "earnings"
        assert NewsCategory.BREAKING.value == "breaking"

    def test_source_type_enum(self):
        """SourceType should have correct values."""
        from news_pipeline.scrapers.base import SourceType

        assert SourceType.OFFICIAL.value == "official"
        assert SourceType.MAINSTREAM.value == "mainstream"
        assert SourceType.RSS.value == "rss"

    def test_news_item_dataclass(self):
        """NewsItem should store news data."""
        from news_pipeline.scrapers.base import NewsCategory, NewsItem, SourceType

        item = NewsItem(
            headline="Fed raises rates",
            source_name="Reuters",
            source_url="https://reuters.com/fed",
            source_type=SourceType.MAINSTREAM,
            snippet="The Federal Reserve announced...",
            category=NewsCategory.FED,
            tickers=["SPY", "TLT"],
        )

        assert item.headline == "Fed raises rates"
        assert item.category == NewsCategory.FED
        assert len(item.tickers) == 2


class TestLocalLLM:
    """Tests for local LLM preprocessing."""

    def test_preprocessor_initialization(self):
        """LocalPreprocessor should initialize correctly."""
        from news_pipeline.local_llm import LocalPreprocessor

        preprocessor = LocalPreprocessor(
            model="qwen2.5:7b",
            use_llm=False,  # Don't require actual Ollama
        )

        assert preprocessor.model == "qwen2.5:7b"
        assert preprocessor.use_llm is False

    def test_preprocessor_rule_based_filter(self):
        """LocalPreprocessor should filter with rules when LLM disabled."""
        from news_pipeline.local_llm import LocalPreprocessor
        from news_pipeline.scrapers.base import NewsCategory, NewsItem, SourceType

        preprocessor = LocalPreprocessor(use_llm=False)

        items = [
            NewsItem(
                headline="Fed raises rates by 25bps",
                source_name="Reuters",
                source_url="https://reuters.com",
                source_type=SourceType.MAINSTREAM,
                category=NewsCategory.FED,
            ),
            NewsItem(
                headline="Celebrity gossip update",
                source_name="Tabloid",
                source_url="https://tabloid.com",
                source_type=SourceType.UNKNOWN,
                category=NewsCategory.OTHER,
            ),
        ]

        async def run_test():
            filtered = await preprocessor.preprocess_batch(items, filter_threshold=0.3)
            # Fed news should pass, gossip should be filtered
            assert len(filtered) >= 1
            assert any("Fed" in item.headline for item in filtered)

        asyncio.run(run_test())
