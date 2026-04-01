"""
Tests for the multi-model news pipeline.

Tests cover:
- Data models and schema
- Provider interfaces
- Orchestrator logic
- Publisher functionality
- Integration layer
"""

import asyncio
import importlib.util
import json
from datetime import datetime

import pytest

from news_pipeline.config import PipelineConfig, ProviderConfig
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


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = PipelineConfig()

        assert config.min_verification_confidence == 6
        assert config.max_concurrent_verifications == 5
        assert config.max_stories_per_run == 50

    def test_config_validation_missing_keys(self):
        """Config validation should catch missing API keys."""
        config = PipelineConfig()
        # Clear any env-loaded keys
        config.grok.api_key = None
        config.gemini.api_key = None
        config.chatgpt.api_key = None
        config.claude.api_key = None

        errors = config.validate()

        assert len(errors) == 4
        assert any("XAI_API_KEY" in e for e in errors)
        assert any("GOOGLE_API_KEY" in e for e in errors)

    def test_config_enabled_providers(self):
        """Config should list enabled providers."""
        config = PipelineConfig()

        providers = config.get_enabled_providers()

        assert "grok" in providers
        assert "gemini" in providers
        assert "chatgpt" in providers
        assert "claude" in providers


class TestProviderInterfaces:
    """Tests for provider implementations."""

    def test_grok_provider_initialization(self):
        """GrokProvider should initialize correctly."""
        from news_pipeline.providers import GrokProvider

        config = ProviderConfig(api_key="test_key", model="grok-beta")
        provider = GrokProvider(config)

        assert provider.name == "GrokProvider"
        assert provider.is_initialized is False

    def test_grok_provider_parse_response(self):
        """GrokProvider should parse discovery responses."""
        from news_pipeline.providers.grok import GrokProvider

        config = ProviderConfig(api_key="test", model="grok-beta")
        provider = GrokProvider(config)

        # Test JSON parsing
        response = """```json
        [
            {
                "headline": "Fed cuts rates",
                "source_name": "Reuters",
                "source_url": "https://reuters.com/fed",
                "tickers": ["SPY"],
                "category": "fed"
            }
        ]
        ```"""

        stories = provider._parse_discovery_response(response)

        assert len(stories) == 1
        assert stories[0]["headline"] == "Fed cuts rates"
        assert stories[0]["tickers"] == ["SPY"]

    def test_gemini_provider_parse_verification(self):
        """GeminiProvider should parse verification responses."""
        from news_pipeline.providers.gemini import GeminiProvider

        config = ProviderConfig(api_key="test", model="gemini-pro")
        provider = GeminiProvider(config)

        response = """```json
        {
            "status": "verified",
            "confidence": 8,
            "verified_facts": ["Fed cut rates by 25bps"],
            "what_happened": "The Fed announced a rate cut.",
            "contradictions": []
        }
        ```"""

        result = provider._parse_verification_response(response)

        assert result["status"] == "verified"
        assert result["confidence"] == 8
        assert len(result["verified_facts"]) == 1

    def test_chatgpt_provider_fallback(self):
        """ChatGPTProvider should have fallback formatting."""
        from news_pipeline.providers.chatgpt import ChatGPTProvider

        config = ProviderConfig(api_key="test", model="gpt-4")
        provider = ChatGPTProvider(config)

        result = provider._fallback_format(
            verified_facts=["Fact 1", "Fact 2"],
            what_happened="Something happened.",
            affected_assets=["AAPL"],
            category="earnings",
        )

        assert result["what_happened"] == "Something happened."
        assert result["bullet_points"] == ["Fact 1", "Fact 2"]
        assert result["affected_assets"] == ["AAPL"]

    def test_claude_provider_fallback(self):
        """ClaudeProvider should have fallback finalization."""
        from news_pipeline.providers.claude import ClaudeProvider

        config = ProviderConfig(api_key="test", model="claude-3-sonnet")
        provider = ClaudeProvider(config)

        result = provider._fallback_finalize(
            title="Fed Rate Decision",
            what_happened="Fed cut rates.",
            bullet_points=["Point 1"],
            affected_assets=["SPY"],
            category="fed",
        )

        assert "why_it_matters" in result
        assert "Federal Reserve" in result["why_it_matters"]
        assert result["priority"] == 5


# Check if playwright is available for browser-based tests
HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
class TestOrchestrator:
    """Tests for pipeline orchestrator (requires playwright)."""

    def test_orchestrator_initialization(self):
        """Orchestrator should initialize with config."""
        from news_pipeline.orchestrator import NewsPipelineOrchestrator, OrchestratorConfig

        config = OrchestratorConfig()
        orchestrator = NewsPipelineOrchestrator(config=config)

        assert orchestrator.config == config

    def test_orchestrator_health_check(self):
        """Orchestrator should have health monitoring."""
        from news_pipeline.orchestrator import NewsPipelineOrchestrator

        orchestrator = NewsPipelineOrchestrator()

        # The browser-based orchestrator uses health_monitor
        assert hasattr(orchestrator, "health_monitor")


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


class TestIntegration:
    """Tests for financial_news integration."""

    def test_integration_category_mapping(self):
        """Integration should map categories correctly."""
        from news_pipeline.integration import FinancialNewsIntegration

        integration = FinancialNewsIntegration()

        assert integration._map_category("federal_reserve") == "fed"
        assert integration._map_category("corporate_earnings") == "earnings"
        assert integration._map_category("crude_oil") == "oil"
        assert integration._map_category("unknown") == "unknown"

    def test_integration_source_type(self):
        """Integration should identify source types."""
        from news_pipeline.integration import FinancialNewsIntegration

        integration = FinancialNewsIntegration()

        assert integration._get_source_type("SEC") == "official"
        assert integration._get_source_type("GDELT") == "mainstream"
        assert integration._get_source_type("Random") == "unknown"
