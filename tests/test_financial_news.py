"""
Comprehensive Test Suite for Financial News Intelligence System

Tests all financial news components:
- Schema validation
- Verification engine
- Calendar events
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import financial news components
from financial_news import (
    # Enums
    SourceType,
    SourceProvider,
    CategoryType,
    EventType,
    EntityType,
    ImportanceLevel,
    BriefType,
    RunStatus,
    # Models
    Source,
    Category,
    ScheduledEvent,
    Entity,
    Article,
    Story,
    # Defaults
    DEFAULT_SOURCES,
    DEFAULT_CATEGORIES,
    # Components
    MacroCalendar,
    # Verification
    VerificationEngine,
    VerificationCandidate,
    VerificationResult,
    VerificationStatus,
    PushPayload,
    run_verification_cycle,
    process_verification_response,
    push_verified_stories,
)


# =============================================================================
# SCHEMA ENUM TESTS
# =============================================================================

class TestSchemaEnums:
    """Test enum definitions."""

    def test_source_type_values(self):
        """Test SourceType enum values."""
        assert SourceType.OFFICIAL.value == "official"
        assert SourceType.LICENSED.value == "licensed"
        assert SourceType.AGGREGATOR.value == "aggregator"

    def test_source_provider_values(self):
        """Test SourceProvider enum has expected providers."""
        providers = [p.value for p in SourceProvider]
        assert "federal_reserve" in providers
        assert "sec_edgar" in providers
        assert "eia" in providers

    def test_category_type_values(self):
        """Test CategoryType enum covers 8 categories."""
        categories = list(CategoryType)
        assert len(categories) == 8

        expected = ["fed_rates", "inflation", "labor", "growth_consumer",
                   "oil_energy", "geopolitics", "sp500_corporate", "market_regime"]
        for cat in expected:
            assert cat in [c.value for c in categories]

    def test_event_type_has_macro_events(self):
        """Test EventType enum has macro events."""
        events = [e.value for e in EventType]
        # Check for key event types
        assert "fomc_decision" in events
        assert "cpi" in events
        assert "nfp" in events

    def test_importance_level_values(self):
        """Test ImportanceLevel enum."""
        levels = [l.value for l in ImportanceLevel]
        assert "critical" in levels
        assert "high" in levels
        assert "medium" in levels
        assert "low" in levels

    def test_verification_status_values(self):
        """Test VerificationStatus enum."""
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.REJECTED.value == "rejected"


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestSource:
    """Test Source dataclass."""

    def test_source_creation(self):
        """Test Source creation."""
        source = Source(
            source_id="test_source",
            name="Test Source",
            provider=SourceProvider.FEDERAL_RESERVE,
            source_type=SourceType.OFFICIAL,
            base_url="https://example.com",
            rate_limit_per_second=1.0
        )

        assert source.source_id == "test_source"
        assert source.provider == SourceProvider.FEDERAL_RESERVE
        assert source.source_type == SourceType.OFFICIAL

    def test_default_sources_exist(self):
        """Test default sources are defined."""
        assert len(DEFAULT_SOURCES) >= 5
        assert all(isinstance(s, Source) for s in DEFAULT_SOURCES)


class TestCategory:
    """Test Category dataclass."""

    def test_category_creation(self):
        """Test Category creation with correct fields."""
        category = Category(
            category_id="test_cat",
            category_type=CategoryType.FED_RATES,
            name="Test Category",
            description="Test description"
        )

        assert category.category_id == "test_cat"
        assert category.category_type == CategoryType.FED_RATES

    def test_default_categories_exist(self):
        """Test default categories are defined."""
        assert len(DEFAULT_CATEGORIES) == 8
        assert all(isinstance(c, Category) for c in DEFAULT_CATEGORIES)


class TestScheduledEvent:
    """Test ScheduledEvent dataclass."""

    def test_event_creation(self):
        """Test ScheduledEvent creation."""
        event = ScheduledEvent(
            event_id="fomc_jan_2026",
            source_id="federal_reserve",
            event_type=EventType.FOMC_DECISION,
            category_id="fed_rates",
            scheduled_at=datetime(2026, 1, 29, 14, 0),
            importance=ImportanceLevel.CRITICAL,
            title="FOMC Meeting",
            description="Federal Reserve policy decision"
        )

        assert event.event_type == EventType.FOMC_DECISION
        assert event.importance == ImportanceLevel.CRITICAL


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test Entity creation."""
        entity = Entity(
            entity_id="ent_001",
            entity_type=EntityType.TICKER,
            value="AAPL",
            confidence=0.95
        )

        assert entity.entity_type == EntityType.TICKER
        assert entity.value == "AAPL"
        assert entity.confidence == 0.95


class TestArticle:
    """Test Article dataclass."""

    def test_article_creation(self):
        """Test Article creation."""
        article = Article(
            article_id="art_001",
            source_id="fed",
            title="Fed Holds Rates Steady",
            canonical_url="https://fed.gov/release",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow()
        )

        assert article.article_id == "art_001"
        assert "Fed" in article.title


class TestStory:
    """Test Story dataclass."""

    def test_story_creation(self):
        """Test Story creation."""
        now = datetime.utcnow()
        story = Story(
            story_id="story_001",
            lead_article_id="art_001",
            headline="Fed Policy Update",
            summary="The Federal Reserve announced policy changes.",
            why_it_matters="This affects interest rates and market sentiment.",
            first_seen_at=now,
            last_updated_at=now
        )

        assert story.story_id == "story_001"
        assert story.headline == "Fed Policy Update"
        assert "Federal Reserve" in story.summary
        assert "interest rates" in story.why_it_matters


# =============================================================================
# MACRO CALENDAR TESTS
# =============================================================================

class TestMacroCalendar:
    """Test MacroCalendar functionality."""

    @pytest.fixture
    def calendar(self):
        return MacroCalendar()

    def test_calendar_initialization(self, calendar):
        """Test calendar initializes with events."""
        assert calendar is not None

    def test_calendar_has_events(self, calendar):
        """Test calendar has scheduled events."""
        events = calendar.get_all_events()
        assert len(events) > 0

    def test_calendar_events_have_dates(self, calendar):
        """Test calendar events have proper dates."""
        events = calendar.get_all_events()
        for event in events[:5]:  # Check first 5
            assert hasattr(event, 'scheduled_at')

    def test_calendar_has_fomc_events(self, calendar):
        """Test calendar has FOMC events."""
        events = calendar.get_all_events()
        fomc_events = [e for e in events
                       if e.event_type == EventType.FOMC_DECISION]
        assert len(fomc_events) >= 8  # At least 8 FOMC meetings per year

    def test_calendar_has_cpi_events(self, calendar):
        """Test calendar has CPI events."""
        events = calendar.get_all_events()
        cpi_events = [e for e in events
                      if e.event_type == EventType.CPI]
        assert len(cpi_events) >= 12  # Monthly CPI


# =============================================================================
# VERIFICATION ENGINE TESTS
# =============================================================================

class TestVerificationEngine:
    """Test VerificationEngine functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_news.db")

    @pytest.fixture
    def engine(self, temp_db_path):
        """Create verification engine with temp database."""
        return VerificationEngine(db_path=temp_db_path)

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None

    def test_add_candidate(self, engine):
        """Test adding verification candidate."""
        candidate_id = engine.add_candidate(
            headline="NVIDIA announces new AI chip",
            source="CNBC",
            source_type="premium",
            url="https://cnbc.com/article",
            tickers=["NVDA"],
            categories=["SP500_EVENTS"]
        )

        assert candidate_id is not None
        assert len(candidate_id) > 0

    def test_add_duplicate_candidate(self, engine):
        """Test adding duplicate candidate returns same ID."""
        id1 = engine.add_candidate(
            headline="Test headline",
            source="Test",
            source_type="public",
        )

        id2 = engine.add_candidate(
            headline="Test headline",
            source="Test",
            source_type="public",
        )

        assert id1 == id2

    def test_get_pending_candidates(self, engine):
        """Test retrieving pending candidates."""
        # Add some candidates
        engine.add_candidate("Headline 1", "Source1", "public", tickers=["AAPL"])
        engine.add_candidate("Headline 2", "Source2", "premium", tickers=["MSFT"])

        pending = engine.get_pending_candidates(limit=10)

        assert len(pending) == 2
        assert all(isinstance(c, VerificationCandidate) for c in pending)

    def test_candidate_to_verification_query(self, engine):
        """Test candidate generates verification query."""
        engine.add_candidate(
            headline="Fed signals rate cut",
            source="Reuters",
            source_type="public",
            tickers=["SPY", "TLT"],
            categories=["FED_RATES"]
        )

        candidates = engine.get_pending_candidates()
        assert len(candidates) > 0

        query = candidates[0].to_verification_query()

        assert "Fed signals rate cut" in query
        assert "SPY" in query or "TLT" in query

    def test_store_verification_result(self, engine):
        """Test storing verification result."""
        candidate_id = engine.add_candidate(
            headline="Test headline",
            source="Test",
            source_type="public"
        )

        result = VerificationResult(
            candidate_id=candidate_id,
            verification_confidence=8,
            status=VerificationStatus.VERIFIED,
            what_happened="Test event occurred",
            why_it_matters="Market impact expected",
            affected_assets=["AAPL", "MSFT"],
            corroborating_sources=[{"name": "Source", "url": "http://example.com", "type": "public"}],
            conflicts_found=[]
        )

        engine.store_result(result)

        # Verify result was stored
        stats = engine.get_verification_stats()
        assert stats["total_verified"] >= 1

    def test_verification_stats(self, engine):
        """Test verification statistics."""
        stats = engine.get_verification_stats()

        assert "candidates_by_status" in stats
        assert "total_verified" in stats
        assert "avg_confidence" in stats


class TestVerificationCandidate:
    """Test VerificationCandidate dataclass."""

    def test_candidate_creation(self):
        """Test candidate creation."""
        candidate = VerificationCandidate(
            candidate_id="test_123",
            headline="Test headline",
            source="Test Source",
            source_type="public",
            url="https://example.com",
            published_at=datetime.utcnow(),
            tickers=["AAPL"],
            categories=["SP500_EVENTS"]
        )

        assert candidate.candidate_id == "test_123"
        assert candidate.headline == "Test headline"

    def test_to_verification_query(self):
        """Test verification query generation."""
        candidate = VerificationCandidate(
            candidate_id="test_123",
            headline="NVIDIA stock surges on AI news",
            source="Bloomberg",
            source_type="premium",
            url="https://bloomberg.com/article",
            published_at=datetime.utcnow(),
            tickers=["NVDA"],
            categories=["SP500_EVENTS"]
        )

        query = candidate.to_verification_query()

        assert "NVIDIA" in query
        assert "NVDA" in query


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = VerificationResult(
            candidate_id="test_123",
            verification_confidence=8,
            status=VerificationStatus.VERIFIED,
            what_happened="Event description",
            why_it_matters="Market significance",
            affected_assets=["AAPL", "MSFT"],
            corroborating_sources=[],
            conflicts_found=[]
        )

        assert result.verification_confidence == 8
        assert result.status == VerificationStatus.VERIFIED

    def test_should_push_high_confidence(self):
        """Test should_push for high confidence."""
        result = VerificationResult(
            candidate_id="test",
            verification_confidence=9,
            status=VerificationStatus.VERIFIED,
            what_happened="",
            why_it_matters="",
            affected_assets=[],
            corroborating_sources=[],
            conflicts_found=[]
        )

        assert result.should_push() is True

    def test_should_push_low_confidence(self):
        """Test should_push for low confidence."""
        result = VerificationResult(
            candidate_id="test",
            verification_confidence=5,
            status=VerificationStatus.NEEDS_REVIEW,
            what_happened="",
            why_it_matters="",
            affected_assets=[],
            corroborating_sources=[],
            conflicts_found=[]
        )

        assert result.should_push() is False


class TestPushPayload:
    """Test PushPayload dataclass."""

    def test_payload_creation(self):
        """Test payload creation."""
        payload = PushPayload(
            story_id="story_123",
            title="Market Update",
            what_happened="Something happened",
            why_it_matters="It matters because...",
            affected_assets=["SPY", "QQQ"],
            verification_confidence=8,
            sources=[{"name": "Source", "url": "http://example.com"}],
            categories=["MARKET_REGIME"],
            original_headline="Original headline",
            original_source="Original Source",
            original_url="http://original.com",
            verified_at=datetime.utcnow().isoformat()
        )

        assert payload.story_id == "story_123"
        assert payload.verification_confidence == 8

    def test_to_json(self):
        """Test JSON serialization."""
        payload = PushPayload(
            story_id="story_123",
            title="Test",
            what_happened="Test",
            why_it_matters="Test",
            affected_assets=[],
            verification_confidence=8,
            sources=[],
            categories=[],
            original_headline="",
            original_source="",
            original_url="",
            verified_at=""
        )

        json_str = payload.to_json()

        assert isinstance(json_str, str)
        assert "story_123" in json_str


# =============================================================================
# WORKFLOW FUNCTION TESTS
# =============================================================================

class TestWorkflowFunctions:
    """Test workflow helper functions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_news.db")

    @pytest.fixture
    def engine(self, temp_db_path):
        return VerificationEngine(db_path=temp_db_path)

    def test_run_verification_cycle(self, engine):
        """Test running verification cycle."""
        # Add candidates
        engine.add_candidate("Headline 1", "Source1", "public")
        engine.add_candidate("Headline 2", "Source2", "premium")

        queries = run_verification_cycle(engine, limit=5)

        assert len(queries) == 2
        assert all("candidate_id" in q for q in queries)
        assert all("verification_query" in q for q in queries)

    def test_process_verification_response(self, engine):
        """Test processing verification response."""
        candidate_id = engine.add_candidate(
            headline="Test headline",
            source="Test",
            source_type="public"
        )

        result = process_verification_response(
            engine=engine,
            candidate_id=candidate_id,
            confidence=8,
            what_happened="Event occurred",
            why_it_matters="Market impact",
            affected_assets=["SPY"],
            sources=[{"name": "Source", "url": "http://example.com", "type": "public"}],
            conflicts=[]
        )

        assert isinstance(result, VerificationResult)
        assert result.verification_confidence == 8
        assert result.status == VerificationStatus.VERIFIED

    def test_process_low_confidence_response(self, engine):
        """Test processing low confidence response."""
        candidate_id = engine.add_candidate(
            headline="Uncertain headline",
            source="Test",
            source_type="public"
        )

        result = process_verification_response(
            engine=engine,
            candidate_id=candidate_id,
            confidence=3,
            what_happened="Maybe something",
            why_it_matters="Unclear",
            affected_assets=[],
            sources=[],
            conflicts=["Conflicting report"]
        )

        assert result.status == VerificationStatus.REJECTED
        assert result.should_push() is False

    def test_push_verified_stories_dry_run(self, engine):
        """Test pushing verified stories in dry run mode."""
        # Add and verify candidate
        candidate_id = engine.add_candidate(
            headline="Verified story",
            source="Official",
            source_type="official"
        )

        process_verification_response(
            engine=engine,
            candidate_id=candidate_id,
            confidence=9,
            what_happened="Important event",
            why_it_matters="Big impact",
            affected_assets=["AAPL"],
            sources=[],
            conflicts=[]
        )

        results = push_verified_stories(engine, dry_run=True)

        assert isinstance(results, list)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_news.db")

    @pytest.fixture
    def engine(self, temp_db_path):
        return VerificationEngine(db_path=temp_db_path)

    def test_empty_pending_candidates(self, engine):
        """Test handling empty pending candidates."""
        pending = engine.get_pending_candidates()
        assert pending == []

    def test_candidate_with_no_tickers(self, engine):
        """Test candidate without tickers."""
        candidate_id = engine.add_candidate(
            headline="General market news",
            source="News",
            source_type="public"
        )

        assert candidate_id is not None

        candidates = engine.get_pending_candidates()
        assert len(candidates) == 1
        assert candidates[0].tickers == []

    def test_candidate_with_special_characters(self, engine):
        """Test candidate with special characters in headline."""
        candidate_id = engine.add_candidate(
            headline="Stock up 50%! $AAPL 'amazing' results",
            source="News",
            source_type="public",
            tickers=["AAPL"]
        )

        assert candidate_id is not None

    def test_verification_with_max_confidence(self, engine):
        """Test verification with maximum confidence."""
        candidate_id = engine.add_candidate(
            headline="Test",
            source="Test",
            source_type="official"
        )

        result = process_verification_response(
            engine=engine,
            candidate_id=candidate_id,
            confidence=10,
            what_happened="Test",
            why_it_matters="Test",
            affected_assets=[],
            sources=[],
            conflicts=[]
        )

        assert result.verification_confidence == 10
        assert result.should_push() is True

    def test_verification_with_zero_confidence(self, engine):
        """Test verification with zero confidence."""
        candidate_id = engine.add_candidate(
            headline="Fake news",
            source="Unreliable",
            source_type="public"
        )

        result = process_verification_response(
            engine=engine,
            candidate_id=candidate_id,
            confidence=0,
            what_happened="",
            why_it_matters="",
            affected_assets=[],
            sources=[],
            conflicts=["Completely false"]
        )

        assert result.verification_confidence == 0
        assert result.status == VerificationStatus.REJECTED


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSystemIntegration:
    """Test system integration scenarios."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_news.db")

    def test_full_verification_workflow(self, temp_db_path):
        """Test complete verification workflow."""
        engine = VerificationEngine(db_path=temp_db_path)

        # 1. Add candidates
        candidates = [
            ("NVIDIA announces Rubin chip", "CNBC", "premium", ["NVDA"]),
            ("Fed holds rates steady", "Reuters", "public", ["SPY", "TLT"]),
            ("Oil prices surge on OPEC cut", "Bloomberg", "premium", ["XOM", "USO"]),
        ]

        candidate_ids = []
        for headline, source, source_type, tickers in candidates:
            cid = engine.add_candidate(
                headline=headline,
                source=source,
                source_type=source_type,
                tickers=tickers,
                categories=["SP500_EVENTS"]
            )
            candidate_ids.append(cid)

        assert len(candidate_ids) == 3

        # 2. Get pending
        pending = engine.get_pending_candidates()
        assert len(pending) == 3

        # 3. Verify each
        for cid, confidence in zip(candidate_ids, [9, 8, 7]):
            process_verification_response(
                engine=engine,
                candidate_id=cid,
                confidence=confidence,
                what_happened=f"Event for {cid}",
                why_it_matters="Market impact",
                affected_assets=["SPY"],
                sources=[],
                conflicts=[]
            )

        # 4. Check stats
        stats = engine.get_verification_stats()
        assert stats["total_verified"] == 3
        assert stats["avg_confidence"] == 8.0  # (9+8+7)/3

        # 5. Push verified
        results = push_verified_stories(engine, dry_run=True)
        assert isinstance(results, list)
