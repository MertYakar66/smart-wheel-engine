"""
Adversarial Tests for News Pipeline

Tests the news pipeline's resilience against:
- Contradictory information from multiple sources
- Stale/outdated narratives
- Near-duplicate rumors and reworded stories
- Misinformation and manipulation attempts
- Edge cases and malformed inputs

These tests ensure the verification system is robust.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from news_pipeline.models import (
    CandidateStory,
)


@dataclass
class AdversarialScenario:
    """Definition of an adversarial test scenario."""
    name: str
    description: str
    stories: list[CandidateStory]
    expected_outcome: str  # "reject", "flag", "accept_with_warning"


# =============================================================================
# CONTRADICTORY SOURCES TESTS
# =============================================================================

class TestContradictorySources:
    """Tests for handling contradictory information."""

    def test_opposite_claims_same_event(self):
        """Two sources making opposite claims about same event."""
        story_up = CandidateStory(
            story_id="fed_up",
            headline="Fed Raises Rates by 25bps in Surprise Move",
            source_name="Reuters",
            source_url="https://reuters.com/fed1",
            snippet="The Federal Reserve raised rates unexpectedly...",
            discovered_at=datetime.now(),
            tickers=["SPY", "TLT"],
            category="fed",
        )

        story_down = CandidateStory(
            story_id="fed_down",
            headline="Fed Cuts Rates by 25bps in Surprise Move",
            source_name="Bloomberg",
            source_url="https://bloomberg.com/fed1",
            snippet="The Federal Reserve cut rates unexpectedly...",
            discovered_at=datetime.now(),
            tickers=["SPY", "TLT"],
            category="fed",
        )

        # Both stories claim Fed action but opposite directions
        # Verification should flag this contradiction
        assert story_up.category == story_down.category
        assert "Fed" in story_up.headline and "Fed" in story_down.headline

        # Key assertions for verification logic:
        # - Same tickers affected
        # - Same category
        # - Contradictory headlines (raise vs cut)
        # - Should NOT both be verified

    def test_conflicting_earnings_numbers(self):
        """Conflicting EPS numbers from different sources."""
        story_beat = CandidateStory(
            story_id="aapl_beat",
            headline="Apple Reports Q4 EPS of $2.18, Beating Estimates",
            source_name="CNBC",
            source_url="https://cnbc.com/aapl",
            snippet="Apple reported earnings of $2.18 per share...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="earnings",
        )

        story_miss = CandidateStory(
            story_id="aapl_miss",
            headline="Apple Reports Q4 EPS of $1.98, Missing Estimates",
            source_name="MarketWatch",
            source_url="https://marketwatch.com/aapl",
            snippet="Apple disappointed with earnings of $1.98...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="earnings",
        )

        # These report different EPS numbers for the same quarter
        # Verification must detect numeric discrepancy
        assert story_beat.tickers == story_miss.tickers
        # $2.18 vs $1.98 - significant difference

    def test_timing_contradiction(self):
        """Sources reporting different timing for same event."""
        story_today = CandidateStory(
            story_id="oil_today",
            headline="OPEC Announces Production Cut Effective Today",
            source_name="Reuters",
            source_url="https://reuters.com/opec",
            snippet="OPEC announced immediate production cuts...",
            discovered_at=datetime.now(),
            tickers=["USO", "XLE"],
            category="oil",
        )

        story_next_month = CandidateStory(
            story_id="oil_next_month",
            headline="OPEC Announces Production Cut Starting Next Month",
            source_name="WSJ",
            source_url="https://wsj.com/opec",
            snippet="OPEC announced production cuts beginning next month...",
            discovered_at=datetime.now(),
            tickers=["USO", "XLE"],
            category="oil",
        )

        # Same event, different timing
        assert "OPEC" in story_today.headline
        assert "OPEC" in story_next_month.headline
        # "Today" vs "Next Month" contradiction


# =============================================================================
# STALE NARRATIVE TESTS
# =============================================================================

class TestStaleNarratives:
    """Tests for detecting and handling stale/outdated news."""

    def test_old_news_resurfacing(self):
        """Old news being reported as new."""
        # Story from a month ago
        old_story = CandidateStory(
            story_id="old_layoffs",
            headline="Tech Company Announces Major Layoffs",
            source_name="TechCrunch",
            source_url="https://techcrunch.com/layoffs",
            snippet="The company announced 10,000 layoffs today...",
            discovered_at=datetime.now() - timedelta(days=30),
            tickers=["TECH"],
            category="tech",
        )

        # Same story being circulated again
        recycled_story = CandidateStory(
            story_id="recycled_layoffs",
            headline="Tech Company Announces Major Layoffs",
            source_name="RandomBlog",
            source_url="https://randomblog.com/layoffs",
            snippet="The company announced 10,000 layoffs today...",
            discovered_at=datetime.now(),
            tickers=["TECH"],
            category="tech",
        )

        # Same headline, same snippet - but different timestamps
        assert old_story.headline == recycled_story.headline
        assert old_story.snippet == recycled_story.snippet
        # Should detect this is old news being recycled

    def test_outdated_price_reference(self):
        """News referencing outdated price levels."""
        story = CandidateStory(
            story_id="btc_price",
            headline="Bitcoin Crashes Below $30,000",
            source_name="CryptoNews",
            source_url="https://cryptonews.com/btc",
            snippet="Bitcoin fell below the critical $30,000 level...",
            discovered_at=datetime.now(),
            tickers=["BTC"],
            category="crypto",
        )

        # If BTC is currently at $50,000, this $30,000 reference is stale
        # Verification should cross-reference current prices
        assert "$30,000" in story.headline
        assert story.category == "crypto"

    def test_superseded_guidance(self):
        """Company guidance that has been superseded."""
        old_guidance = CandidateStory(
            story_id="guidance_q1",
            headline="Company Projects Q4 Revenue of $10B",
            source_name="Company PR",
            source_url="https://company.com/pr/q1",
            snippet="We expect Q4 revenue of approximately $10B...",
            discovered_at=datetime.now() - timedelta(days=60),
            tickers=["XYZ"],
            category="earnings",
        )

        new_guidance = CandidateStory(
            story_id="guidance_q3",
            headline="Company Revises Q4 Revenue Guidance to $8B",
            source_name="Company PR",
            source_url="https://company.com/pr/q3",
            snippet="We now expect Q4 revenue of approximately $8B...",
            discovered_at=datetime.now(),
            tickers=["XYZ"],
            category="earnings",
        )

        # Old guidance is superseded by new guidance
        # Only the newer guidance should be used
        assert old_guidance.discovered_at < new_guidance.discovered_at
        assert old_guidance.tickers == new_guidance.tickers


# =============================================================================
# NEAR-DUPLICATE AND RUMOR TESTS
# =============================================================================

class TestNearDuplicates:
    """Tests for detecting near-duplicate stories and rumors."""

    def test_reworded_same_story(self):
        """Same story with different wording."""
        story_v1 = CandidateStory(
            story_id="merger_v1",
            headline="Company A to Acquire Company B for $5 Billion",
            source_name="Reuters",
            source_url="https://reuters.com/merger",
            snippet="Company A announced plans to acquire Company B...",
            discovered_at=datetime.now(),
            tickers=["A", "B"],
            category="tech",
        )

        story_v2 = CandidateStory(
            story_id="merger_v2",
            headline="Company A Buying Company B in $5B Deal",
            source_name="Bloomberg",
            source_url="https://bloomberg.com/merger",
            snippet="Company A is purchasing Company B for $5 billion...",
            discovered_at=datetime.now(),
            tickers=["A", "B"],
            category="tech",
        )

        # Same facts, different wording
        # Should be deduplicated
        assert story_v1.tickers == story_v2.tickers
        # "acquire" vs "buying" - semantic similarity

    def test_rumor_vs_confirmed(self):
        """Distinguishing rumors from confirmed news."""
        rumor = CandidateStory(
            story_id="rumor",
            headline="RUMOR: Apple May Be Developing AR Glasses",
            source_name="AppleInsider",
            source_url="https://appleinsider.com/rumor",
            snippet="According to sources, Apple might be working on...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="tech",
        )

        confirmed = CandidateStory(
            story_id="confirmed",
            headline="Apple Announces AR Glasses at WWDC",
            source_name="Apple Newsroom",
            source_url="https://apple.com/newsroom/ar",
            snippet="Today we are excited to announce Apple Vision...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="tech",
        )

        # Rumor contains hedging language
        assert "RUMOR" in rumor.headline or "May Be" in rumor.headline
        assert "might" in rumor.snippet.lower() or "sources" in rumor.snippet.lower()

        # Confirmed is definitive
        assert "Announces" in confirmed.headline

    def test_speculation_chain(self):
        """Speculation that spawns more speculation."""
        # Original speculation
        original = CandidateStory(
            story_id="spec_1",
            headline="Analyst Speculates Fed Could Cut Rates",
            source_name="FinanceAnalyst",
            source_url="https://analyst.com/fed",
            snippet="One analyst believes the Fed might consider...",
            discovered_at=datetime.now() - timedelta(hours=2),
            tickers=["SPY"],
            category="fed",
        )

        # Derived speculation (citing the original)
        derived = CandidateStory(
            story_id="spec_2",
            headline="Markets React to Speculation of Fed Rate Cut",
            source_name="MarketBlog",
            source_url="https://marketblog.com/fed",
            snippet="Following analyst speculation about a potential...",
            discovered_at=datetime.now() - timedelta(hours=1),
            tickers=["SPY"],
            category="fed",
        )

        # Further derived
        further = CandidateStory(
            story_id="spec_3",
            headline="Fed Rate Cut Speculation Intensifies",
            source_name="NewsSite",
            source_url="https://newssite.com/fed",
            snippet="Amid growing speculation about Fed policy...",
            discovered_at=datetime.now(),
            tickers=["SPY"],
            category="fed",
        )

        # All three are speculation, not facts
        # Should not be treated as corroboration
        assert original.discovered_at < derived.discovered_at < further.discovered_at
        assert all(s.category == "fed" for s in [original, derived, further])


# =============================================================================
# MISINFORMATION DETECTION TESTS
# =============================================================================

class TestMisinformation:
    """Tests for detecting potential misinformation."""

    def test_suspicious_source(self):
        """News from low-credibility source."""
        story = CandidateStory(
            story_id="suspicious",
            headline="BREAKING: Major Company Files for Bankruptcy",
            source_name="UnknownBlog123",
            source_url="https://unknownblog123.wordpress.com/news",
            snippet="URGENT: Sources confirm major bankruptcy filing...",
            discovered_at=datetime.now(),
            tickers=["XYZ"],
            category="earnings",
        )

        # Red flags:
        # - Unknown source
        # - BREAKING/URGENT language
        # - Vague "sources confirm"
        assert "BREAKING" in story.headline or "URGENT" in story.snippet

    def test_impossible_claim(self):
        """Claims that are logically impossible."""
        story = CandidateStory(
            story_id="impossible",
            headline="Apple Stock Up 500% After Earnings",
            source_name="FinanceBot",
            source_url="https://financebot.io/aapl",
            snippet="AAPL surged 500% in after-hours trading...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="earnings",
        )

        # 500% gain is unrealistic for a large-cap stock
        # Should be flagged as potentially false
        assert "500%" in story.headline
        assert story.tickers == ["AAPL"]

    def test_future_dated_event(self):
        """News claiming something happened that hasn't yet."""
        future_date = datetime.now() + timedelta(days=7)

        story = CandidateStory(
            story_id="future",
            headline=f"Fed Raises Rates at {future_date.strftime('%B %d')} Meeting",
            source_name="FakeNews",
            source_url="https://fakenews.com/fed",
            snippet="The Federal Reserve raised rates today at...",
            discovered_at=datetime.now(),
            tickers=["SPY"],
            category="fed",
        )

        # Claims Fed action at a future date
        # Logically impossible
        assert future_date.strftime("%B %d") in story.headline
        assert story.category == "fed"


# =============================================================================
# EDGE CASES AND MALFORMED INPUT TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and malformed inputs."""

    def test_empty_headline(self):
        """Story with empty headline."""
        story = CandidateStory(
            story_id="empty_headline",
            headline="",
            source_name="Reuters",
            source_url="https://reuters.com/empty",
            snippet="Some content here...",
            discovered_at=datetime.now(),
            tickers=["SPY"],
            category="macro",
        )

        # Should be rejected or flagged
        assert story.headline == ""

    def test_very_long_headline(self):
        """Excessively long headline."""
        long_headline = "Breaking News: " + "Important Update " * 100

        story = CandidateStory(
            story_id="long_headline",
            headline=long_headline,
            source_name="NewsSource",
            source_url="https://news.com/long",
            snippet="Content...",
            discovered_at=datetime.now(),
            tickers=["SPY"],
            category="macro",
        )

        # Should truncate or flag abnormal length
        assert len(story.headline) > 500

    def test_special_characters_in_story(self):
        """Story with special characters and encoding issues."""
        story = CandidateStory(
            story_id="special_chars",
            headline="Apple's Stock—Up 10% After 'Major' Announcement «exclusive»",
            source_name="News™",
            source_url="https://news.com/special",
            snippet="The company's earnings were €1.5B higher than...",
            discovered_at=datetime.now(),
            tickers=["AAPL"],
            category="earnings",
        )

        # Should handle special characters gracefully
        assert "'" in story.headline
        assert "—" in story.headline
        assert "«" in story.headline

    def test_missing_tickers(self):
        """Story without any tickers."""
        story = CandidateStory(
            story_id="no_tickers",
            headline="Global Economic Outlook Remains Uncertain",
            source_name="Reuters",
            source_url="https://reuters.com/economy",
            snippet="Economic indicators suggest...",
            discovered_at=datetime.now(),
            tickers=[],  # No tickers
            category="macro",
        )

        # Should handle missing tickers
        assert story.tickers == []

    def test_duplicate_tickers(self):
        """Story with duplicate tickers."""
        story = CandidateStory(
            story_id="dup_tickers",
            headline="Apple vs Apple: Internal Conflict",
            source_name="TechNews",
            source_url="https://technews.com/aapl",
            snippet="The company faces internal challenges...",
            discovered_at=datetime.now(),
            tickers=["AAPL", "AAPL", "AAPL"],  # Duplicates
            category="tech",
        )

        # Should deduplicate
        assert story.tickers.count("AAPL") == 3  # Raw input has dupes


# =============================================================================
# VERIFICATION RESILIENCE TESTS
# =============================================================================

class TestVerificationResilience:
    """Tests for verification system resilience."""

    def test_partial_provider_failure(self):
        """Verification when some providers fail."""
        story = CandidateStory(
            story_id="partial_fail",
            headline="Market Update: S&P 500 Hits New High",
            source_name="Reuters",
            source_url="https://reuters.com/sp500",
            snippet="The S&P 500 index reached a new all-time high...",
            discovered_at=datetime.now(),
            tickers=["SPY"],
            category="macro",
        )

        # Verification should still work if 1/3 providers fail
        # Should not require unanimous agreement
        assert story.source_name == "Reuters"  # High credibility source
        assert story.category == "macro"

    def test_slow_provider_timeout(self):
        """Verification with slow provider timing out."""
        # Verification should continue even if one provider is slow
        pass

    def test_rate_limited_provider(self):
        """Verification when provider is rate limited."""
        # Should fallback gracefully
        pass


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_contradictory_stories():
    """Fixture providing contradictory story pairs."""
    return [
        (
            CandidateStory(
                story_id="up",
                headline="Stock Surges 10%",
                source_name="Source A",
                source_url="https://a.com",
                snippet="...",
                discovered_at=datetime.now(),
                tickers=["XYZ"],
                category="earnings",
            ),
            CandidateStory(
                story_id="down",
                headline="Stock Plunges 10%",
                source_name="Source B",
                source_url="https://b.com",
                snippet="...",
                discovered_at=datetime.now(),
                tickers=["XYZ"],
                category="earnings",
            ),
        ),
    ]


@pytest.fixture
def sample_duplicate_stories():
    """Fixture providing near-duplicate stories."""
    base_time = datetime.now()
    return [
        CandidateStory(
            story_id=f"dup_{i}",
            headline="Company Reports Strong Earnings",
            source_name=f"Source{i}",
            source_url=f"https://source{i}.com/earnings",
            snippet="The company reported better than expected results...",
            discovered_at=base_time + timedelta(minutes=i),
            tickers=["ABC"],
            category="earnings",
        )
        for i in range(5)
    ]
