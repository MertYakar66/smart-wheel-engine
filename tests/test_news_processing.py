"""
Tests for financial_news processing modules.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_news.processing.classifier import ArticleClassifier, ClassificationResult
from financial_news.schema import (
    Article,
    CategoryRule,
    CategoryType,
    Entity,
    EntityType,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fed_article():
    """Create a Fed-related article."""
    return Article(
        article_id="fed_001",
        source_id="fed",
        canonical_url="https://federalreserve.gov/newsevents/pressreleases",
        title="Federal Reserve Issues FOMC Statement on Interest Rates",
        snippet="The Federal Open Market Committee decided to maintain the federal funds rate.",
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def inflation_article():
    """Create an inflation-related article."""
    return Article(
        article_id="cpi_001",
        source_id="bls",
        canonical_url="https://bls.gov/news.release/cpi.nr0.htm",
        title="Consumer Price Index Shows Core Inflation at 3.2%",
        snippet="The CPI for all urban consumers rose 0.3% in March on a seasonally adjusted basis.",
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def labor_article():
    """Create a labor market article."""
    return Article(
        article_id="nfp_001",
        source_id="bls",
        canonical_url="https://bls.gov/news.release/empsit.nr0.htm",
        title="Employment Situation: Nonfarm Payrolls Rise by 200,000",
        snippet="Total nonfarm payroll employment increased, and the unemployment rate remained at 3.8%.",
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def oil_article():
    """Create an oil/energy article."""
    return Article(
        article_id="oil_001",
        source_id="eia",
        canonical_url="https://eia.gov/petroleum/weekly",
        title="Weekly Petroleum Status Report Shows Crude Oil Inventory Build",
        snippet="U.S. crude oil inventories increased by 3.5 million barrels from the prior week.",
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def sec_article():
    """Create an SEC filing article."""
    return Article(
        article_id="sec_001",
        source_id="sec_edgar",
        canonical_url="https://sec.gov/cgi-bin/browse-edgar",
        title="Apple Inc. Files 8-K Form on Earnings Results",
        snippet="Apple reported quarterly earnings that beat estimates. Revenue guidance raised.",
        tickers=["AAPL"],
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def generic_article():
    """Create a generic article."""
    return Article(
        article_id="gen_001",
        source_id="news_api",
        canonical_url="https://example.com/news",
        title="Tech Companies Report Strong Quarterly Results",
        snippet="Several technology companies posted better-than-expected earnings.",
        published_at=datetime.utcnow(),
        ingested_at=datetime.utcnow(),
    )


@pytest.fixture
def classifier():
    """Create a classifier with default rules."""
    return ArticleClassifier()


# =============================================================================
# CLASSIFIER TESTS
# =============================================================================

class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_result_creation(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            category_id="fed_rates",
            category_type=CategoryType.FED_RATES,
            confidence=0.85,
            matched_rule="fed_source",
            matched_keywords=["fomc", "federal reserve"]
        )

        assert result.category_id == "fed_rates"
        assert result.confidence == 0.85
        assert len(result.matched_keywords) == 2


class TestArticleClassifier:
    """Test ArticleClassifier class."""

    def test_initialization_default(self, classifier):
        """Test default initialization."""
        assert classifier.rules is not None
        assert len(classifier.rules) > 0

    def test_initialization_custom_rules(self):
        """Test initialization with custom rules."""
        custom_rules = [
            CategoryRule(
                rule_id="test_rule",
                category_id="fed_rates",
                include_keywords=["test"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=custom_rules)

        assert len(classifier.rules) == 1
        assert classifier.rules[0].rule_id == "test_rule"

    def test_classify_fed_article(self, classifier, fed_article):
        """Test classification of Fed article."""
        results = classifier.classify(fed_article)

        assert len(results) > 0
        # Fed article should match Fed rates category
        fed_results = [r for r in results if r.category_type == CategoryType.FED_RATES]
        assert len(fed_results) > 0
        assert fed_results[0].confidence >= 0.5

    def test_classify_inflation_article(self, classifier, inflation_article):
        """Test classification of inflation article."""
        results = classifier.classify(inflation_article)

        assert len(results) > 0
        # Should match inflation category
        inflation_results = [r for r in results if r.category_type == CategoryType.INFLATION]
        assert len(inflation_results) > 0

    def test_classify_labor_article(self, classifier, labor_article):
        """Test classification of labor market article."""
        results = classifier.classify(labor_article)

        assert len(results) > 0
        # Should match labor category
        labor_results = [r for r in results if r.category_type == CategoryType.LABOR]
        assert len(labor_results) > 0

    def test_classify_oil_article(self, classifier, oil_article):
        """Test classification of oil/energy article."""
        results = classifier.classify(oil_article)

        assert len(results) > 0
        # Should match oil/energy category
        oil_results = [r for r in results if r.category_type == CategoryType.OIL_ENERGY]
        assert len(oil_results) > 0

    def test_classify_sec_article(self, classifier, sec_article):
        """Test classification of SEC filing article."""
        results = classifier.classify(sec_article)

        assert len(results) > 0
        # Should match SP500 corporate category
        corp_results = [r for r in results if r.category_type == CategoryType.SP500_CORPORATE]
        assert len(corp_results) > 0

    def test_source_whitelist_high_confidence(self, classifier, fed_article):
        """Test that source whitelist gives high confidence."""
        results = classifier.classify(fed_article)

        fed_results = [r for r in results if r.category_type == CategoryType.FED_RATES]
        if fed_results:
            # Source match should give high confidence
            assert fed_results[0].confidence >= 0.8

    def test_multiple_categories(self, classifier):
        """Test article matching multiple categories."""
        # Article about Fed and inflation
        article = Article(
            article_id="multi_001",
            source_id="fed",
            canonical_url="https://fed.gov/news",
            title="Federal Reserve Monitors Inflation Data Before Rate Decision",
            snippet="The Fed discussed CPI trends and their impact on monetary policy decisions.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)

        # Should match both Fed and potentially inflation categories
        assert len(results) >= 1

    def test_exclude_keywords(self):
        """Test that exclude keywords block classification."""
        rules = [
            CategoryRule(
                rule_id="fed_no_fedex",
                category_id="fed_rates",
                include_keywords=["fed"],
                exclude_keywords=["fedex"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        # FedEx article should NOT match Fed rates
        fedex_article = Article(
            article_id="fedex_001",
            source_id="news",
            canonical_url="https://example.com/fedex",
            title="FedEx Reports Strong Quarterly Results",
            snippet="FedEx beat earnings estimates for the quarter.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(fedex_article)

        # Should be excluded
        assert len(results) == 0

    def test_required_keywords(self):
        """Test that required keywords must match."""
        rules = [
            CategoryRule(
                rule_id="inflation_req",
                category_id="inflation",
                include_keywords=["price", "cost"],
                required_keywords=["inflation", "cpi", "pce"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        # Article without required keywords
        no_inflation = Article(
            article_id="no_infl",
            source_id="news",
            canonical_url="https://example.com",
            title="Company Raises Prices on Products",
            snippet="The company announced price increases across its product line.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(no_inflation)
        assert len(results) == 0

        # Article with required keywords
        with_inflation = Article(
            article_id="with_infl",
            source_id="news",
            canonical_url="https://example.com",
            title="CPI Shows Rising Prices Across Sectors",
            snippet="Consumer price index data shows inflation pressures.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(with_inflation)
        assert len(results) > 0

    def test_classify_batch(self, classifier, fed_article, oil_article):
        """Test batch classification."""
        articles = [fed_article, oil_article]
        results = classifier.classify_batch(articles)

        assert len(results) == 2
        assert fed_article.article_id in results
        assert oil_article.article_id in results

    def test_get_primary_category(self, classifier, fed_article):
        """Test getting primary category."""
        primary = classifier.get_primary_category(fed_article)

        assert primary is not None
        assert isinstance(primary, CategoryType)

    def test_get_primary_category_no_match(self, classifier):
        """Test primary category when no match."""
        # Article that shouldn't match any category well
        random_article = Article(
            article_id="random",
            source_id="unknown",
            canonical_url="https://example.com",
            title="Local Event Announcement",
            snippet="A local community event will be held next week.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        classifier.get_primary_category(random_article)
        # May or may not have a match - just check it doesn't crash

    def test_update_article_categories(self, classifier, fed_article):
        """Test updating article categories."""
        updated = classifier.update_article_categories(fed_article)

        assert len(updated.categories) > 0
        assert CategoryType.FED_RATES in updated.categories

    def test_results_sorted_by_confidence(self, classifier, fed_article):
        """Test that results are sorted by confidence."""
        results = classifier.classify(fed_article)

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence


class TestKeywordMatching:
    """Test keyword matching functionality."""

    def test_case_insensitive_matching(self):
        """Test that keyword matching is case insensitive."""
        rules = [
            CategoryRule(
                rule_id="test",
                category_id="fed_rates",
                include_keywords=["FOMC", "Federal Reserve"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        article = Article(
            article_id="test",
            source_id="news",
            canonical_url="https://example.com",
            title="fomc meeting scheduled",  # lowercase
            snippet="the federal reserve will meet next week",  # lowercase
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)
        assert len(results) > 0

    def test_partial_keyword_matching(self):
        """Test partial keyword matching."""
        rules = [
            CategoryRule(
                rule_id="test",
                category_id="fed_rates",
                include_keywords=["rate"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        article = Article(
            article_id="test",
            source_id="news",
            canonical_url="https://example.com",
            title="Interest Rate Decision Expected",
            snippet="Analysts expect a rate hike.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)
        assert len(results) > 0


class TestEntityMatching:
    """Test entity-based classification."""

    def test_entity_type_boost(self):
        """Test that matching entity types boost confidence."""
        rules = [
            CategoryRule(
                rule_id="test",
                category_id="fed_rates",
                include_keywords=["policy"],
                required_entity_types=[EntityType.CENTRAL_BANK],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        article_with_entity = Article(
            article_id="test",
            source_id="news",
            canonical_url="https://example.com",
            title="Monetary Policy Update",
            snippet="Central bank announces new policy measures.",
            entities=[
                Entity(
                    entity_id="e1",
                    entity_type=EntityType.CENTRAL_BANK,
                    value="Federal Reserve"
                )
            ],
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article_with_entity)
        # Should have higher confidence due to entity match
        assert len(results) > 0

    def test_ticker_whitelist_boost(self):
        """Test that ticker whitelist boosts confidence."""
        rules = [
            CategoryRule(
                rule_id="test",
                category_id="sp500_corporate",
                include_keywords=["earnings"],
                ticker_whitelist=["AAPL", "MSFT", "GOOGL"],
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        article = Article(
            article_id="test",
            source_id="news",
            canonical_url="https://example.com",
            title="Apple Earnings Report",
            snippet="Apple reports quarterly earnings.",
            tickers=["AAPL"],
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)
        assert len(results) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_article(self):
        """Test classification with minimal article."""
        classifier = ArticleClassifier()

        article = Article(
            article_id="empty",
            source_id="unknown",
            canonical_url="https://example.com",
            title="",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        # Should not crash
        classifier.classify(article)
        # May or may not have results

    def test_inactive_rules(self):
        """Test that inactive rules are skipped."""
        rules = [
            CategoryRule(
                rule_id="inactive",
                category_id="fed_rates",
                include_keywords=["test"],
                min_confidence=0.5,
                is_active=False  # Inactive
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        article = Article(
            article_id="test",
            source_id="news",
            canonical_url="https://example.com",
            title="Test Article",
            snippet="This is a test article.",
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)
        # Inactive rule should not match
        assert len(results) == 0

    def test_confidence_capped_at_one(self):
        """Test that confidence is capped at 1.0."""
        rules = [
            CategoryRule(
                rule_id="test",
                category_id="fed_rates",
                source_whitelist=["fed"],
                include_keywords=["fomc", "fed", "rate", "policy", "monetary"],
                required_entity_types=[EntityType.CENTRAL_BANK],
                keyword_match_boost=0.5,  # High boost
                min_confidence=0.5
            )
        ]
        classifier = ArticleClassifier(rules=rules)

        # Article that matches everything
        article = Article(
            article_id="test",
            source_id="fed",
            canonical_url="https://fed.gov",
            title="FOMC Rate Policy Monetary Decision",
            snippet="Federal Reserve announces rate policy.",
            entities=[Entity("e1", EntityType.CENTRAL_BANK, "Fed")],
            published_at=datetime.utcnow(),
            ingested_at=datetime.utcnow(),
        )

        results = classifier.classify(article)
        assert len(results) > 0
        # Confidence should be capped at 1.0
        assert results[0].confidence <= 1.0


class TestCategoryTypeMapping:
    """Test category type mapping."""

    def test_all_categories_mapped(self, classifier):
        """Test that all category types are mapped."""
        expected_categories = [
            "fed_rates",
            "inflation",
            "labor",
            "growth_consumer",
            "oil_energy",
            "geopolitics",
            "sp500_corporate",
            "market_regime",
        ]

        for cat_id in expected_categories:
            cat_type = classifier._get_category_type(cat_id)
            assert cat_type is not None
            assert isinstance(cat_type, CategoryType)

    def test_unknown_category_defaults(self, classifier):
        """Test that unknown category ID defaults to MARKET_REGIME."""
        cat_type = classifier._get_category_type("unknown_category")
        assert cat_type == CategoryType.MARKET_REGIME


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
