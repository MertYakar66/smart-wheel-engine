"""
Article Classifier - Deterministic Rule-Based Classification

Classification order:
1. Source-to-category mapping (Fed pages -> FED_RATES)
2. Keyword matching with required/exclude lists
3. Entity-based boosts

No ML needed for v1 - deterministic rules are sufficient
for structured official sources.
"""

import logging
from dataclasses import dataclass

from financial_news.schema import (
    DEFAULT_CATEGORY_RULES,
    Article,
    CategoryRule,
    CategoryType,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of classifying an article"""

    category_id: str
    category_type: CategoryType
    confidence: float
    matched_rule: str
    matched_keywords: list[str]


class ArticleClassifier:
    """
    Rule-based article classifier.

    Uses deterministic rules to classify articles into categories.
    Priority:
    1. Source whitelist (highest confidence)
    2. Required keywords
    3. Include keywords (with boost)
    4. Entity matches
    """

    def __init__(self, rules: list[CategoryRule] | None = None):
        self.rules = rules or DEFAULT_CATEGORY_RULES
        # Index rules by category for faster lookup
        self._rules_by_category: dict[str, list[CategoryRule]] = {}
        for rule in self.rules:
            if rule.category_id not in self._rules_by_category:
                self._rules_by_category[rule.category_id] = []
            self._rules_by_category[rule.category_id].append(rule)

        # Sort by priority
        for cat_rules in self._rules_by_category.values():
            cat_rules.sort(key=lambda r: r.priority)

    def classify(self, article: Article) -> list[ClassificationResult]:
        """
        Classify an article into categories.

        Returns list of matching categories with confidence scores.
        An article can belong to multiple categories.
        """
        results = []
        text = self._prepare_text(article)

        for rule in self.rules:
            if not rule.is_active:
                continue

            confidence, matched_keywords = self._evaluate_rule(article, rule, text)

            if confidence >= rule.min_confidence:
                results.append(
                    ClassificationResult(
                        category_id=rule.category_id,
                        category_type=self._get_category_type(rule.category_id),
                        confidence=confidence,
                        matched_rule=rule.rule_id,
                        matched_keywords=matched_keywords,
                    )
                )

        # Deduplicate by category (keep highest confidence)
        best_by_category: dict[str, ClassificationResult] = {}
        for result in results:
            if (
                result.category_id not in best_by_category
                or result.confidence > best_by_category[result.category_id].confidence
            ):
                best_by_category[result.category_id] = result

        # Sort by confidence
        final = list(best_by_category.values())
        final.sort(key=lambda r: r.confidence, reverse=True)

        return final

    def classify_batch(self, articles: list[Article]) -> dict[str, list[ClassificationResult]]:
        """Classify multiple articles."""
        return {article.article_id: self.classify(article) for article in articles}

    def _prepare_text(self, article: Article) -> str:
        """Prepare article text for matching."""
        parts = [article.title]
        if article.snippet:
            parts.append(article.snippet)
        return " ".join(parts).lower()

    def _evaluate_rule(
        self,
        article: Article,
        rule: CategoryRule,
        text: str,
    ) -> tuple[float, list[str]]:
        """
        Evaluate a rule against an article.

        Returns (confidence, matched_keywords).
        """
        confidence = 0.0
        matched_keywords = []

        # 1. Source whitelist (highest priority)
        if rule.source_whitelist:
            if article.source_id in rule.source_whitelist:
                confidence = 0.95
                matched_keywords.append(f"source:{article.source_id}")
            else:
                # If source whitelist exists but doesn't match, reduce confidence
                confidence = max(0, confidence - 0.2)

        # 2. Check exclude keywords first
        if rule.exclude_keywords:
            for keyword in rule.exclude_keywords:
                if keyword.lower() in text:
                    return 0.0, []  # Excluded

        # 3. Required keywords (must have at least one)
        if rule.required_keywords:
            found_required = False
            for keyword in rule.required_keywords:
                if keyword.lower() in text:
                    found_required = True
                    matched_keywords.append(keyword)
                    break
            if not found_required:
                return 0.0, []  # Missing required keyword

        # 4. Include keywords (boost)
        if rule.include_keywords:
            match_count = 0
            for keyword in rule.include_keywords:
                if keyword.lower() in text:
                    match_count += 1
                    matched_keywords.append(keyword)

            if match_count > 0:
                # More matches = higher confidence
                keyword_boost = min(0.4, match_count * rule.keyword_match_boost)
                confidence = max(confidence, 0.5) + keyword_boost

        # 5. Entity type matching
        if rule.required_entity_types and article.entities:
            entity_types = {e.entity_type for e in article.entities}
            for req_type in rule.required_entity_types:
                if req_type in entity_types:
                    confidence += 0.1
                    matched_keywords.append(f"entity:{req_type.value}")

        # 6. Ticker whitelist
        if rule.ticker_whitelist and article.tickers:
            for ticker in article.tickers:
                if ticker in rule.ticker_whitelist:
                    confidence += 0.15
                    matched_keywords.append(f"ticker:{ticker}")
                    break

        return min(1.0, confidence), matched_keywords

    def _get_category_type(self, category_id: str) -> CategoryType:
        """Get CategoryType from category_id."""
        mapping = {
            "fed_rates": CategoryType.FED_RATES,
            "inflation": CategoryType.INFLATION,
            "labor": CategoryType.LABOR,
            "growth_consumer": CategoryType.GROWTH_CONSUMER,
            "oil_energy": CategoryType.OIL_ENERGY,
            "geopolitics": CategoryType.GEOPOLITICS,
            "sp500_corporate": CategoryType.SP500_CORPORATE,
            "market_regime": CategoryType.MARKET_REGIME,
        }
        return mapping.get(category_id, CategoryType.MARKET_REGIME)

    def get_primary_category(self, article: Article) -> CategoryType | None:
        """Get the primary (highest confidence) category for an article."""
        results = self.classify(article)
        if results:
            return results[0].category_type
        return None

    def update_article_categories(self, article: Article) -> Article:
        """Classify and update article's categories field."""
        results = self.classify(article)
        article.categories = [r.category_type for r in results]
        return article
