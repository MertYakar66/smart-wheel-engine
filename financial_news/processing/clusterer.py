"""
Story Clustering - Group Related Articles

Clusters articles into stories based on:
1. Title similarity (n-gram overlap)
2. Entity overlap (tickers, companies)
3. Temporal proximity
4. Category alignment

No ML needed for v1 - simple similarity metrics work well
for structured news.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from financial_news.schema import (
    Article,
    CategoryType,
    Story,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterCandidate:
    """A potential story cluster"""
    articles: list[Article]
    representative_article: Article
    tickers: set[str]
    categories: set[CategoryType]
    first_seen: datetime
    last_updated: datetime

    @property
    def article_count(self) -> int:
        return len(self.articles)

    @property
    def source_count(self) -> int:
        return len({a.source_id for a in self.articles})


class StoryClustering:
    """
    Cluster articles into stories.

    Algorithm:
    1. Group by primary ticker (if present)
    2. Within groups, cluster by title similarity
    3. Merge clusters with high entity overlap
    4. Create Story objects from clusters
    """

    def __init__(
        self,
        title_similarity_threshold: float = 0.4,
        entity_overlap_threshold: float = 0.3,
        time_window_hours: int = 48,
        min_ngram_size: int = 2,
        max_ngram_size: int = 4,
    ):
        self.title_similarity_threshold = title_similarity_threshold
        self.entity_overlap_threshold = entity_overlap_threshold
        self.time_window_hours = time_window_hours
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size

        # Cache for existing stories
        self._story_cache: dict[str, Story] = {}

    def cluster_articles(
        self,
        articles: list[Article],
        existing_stories: list[Story] | None = None,
    ) -> list[Story]:
        """
        Cluster articles into stories.

        Args:
            articles: New articles to cluster
            existing_stories: Existing stories to potentially merge into

        Returns:
            List of Story objects (new and updated)
        """
        if not articles:
            return []

        # Load existing stories into cache
        if existing_stories:
            for story in existing_stories:
                self._story_cache[story.story_id] = story

        # Step 1: Initial clustering by primary ticker
        ticker_groups = self._group_by_ticker(articles)

        # Step 2: Within each group, cluster by title similarity
        candidates: list[ClusterCandidate] = []
        for _ticker, group_articles in ticker_groups.items():
            group_clusters = self._cluster_by_title(group_articles)
            candidates.extend(group_clusters)

        # Step 3: Merge clusters with high entity overlap
        merged = self._merge_similar_clusters(candidates)

        # Step 4: Match with existing stories or create new ones
        stories = self._create_or_update_stories(merged)

        logger.info(f"Clustered {len(articles)} articles into {len(stories)} stories")
        return stories

    def _group_by_ticker(self, articles: list[Article]) -> dict[str, list[Article]]:
        """Group articles by primary ticker."""
        groups: dict[str, list[Article]] = defaultdict(list)

        for article in articles:
            if article.tickers:
                # Use first ticker as primary
                groups[article.tickers[0]].append(article)
            else:
                # No ticker - group by category
                if article.categories:
                    groups[f"_cat_{article.categories[0].value}"].append(article)
                else:
                    groups["_uncategorized"].append(article)

        return dict(groups)

    def _cluster_by_title(self, articles: list[Article]) -> list[ClusterCandidate]:
        """Cluster articles by title similarity using n-gram overlap."""
        if not articles:
            return []

        # Sort by time
        articles = sorted(articles, key=lambda a: a.published_at)

        clusters: list[ClusterCandidate] = []
        assigned: set[str] = set()

        for article in articles:
            if article.article_id in assigned:
                continue

            # Find similar articles
            similar = [article]
            article_ngrams = self._get_ngrams(article.title)

            for other in articles:
                if other.article_id in assigned or other.article_id == article.article_id:
                    continue

                # Check time proximity
                time_diff = abs((article.published_at - other.published_at).total_seconds())
                if time_diff > self.time_window_hours * 3600:
                    continue

                # Check title similarity
                other_ngrams = self._get_ngrams(other.title)
                similarity = self._jaccard_similarity(article_ngrams, other_ngrams)

                if similarity >= self.title_similarity_threshold:
                    similar.append(other)
                    assigned.add(other.article_id)

            assigned.add(article.article_id)

            # Create cluster candidate
            all_tickers = set()
            all_categories = set()
            for a in similar:
                all_tickers.update(a.tickers)
                all_categories.update(a.categories)

            clusters.append(ClusterCandidate(
                articles=similar,
                representative_article=similar[0],
                tickers=all_tickers,
                categories=all_categories,
                first_seen=min(a.published_at for a in similar),
                last_updated=max(a.published_at for a in similar),
            ))

        return clusters

    def _get_ngrams(self, text: str) -> set[str]:
        """Extract n-grams from text."""
        # Normalize
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        ngrams = set()
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            for i in range(len(words) - n + 1):
                ngrams.add(" ".join(words[i:i+n]))

        return ngrams

    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _merge_similar_clusters(
        self,
        candidates: list[ClusterCandidate],
    ) -> list[ClusterCandidate]:
        """Merge clusters with high entity overlap."""
        if len(candidates) <= 1:
            return candidates

        merged: list[ClusterCandidate] = []
        used: set[int] = set()

        for i, c1 in enumerate(candidates):
            if i in used:
                continue

            # Find clusters to merge with
            to_merge = [c1]
            for j, c2 in enumerate(candidates):
                if j <= i or j in used:
                    continue

                # Check ticker overlap
                ticker_overlap = self._jaccard_similarity(c1.tickers, c2.tickers)
                if ticker_overlap >= self.entity_overlap_threshold:
                    to_merge.append(c2)
                    used.add(j)

            used.add(i)

            # Merge clusters
            if len(to_merge) > 1:
                merged_articles = []
                merged_tickers = set()
                merged_categories = set()
                for c in to_merge:
                    merged_articles.extend(c.articles)
                    merged_tickers.update(c.tickers)
                    merged_categories.update(c.categories)

                # Deduplicate articles
                seen_ids = set()
                unique_articles = []
                for a in merged_articles:
                    if a.article_id not in seen_ids:
                        unique_articles.append(a)
                        seen_ids.add(a.article_id)

                merged.append(ClusterCandidate(
                    articles=unique_articles,
                    representative_article=unique_articles[0],
                    tickers=merged_tickers,
                    categories=merged_categories,
                    first_seen=min(a.published_at for a in unique_articles),
                    last_updated=max(a.published_at for a in unique_articles),
                ))
            else:
                merged.append(c1)

        return merged

    def _create_or_update_stories(
        self,
        candidates: list[ClusterCandidate],
    ) -> list[Story]:
        """Create Story objects from cluster candidates."""
        stories = []

        for candidate in candidates:
            lead = candidate.representative_article
            now = datetime.utcnow()

            # Check if this matches an existing story
            existing = self._find_existing_story(candidate)

            if existing:
                # Update existing story
                existing.article_ids.extend(
                    a.article_id for a in candidate.articles
                    if a.article_id not in existing.article_ids
                )
                existing.last_updated_at = now
                existing.source_count = len({
                    a.source_id for a in candidate.articles
                })
                existing.is_developing = True
                stories.append(existing)
            else:
                # Create new story
                story = Story(
                    story_id=Story.generate_id(lead.article_id, candidate.first_seen),
                    lead_article_id=lead.article_id,
                    headline=self._generate_headline(candidate),
                    summary=self._generate_summary(candidate),
                    why_it_matters="",  # To be filled by LLM
                    first_seen_at=candidate.first_seen,
                    last_updated_at=now,
                    tickers=list(candidate.tickers)[:10],
                    affected_sectors=self._infer_sectors(candidate.tickers),
                    affected_factors=self._infer_factors(candidate.categories),
                    article_ids=[a.article_id for a in candidate.articles],
                    source_count=candidate.source_count,
                    category_scores={
                        cat.value: 1.0 / len(candidate.categories)
                        for cat in candidate.categories
                    } if candidate.categories else {},
                    is_developing=candidate.source_count < 3,
                )
                stories.append(story)
                self._story_cache[story.story_id] = story

        return stories

    def _find_existing_story(self, candidate: ClusterCandidate) -> Story | None:
        """Find an existing story that matches this cluster."""
        for story in self._story_cache.values():
            # Check ticker overlap
            story_tickers = set(story.tickers)
            if story_tickers & candidate.tickers:
                # Check time proximity
                time_diff = abs((story.last_updated_at - candidate.first_seen).total_seconds())
                if time_diff < self.time_window_hours * 3600:
                    return story
        return None

    def _generate_headline(self, candidate: ClusterCandidate) -> str:
        """Generate headline from cluster."""
        # Use lead article title, cleaned up
        title = candidate.representative_article.title
        # Remove source prefixes like "Reuters: " or "[WSJ]"
        title = re.sub(r'^\[[^\]]+\]\s*', '', title)
        title = re.sub(r'^[A-Z]+:\s*', '', title)
        return title[:200]

    def _generate_summary(self, candidate: ClusterCandidate) -> str:
        """Generate summary from cluster."""
        lead = candidate.representative_article
        if lead.snippet:
            return lead.snippet[:500]
        return lead.title

    def _infer_sectors(self, tickers: set[str]) -> list[str]:
        """Infer affected sectors from tickers."""
        # Simple sector inference based on common tickers
        sector_map = {
            "XLE": "Energy", "XLF": "Financials", "XLK": "Technology",
            "XLV": "Healthcare", "XLI": "Industrials", "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary", "XLB": "Materials", "XLU": "Utilities",
            "XLRE": "Real Estate", "XLC": "Communications",
        }
        sectors = []
        for ticker in tickers:
            if ticker in sector_map:
                sectors.append(sector_map[ticker])
        return list(set(sectors))[:5]

    def _infer_factors(self, categories: set[CategoryType]) -> list[str]:
        """Infer affected factors from categories."""
        factor_map = {
            CategoryType.FED_RATES: ["rates", "duration"],
            CategoryType.INFLATION: ["inflation", "rates"],
            CategoryType.LABOR: ["growth", "wages"],
            CategoryType.GROWTH_CONSUMER: ["growth", "consumer"],
            CategoryType.OIL_ENERGY: ["oil", "inflation"],
            CategoryType.GEOPOLITICS: ["risk", "defense"],
            CategoryType.SP500_CORPORATE: ["earnings", "equities"],
            CategoryType.MARKET_REGIME: ["risk", "volatility"],
        }
        factors = []
        for cat in categories:
            factors.extend(factor_map.get(cat, []))
        return list(set(factors))[:5]
