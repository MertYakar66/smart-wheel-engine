"""
Story Clustering and Deduplication

Clusters similar articles into stories to:
- Remove duplicate coverage
- Track story evolution over time
- Measure source diversity (confidence)

Uses:
- URL canonicalization
- Title similarity (simhash/minhash)
- Entity overlap
- Temporal proximity
"""

import logging
import re
from datetime import datetime, timedelta

from financial_news.models import Article, Story

logger = logging.getLogger(__name__)


class StoryClusterer:
    """
    Clusters articles into stories and deduplicates coverage.

    Key principles:
    - Same URL = same article (dedupe immediately)
    - Similar titles + temporal proximity = same story
    - Entity overlap strengthens clustering
    - More sources = higher confidence
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        time_window_hours: int = 48,
        min_sources_for_confidence: int = 3,
    ):
        """
        Initialize story clusterer.

        Args:
            similarity_threshold: Minimum title similarity to cluster (0-1)
            time_window_hours: Max hours between articles in same story
            min_sources_for_confidence: Sources needed for high confidence
        """
        self.similarity_threshold = similarity_threshold
        self.time_window_hours = time_window_hours
        self.min_sources_for_confidence = min_sources_for_confidence

        # Existing stories for incremental clustering
        self._stories: dict[str, Story] = {}
        self._url_to_story: dict[str, str] = {}
        self._article_hashes: dict[str, str] = {}
        self._article_story_map: dict[str, str] = {}

    def cluster_articles(
        self,
        articles: list[Article],
        existing_stories: list[Story] | None = None,
    ) -> list[Story]:
        """
        Cluster articles into stories.

        Args:
            articles: New articles to cluster
            existing_stories: Existing stories to merge into

        Returns:
            List of updated/new stories
        """
        # Load existing stories
        if existing_stories:
            for story in existing_stories:
                self._stories[story.story_id] = story
                for article_id in story.article_ids:
                    # Track article IDs for deduplication
                    self._article_story_map[article_id] = story.story_id

        # Deduplicate by URL first
        unique_articles = self._dedupe_by_url(articles)

        # Group by entity/topic similarity
        clusters = self._cluster_by_similarity(unique_articles)

        # Convert clusters to stories
        stories = []
        for cluster in clusters:
            story = self._cluster_to_story(cluster)
            if story:
                stories.append(story)
                self._stories[story.story_id] = story

        logger.info(
            f"Clustered {len(articles)} articles into {len(stories)} stories "
            f"(after deduping {len(articles) - len(unique_articles)} duplicates)"
        )

        return stories

    def _dedupe_by_url(self, articles: list[Article]) -> list[Article]:
        """Remove duplicate articles by canonical URL"""
        seen_urls = set()
        unique = []

        for article in articles:
            url = article.canonical_url.lower()
            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(article)

        return unique

    def _cluster_by_similarity(
        self,
        articles: list[Article],
    ) -> list[list[Article]]:
        """
        Cluster articles by title similarity and entity overlap.

        Uses greedy clustering with similarity threshold.
        """
        if not articles:
            return []

        # Sort by time for temporal locality
        sorted_articles = sorted(articles, key=lambda a: a.published_at_utc)

        clusters: list[list[Article]] = []
        clustered: set[str] = set()

        for article in sorted_articles:
            if article.article_id in clustered:
                continue

            # Start new cluster with this article
            cluster = [article]
            clustered.add(article.article_id)

            # Find similar articles
            for candidate in sorted_articles:
                if candidate.article_id in clustered:
                    continue

                # Check temporal proximity
                time_diff = abs(
                    (article.published_at_utc - candidate.published_at_utc).total_seconds()
                )
                if time_diff > self.time_window_hours * 3600:
                    continue

                # Check similarity
                similarity = self._compute_similarity(article, candidate)
                if similarity >= self.similarity_threshold:
                    cluster.append(candidate)
                    clustered.add(candidate.article_id)

            clusters.append(cluster)

        return clusters

    def _compute_similarity(self, a1: Article, a2: Article) -> float:
        """
        Compute similarity between two articles.

        Combines:
        - Title similarity (jaccard on words)
        - Entity overlap
        - Topic overlap
        """
        # Title similarity (word-level jaccard)
        title_sim = self._title_similarity(a1.title, a2.title)

        # Entity overlap
        entities1 = {e.name.lower() for e in a1.entities}
        entities2 = {e.name.lower() for e in a2.entities}
        entity_sim = self._jaccard(entities1, entities2) if entities1 and entities2 else 0

        # Ticker overlap
        tickers1 = set(a1.tickers)
        tickers2 = set(a2.tickers)
        ticker_sim = self._jaccard(tickers1, tickers2) if tickers1 and tickers2 else 0

        # Topic overlap
        topics1 = {t.value for t in a1.topics}
        topics2 = {t.value for t in a2.topics}
        topic_sim = self._jaccard(topics1, topics2) if topics1 and topics2 else 0

        # Weighted combination
        similarity = (
            0.5 * title_sim +
            0.2 * entity_sim +
            0.2 * ticker_sim +
            0.1 * topic_sim
        )

        return similarity

    def _title_similarity(self, t1: str, t2: str) -> float:
        """Compute word-level Jaccard similarity between titles"""
        words1 = set(self._normalize_title(t1).split())
        words2 = set(self._normalize_title(t2).split())
        return self._jaccard(words1, words2)

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        # Lowercase
        title = title.lower()
        # Remove punctuation
        title = re.sub(r'[^\w\s]', '', title)
        # Remove common words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
        words = [w for w in title.split() if w not in stopwords]
        return ' '.join(words)

    def _jaccard(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _cluster_to_story(self, cluster: list[Article]) -> Story | None:
        """Convert article cluster to Story object"""
        if not cluster:
            return None

        # Sort by time
        sorted_cluster = sorted(cluster, key=lambda a: a.published_at_utc)

        # Lead article is the one with most entities or earliest
        lead_article = max(cluster, key=lambda a: len(a.entities))

        # Aggregate metadata
        all_entities = []
        all_tickers = set()
        all_topics = set()
        all_regions = set()
        source_names = set()

        for article in cluster:
            all_entities.extend(article.entities)
            all_tickers.update(article.tickers)
            all_topics.update(article.topics)
            source_names.add(article.source_name)

        # Dedupe entities
        seen_entities = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity)

        # Create story
        story_id = Story.generate_id(
            lead_article.article_id,
            sorted_cluster[0].published_at_utc
        )

        story = Story(
            story_id=story_id,
            lead_article_id=lead_article.article_id,
            headline=lead_article.title,
            summary="",  # Will be generated by BriefGenerator
            why_it_matters="",  # Will be generated by BriefGenerator
            first_seen_at=sorted_cluster[0].published_at_utc,
            last_updated_at=sorted_cluster[-1].published_at_utc,
            entities=unique_entities,
            tickers=list(all_tickers),
            topics=list(all_topics),
            regions=list(all_regions),
            article_ids=[a.article_id for a in cluster],
            source_count=len(source_names),
        )

        # Confidence based on source diversity
        story.confidence_score = min(1.0, story.source_count / self.min_sources_for_confidence)

        return story

    def merge_story(self, existing: Story, new_articles: list[Article]) -> Story:
        """
        Merge new articles into an existing story.

        Updates the story's metadata and tracks changes.
        """
        # Store previous summary for change tracking
        existing.previous_summary = existing.summary

        # Add new article IDs
        new_ids = [a.article_id for a in new_articles]
        existing.article_ids.extend(new_ids)

        # Update time
        latest_time = max(a.published_at_utc for a in new_articles)
        if latest_time > existing.last_updated_at:
            existing.last_updated_at = latest_time

        # Update source count
        new_sources = {a.source_name for a in new_articles}
        # This is approximate - would need to track source names in story
        existing.source_count += len(new_sources)

        # Add new entities
        for article in new_articles:
            for entity in article.entities:
                key = (entity.name.lower(), entity.entity_type)
                existing_keys = {(e.name.lower(), e.entity_type) for e in existing.entities}
                if key not in existing_keys:
                    existing.entities.append(entity)

        # Update tickers
        for article in new_articles:
            for ticker in article.tickers:
                if ticker not in existing.tickers:
                    existing.tickers.append(ticker)

        # Recalculate confidence
        existing.confidence_score = min(
            1.0,
            existing.source_count / self.min_sources_for_confidence
        )

        return existing

    def find_matching_story(self, article: Article) -> Story | None:
        """
        Find an existing story that this article belongs to.

        Used for incremental updates.
        """
        best_match = None
        best_similarity = 0.0

        for story in self._stories.values():
            # Check temporal proximity
            time_diff = abs(
                (article.published_at_utc - story.last_updated_at).total_seconds()
            )
            if time_diff > self.time_window_hours * 3600:
                continue

            # Check title similarity with story headline
            title_sim = self._title_similarity(article.title, story.headline)

            # Check entity/ticker overlap
            article_tickers = set(article.tickers)
            story_tickers = set(story.tickers)
            ticker_sim = self._jaccard(article_tickers, story_tickers)

            similarity = 0.6 * title_sim + 0.4 * ticker_sim

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = story

        return best_match

    def get_story(self, story_id: str) -> Story | None:
        """Get story by ID"""
        return self._stories.get(story_id)

    def get_all_stories(self) -> list[Story]:
        """Get all current stories"""
        return list(self._stories.values())

    def clear_old_stories(self, max_age_hours: int = 72) -> int:
        """Remove stories older than max_age_hours"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_ids = [
            sid for sid, story in self._stories.items()
            if story.last_updated_at < cutoff
        ]

        for sid in old_ids:
            del self._stories[sid]

        return len(old_ids)
