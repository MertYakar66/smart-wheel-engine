"""
Base Provider Interface

Defines the abstract interface for AI model providers used in the pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for AI model providers.

    Each provider (Grok, Gemini, ChatGPT, Claude) implements this interface.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider. If None, will try to load from environment.
        """
        self.api_key = api_key
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider connection.

        Should validate API key and set up any required clients.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available and responding.

        Returns:
            True if healthy, False otherwise.
        """
        ...

    def _log_request(self, operation: str, params: dict[str, Any]) -> None:
        """Log an outgoing request."""
        logger.info(f"[{self.name}] {operation}: {params}")

    def _log_response(self, operation: str, result: Any) -> None:
        """Log an incoming response."""
        logger.debug(f"[{self.name}] {operation} response: {result}")

    def _log_error(self, operation: str, error: Exception) -> None:
        """Log an error."""
        logger.error(f"[{self.name}] {operation} error: {error}")


class DiscoveryProvider(BaseProvider):
    """
    Provider interface for news discovery.

    Used by Grok for web search and news gathering.
    """

    @abstractmethod
    async def discover_news(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Discover news stories matching the criteria.

        Args:
            tickers: Stock tickers to search for
            categories: News categories to include
            time_window: Time window (e.g., "overnight", "last_6h")
            max_results: Maximum number of results

        Returns:
            List of raw story dictionaries
        """
        ...


class VerificationProvider(BaseProvider):
    """
    Provider interface for story verification.

    Used by Gemini for cross-source verification.
    """

    @abstractmethod
    async def verify_story(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> dict[str, Any]:
        """
        Verify a story across multiple sources.

        Args:
            headline: Story headline to verify
            source_url: Original source URL
            tickers: Related tickers
            category: Story category

        Returns:
            Verification result with confidence score and evidence
        """
        ...


class FormattingProvider(BaseProvider):
    """
    Provider interface for story formatting.

    Used by ChatGPT for structuring verified stories.
    """

    @abstractmethod
    async def format_story(
        self,
        story_id: str,
        verified_facts: list[str],
        verification_confidence: int,
        affected_assets: list[str],
        category: str,
    ) -> dict[str, Any]:
        """
        Format verified facts into a structured story.

        Args:
            story_id: Story identifier
            verified_facts: List of verified facts
            verification_confidence: Confidence score (0-10)
            affected_assets: List of affected assets
            category: Story category

        Returns:
            Formatted story dictionary
        """
        ...


class EditorialProvider(BaseProvider):
    """
    Provider interface for editorial refinement.

    Used by Claude for final polish and "why it matters" generation.
    """

    @abstractmethod
    async def finalize_story(
        self,
        story_id: str,
        title: str,
        what_happened: str,
        bullet_points: list[str],
        affected_assets: list[str],
        category: str,
        confidence: int,
    ) -> dict[str, Any]:
        """
        Finalize a story with editorial polish.

        Args:
            story_id: Story identifier
            title: Draft title
            what_happened: Draft summary
            bullet_points: Draft bullet points
            affected_assets: List of affected assets
            category: Story category
            confidence: Verification confidence

        Returns:
            Finalized story with "why it matters" section
        """
        ...
