"""
Base Provider Interfaces

Abstract base classes defining the contract for each pipeline stage.
All providers must implement these interfaces for consistency.
"""

from abc import ABC, abstractmethod
from typing import Any

from news_pipeline.config import ProviderConfig


class BaseProvider(ABC):
    """
    Base class for all AI providers.

    Provides common functionality for initialization, health checks,
    and error handling.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self._client: Any = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Provider name for logging."""
        return self.__class__.__name__

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider and its API client."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass

    async def shutdown(self) -> None:
        """Clean up provider resources."""
        self._client = None
        self._initialized = False


class DiscoveryProvider(BaseProvider):
    """
    Interface for news discovery providers.

    Discovery providers search for breaking financial news and return
    candidate stories for verification.
    """

    @abstractmethod
    async def discover_news(
        self,
        tickers: list[str],
        categories: list[str],
        time_window: str,
        max_results: int = 50,
    ) -> list[dict]:
        """
        Discover news stories matching criteria.

        Args:
            tickers: Stock symbols to search for
            categories: News categories to include
            time_window: Time range (e.g., "overnight", "last_6h")
            max_results: Maximum stories to return

        Returns:
            List of candidate story dictionaries
        """
        pass


class VerificationProvider(BaseProvider):
    """
    Interface for story verification providers.

    Verification providers check stories against multiple sources
    and assign confidence scores.
    """

    @abstractmethod
    async def verify_story(
        self,
        headline: str,
        source_url: str,
        tickers: list[str],
        category: str,
    ) -> dict:
        """
        Verify a story using cross-source fact-checking.

        Args:
            headline: Story headline to verify
            source_url: Original source URL
            tickers: Related stock symbols
            category: Story category

        Returns:
            Verification result with confidence score
        """
        pass


class FormattingProvider(BaseProvider):
    """
    Interface for story formatting providers.

    Formatting providers transform verified facts into clear,
    structured content.
    """

    @abstractmethod
    async def format_story(
        self,
        story_id: str,
        verified_facts: list[str],
        what_happened: str,
        affected_assets: list[str],
        category: str,
    ) -> dict:
        """
        Format verified story into structured content.

        Args:
            story_id: Unique story identifier
            verified_facts: List of verified facts
            what_happened: Summary of what occurred
            affected_assets: Related tickers/assets
            category: Story category

        Returns:
            Formatted story dictionary
        """
        pass


class EditorialProvider(BaseProvider):
    """
    Interface for editorial polish providers.

    Editorial providers add "why it matters" analysis and
    ensure professional quality.
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
    ) -> dict:
        """
        Finalize story with editorial polish.

        Args:
            story_id: Unique story identifier
            title: Story title
            what_happened: Summary of what occurred
            bullet_points: Key points
            affected_assets: Related tickers/assets
            category: Story category
            confidence: Verification confidence (0-10)

        Returns:
            Finalized story with "why it matters"
        """
        pass
