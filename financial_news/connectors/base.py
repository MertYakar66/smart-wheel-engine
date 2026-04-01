"""
Base Connector with Rate Limiting and Error Handling

Provides:
- Token bucket rate limiting (respects SEC's 10 req/sec)
- Exponential backoff retry
- Request/response logging
- Connection pooling via aiohttp
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp

from financial_news.schema import Article, Source

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Enforces rate limits like SEC EDGAR's 10 requests/second.
    """

    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0

    # Internal state
    _tokens: float = field(default=10.0, init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)
    _minute_requests: int = field(default=0, init=False)
    _minute_start: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()

            # Refill tokens based on time elapsed
            elapsed = now - self._last_refill
            self._tokens = min(
                self.requests_per_second, self._tokens + elapsed * self.requests_per_second
            )
            self._last_refill = now

            # Check minute limit
            if now - self._minute_start >= 60:
                self._minute_requests = 0
                self._minute_start = now

            # Wait if no tokens available
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.requests_per_second
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._tokens = 1

            # Wait if minute limit exceeded
            if self._minute_requests >= self.requests_per_minute:
                wait_time = 60 - (now - self._minute_start)
                if wait_time > 0:
                    logger.debug(f"Minute limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                self._minute_requests = 0
                self._minute_start = time.monotonic()

            # Consume token
            self._tokens -= 1
            self._minute_requests += 1


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    success: bool
    url: str
    status_code: int = 0
    content: str | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    fetch_time_ms: float = 0
    retry_count: int = 0


class BaseConnector(ABC):
    """
    Base class for all source connectors.

    Provides:
    - Async HTTP client with connection pooling
    - Rate limiting
    - Exponential backoff retry
    - Request logging
    """

    def __init__(
        self,
        source: Source,
        user_agent: str = "SmartWheelEngine/2.0 (Financial Research)",
        max_retries: int = 3,
        base_timeout: float = 30.0,
    ):
        self.source = source
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.base_timeout = base_timeout

        # Rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_second=source.rate_limit_per_second,
            requests_per_minute=source.rate_limit_per_minute,
        )

        # HTTP session (created on first use)
        self._session: aiohttp.ClientSession | None = None

        # Stats
        self.requests_made = 0
        self.requests_failed = 0
        self.last_request_at: datetime | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.base_timeout)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json, text/html, application/xml",
                },
            )
        return self._session

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        """
        Fetch a URL with rate limiting and retry.

        Args:
            url: URL to fetch
            method: HTTP method
            params: Query parameters
            headers: Additional headers

        Returns:
            FetchResult with success/failure and content
        """
        await self.rate_limiter.acquire()

        session = await self._get_session()
        start_time = time.monotonic()
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                async with session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                ) as response:
                    content = await response.text()
                    fetch_time = (time.monotonic() - start_time) * 1000

                    self.requests_made += 1
                    self.last_request_at = datetime.utcnow()

                    if response.status == 200:
                        return FetchResult(
                            success=True,
                            url=url,
                            status_code=response.status,
                            content=content,
                            content_type=response.headers.get("Content-Type"),
                            headers=dict(response.headers),
                            fetch_time_ms=fetch_time,
                            retry_count=retry_count,
                        )

                    # Rate limited - wait and retry
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited by {url}, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        retry_count += 1
                        continue

                    # Server error - retry with backoff
                    if response.status >= 500:
                        wait_time = (2**retry_count) * 1.0
                        logger.warning(
                            f"Server error {response.status} from {url}, retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        retry_count += 1
                        continue

                    # Client error - don't retry
                    self.requests_failed += 1
                    return FetchResult(
                        success=False,
                        url=url,
                        status_code=response.status,
                        content=content,
                        error=f"HTTP {response.status}",
                        fetch_time_ms=fetch_time,
                        retry_count=retry_count,
                    )

            except TimeoutError:
                wait_time = (2**retry_count) * 1.0
                logger.warning(f"Timeout fetching {url}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                retry_count += 1

            except aiohttp.ClientError as e:
                wait_time = (2**retry_count) * 1.0
                logger.warning(f"Client error fetching {url}: {e}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                retry_count += 1

            except Exception as e:
                self.requests_failed += 1
                logger.error(f"Unexpected error fetching {url}: {e}")
                return FetchResult(
                    success=False,
                    url=url,
                    error=str(e),
                    fetch_time_ms=(time.monotonic() - start_time) * 1000,
                    retry_count=retry_count,
                )

        # Max retries exceeded
        self.requests_failed += 1
        return FetchResult(
            success=False,
            url=url,
            error=f"Max retries ({self.max_retries}) exceeded",
            fetch_time_ms=(time.monotonic() - start_time) * 1000,
            retry_count=retry_count,
        )

    async def fetch_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[bool, Any]:
        """
        Fetch and parse JSON.

        Returns:
            Tuple of (success, data_or_error)
        """
        import json

        result = await self.fetch(url, params=params, headers={"Accept": "application/json"})

        if not result.success:
            return False, result.error

        try:
            data = json.loads(result.content)
            return True, data
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e}"

    @abstractmethod
    async def fetch_latest(
        self,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Article]:
        """
        Fetch latest articles from the source.

        Args:
            since: Only fetch articles after this time
            limit: Maximum articles to fetch

        Returns:
            List of Article objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the source is accessible.

        Returns:
            True if healthy
        """
        pass

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics."""
        return {
            "source_id": self.source.source_id,
            "requests_made": self.requests_made,
            "requests_failed": self.requests_failed,
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
            "success_rate": (
                (self.requests_made - self.requests_failed) / self.requests_made
                if self.requests_made > 0
                else 1.0
            ),
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
