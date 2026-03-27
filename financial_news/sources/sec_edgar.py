"""
SEC EDGAR Fetcher - US Regulatory Filings

SEC EDGAR provides:
- Official company filings (10-K, 10-Q, 8-K, etc.)
- Insider transactions (Form 4)
- Proxy statements
- IPO filings (S-1)

Rate Limits (from SEC):
- Max 10 requests per second
- Must include User-Agent with contact email
- Use Fair Access for heavy users

Reference: https://www.sec.gov/developer
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging
import re

import httpx

from financial_news.models import Article, ArticleSource, Category, Entity, TopicCategory
from financial_news.sources.base import BaseSourceFetcher

logger = logging.getLogger(__name__)


# Filing type to topic mapping
FILING_TOPICS = {
    "8-K": [TopicCategory.EARNINGS],  # Material events
    "10-K": [TopicCategory.EARNINGS],  # Annual report
    "10-Q": [TopicCategory.EARNINGS],  # Quarterly report
    "S-1": [TopicCategory.IPO],  # IPO registration
    "DEF 14A": [],  # Proxy statement
    "4": [],  # Insider transaction
    "SC 13D": [TopicCategory.M_AND_A],  # Activist stake
    "SC 13G": [],  # Passive stake
    "8-A": [],  # Registration
}


class SECEdgarFetcher(BaseSourceFetcher):
    """
    Fetches regulatory filings from SEC EDGAR.

    Provides "filings-driven news" for:
    - 8-K material events (earnings, guidance, leadership changes)
    - 10-Q/10-K periodic reports
    - Insider transactions (Form 4)
    - IPO filings (S-1)
    """

    # SEC EDGAR API endpoints
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    # User agent required by SEC
    USER_AGENT = "SmartWheelEngine/1.0 (contact@example.com)"

    def __init__(self, rate_limit_per_second: float = 10.0, user_email: str = "contact@example.com"):
        """
        Initialize SEC EDGAR fetcher.

        Args:
            rate_limit_per_second: Max 10 per SEC guidelines
            user_email: Contact email for User-Agent (required by SEC)
        """
        super().__init__(rate_limit_per_second=rate_limit_per_second)
        self.user_email = user_email
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with SEC-required headers"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers={
                    "User-Agent": f"SmartWheelEngine/1.0 ({self.user_email})",
                    "Accept-Encoding": "gzip, deflate",
                    "Host": "data.sec.gov",
                },
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if SEC EDGAR is accessible"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.BASE_URL}/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=1&output=atom")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"SEC EDGAR health check failed: {e}")
            return False

    async def fetch_articles(
        self,
        category: Category,
        start_time: datetime,
        end_time: datetime,
        max_results: int = 100,
    ) -> List[Article]:
        """
        Fetch SEC filings as articles.

        Maps filings to news-like articles with metadata.
        """
        articles = []

        # Determine which filing types to fetch based on category
        filing_types = self._get_filing_types_for_category(category)

        for filing_type in filing_types:
            await self._rate_limit()

            filings = await self._fetch_recent_filings(
                filing_type=filing_type,
                start_date=start_time.date(),
                end_date=end_time.date(),
                max_results=max_results // len(filing_types),
                tickers=category.tickers,
            )

            articles.extend(filings)

        logger.info(
            f"SEC EDGAR: Fetched {len(articles)} filings for '{category.name}'"
        )

        return articles[:max_results]

    def _get_filing_types_for_category(self, category: Category) -> List[str]:
        """Map category to relevant SEC filing types"""
        filing_types = []

        if TopicCategory.EARNINGS in category.topics:
            filing_types.extend(["8-K", "10-Q", "10-K"])

        if TopicCategory.IPO in category.topics:
            filing_types.append("S-1")

        if TopicCategory.M_AND_A in category.topics:
            filing_types.extend(["8-K", "SC 13D"])

        # Default to 8-K for any financial category
        if not filing_types:
            filing_types = ["8-K"]

        return list(set(filing_types))

    async def _fetch_recent_filings(
        self,
        filing_type: str,
        start_date: datetime.date,
        end_date: datetime.date,
        max_results: int = 50,
        tickers: Optional[List[str]] = None,
    ) -> List[Article]:
        """
        Fetch recent filings of a specific type.

        Uses SEC's RSS feed for recent filings.
        """
        articles = []

        # Use SEC RSS feed for recent filings
        rss_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcurrent&type={filing_type}&count={min(max_results, 100)}&output=atom"
        )

        try:
            client = await self._get_client()
            # Override host header for this request
            response = await client.get(
                rss_url,
                headers={"Host": "www.sec.gov"}
            )
            response.raise_for_status()

            # Parse Atom feed
            articles = self._parse_atom_feed(response.text, filing_type, tickers)

            # Filter by date
            articles = [
                a for a in articles
                if start_date <= a.published_at_utc.date() <= end_date
            ]

        except Exception as e:
            logger.error(f"SEC filing fetch error for {filing_type}: {e}")

        return articles

    def _parse_atom_feed(
        self,
        xml_content: str,
        filing_type: str,
        tickers: Optional[List[str]] = None,
    ) -> List[Article]:
        """Parse SEC Atom feed into Article objects"""
        import xml.etree.ElementTree as ET

        articles = []

        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall(".//atom:entry", ns):
                try:
                    title = entry.findtext("atom:title", "", ns)
                    link = entry.find("atom:link", ns)
                    url = link.get("href", "") if link is not None else ""
                    updated = entry.findtext("atom:updated", "", ns)
                    summary = entry.findtext("atom:summary", "", ns)

                    if not url:
                        continue

                    # Parse date
                    pub_date = datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else datetime.utcnow()

                    # Extract company name and CIK from title
                    company_name, cik = self._parse_filing_title(title)

                    # Filter by ticker if specified
                    if tickers:
                        # Would need CIK->ticker mapping here
                        # For now, include all
                        pass

                    article_id = Article.generate_id(url, pub_date, "sec_edgar")

                    article = Article(
                        article_id=article_id,
                        canonical_url=url,
                        source=ArticleSource.SEC_EDGAR,
                        source_name="SEC EDGAR",
                        published_at_utc=pub_date,
                        title=f"[{filing_type}] {company_name}",
                        snippet=summary[:300] if summary else title,
                        language="en",
                        country="US",
                        retrieval_provider="sec_edgar",
                    )

                    # Add company entity
                    if company_name:
                        article.entities.append(Entity(
                            name=company_name,
                            entity_type="company",
                            confidence=1.0,
                        ))

                    # Add topics
                    article.topics = FILING_TOPICS.get(filing_type, []).copy()

                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Error parsing SEC entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return articles

    def _parse_filing_title(self, title: str) -> tuple:
        """Extract company name and CIK from SEC filing title"""
        # SEC titles often follow pattern: "FORM-TYPE - COMPANY NAME (CIK)"
        company_name = title
        cik = None

        # Try to extract CIK
        cik_match = re.search(r'\((\d+)\)', title)
        if cik_match:
            cik = cik_match.group(1)
            company_name = title[:cik_match.start()].strip()

        # Remove form type prefix
        for prefix in ["8-K", "10-K", "10-Q", "S-1", "DEF 14A", "Form 4"]:
            if company_name.startswith(prefix):
                company_name = company_name[len(prefix):].strip(" -")

        return company_name, cik

    async def get_company_filings(
        self,
        cik: str,
        filing_types: Optional[List[str]] = None,
        max_results: int = 20,
    ) -> List[Article]:
        """
        Get all recent filings for a specific company by CIK.

        Args:
            cik: SEC Central Index Key (10-digit, zero-padded)
            filing_types: Filter to specific types
            max_results: Maximum filings to return
        """
        await self._rate_limit()

        # Zero-pad CIK to 10 digits
        cik_padded = cik.zfill(10)

        try:
            client = await self._get_client()
            response = await client.get(f"{self.SUBMISSIONS_URL}/CIK{cik_padded}.json")
            response.raise_for_status()

            data = response.json()
            articles = self._parse_company_submissions(data, filing_types)

            return articles[:max_results]

        except Exception as e:
            logger.error(f"Error fetching company filings for CIK {cik}: {e}")
            return []

    def _parse_company_submissions(
        self,
        data: Dict[str, Any],
        filing_types: Optional[List[str]] = None,
    ) -> List[Article]:
        """Parse company submissions JSON into Article objects"""
        articles = []

        company_name = data.get("name", "Unknown")
        cik = data.get("cik", "")
        tickers = data.get("tickers", [])

        # Get recent filings
        recent = data.get("filings", {}).get("recent", {})

        form_types = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_nums = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i in range(min(len(form_types), 50)):
            form_type = form_types[i]

            # Filter by type if specified
            if filing_types and form_type not in filing_types:
                continue

            filing_date = filing_dates[i]
            accession = accession_nums[i].replace("-", "")
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            # Build URL to filing
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
            )

            pub_date = datetime.strptime(filing_date, "%Y-%m-%d")
            article_id = Article.generate_id(url, pub_date, "sec_edgar")

            article = Article(
                article_id=article_id,
                canonical_url=url,
                source=ArticleSource.SEC_EDGAR,
                source_name="SEC EDGAR",
                published_at_utc=pub_date,
                title=f"[{form_type}] {company_name}",
                snippet=f"{company_name} filed {form_type} with the SEC on {filing_date}",
                language="en",
                country="US",
                tickers=tickers,
                retrieval_provider="sec_edgar",
            )

            # Add company entity
            article.entities.append(Entity(
                name=company_name,
                entity_type="company",
                ticker=tickers[0] if tickers else None,
                confidence=1.0,
            ))

            article.topics = FILING_TOPICS.get(form_type, []).copy()
            articles.append(article)

        return articles

    async def search_filings(
        self,
        query: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 50,
    ) -> List[Article]:
        """
        Full-text search across SEC filings.

        Uses SEC's EFTS (Electronic Filing Text Search) API.
        """
        await self._rate_limit()

        params = {
            "q": query,
            "dateRange": "custom" if start_date or end_date else "all",
            "startdt": start_date.strftime("%Y-%m-%d") if start_date else None,
            "enddt": end_date.strftime("%Y-%m-%d") if end_date else None,
            "forms": ",".join(filing_types) if filing_types else None,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            client = await self._get_client()
            response = await client.get(
                self.FULL_TEXT_SEARCH_URL,
                params=params,
                headers={"Host": "efts.sec.gov"}
            )
            response.raise_for_status()

            data = response.json()
            # Parse search results (structure varies)
            return self._parse_search_results(data)

        except Exception as e:
            logger.error(f"SEC search error: {e}")
            return []

    def _parse_search_results(self, data: Dict[str, Any]) -> List[Article]:
        """Parse EFTS search results"""
        # EFTS returns different structure than other endpoints
        # Implementation depends on actual response format
        return []
