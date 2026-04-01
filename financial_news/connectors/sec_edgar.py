"""
SEC EDGAR Connector

Fetches SEC filings:
- 8-K (Current Reports - material events)
- 10-Q (Quarterly Reports)
- 10-K (Annual Reports)
- 4 (Insider Trading)

CRITICAL: SEC enforces 10 requests/second fair access limit.
Exceeding this will result in IP blocking.

Source: https://www.sec.gov/cgi-bin/browse-edgar
API: https://data.sec.gov/
"""

import logging
import re
from datetime import datetime

from financial_news.schema import (
    DEFAULT_SOURCES,
    Article,
    CategoryType,
    Entity,
    EntityType,
    Source,
)

from .base import BaseConnector

logger = logging.getLogger(__name__)


class SECEdgarConnector(BaseConnector):
    """
    Connector for SEC EDGAR filings.

    Uses the SEC's JSON API for recent filings.

    IMPORTANT: Respects SEC's 10 req/sec rate limit.
    User-Agent must identify the application.
    """

    # SEC API endpoints
    SUBMISSIONS_API = "https://data.sec.gov/submissions"
    FILINGS_RSS = "https://www.sec.gov/cgi-bin/browse-edgar"

    # Recent filings API (JSON)
    RECENT_FILINGS_URL = "https://efts.sec.gov/LATEST/search-index"

    # Company tickers API
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # Filing types we care about
    RELEVANT_FILING_TYPES = {
        "8-K": "Current Report - Material Event",
        "8-K/A": "Current Report Amendment",
        "10-Q": "Quarterly Report",
        "10-K": "Annual Report",
        "10-K/A": "Annual Report Amendment",
        "4": "Insider Trading Report",
        "SC 13D": "Beneficial Ownership > 5%",
        "SC 13G": "Beneficial Ownership (Passive)",
        "S-1": "Registration Statement (IPO)",
        "424B": "Prospectus",
    }

    # 8-K item types that matter for trading
    MATERIAL_8K_ITEMS = {
        "1.01": "Entry into Material Agreement",
        "1.02": "Termination of Material Agreement",
        "1.03": "Bankruptcy",
        "2.01": "Completion of Acquisition/Disposition",
        "2.02": "Results of Operations (Earnings)",
        "2.03": "Creation of Direct Financial Obligation",
        "2.04": "Triggering Events (Defaults)",
        "2.05": "Costs Associated with Exit Activities",
        "2.06": "Material Impairments",
        "3.01": "Delisting",
        "4.01": "Change in Accountant",
        "4.02": "Non-Reliance on Financial Statements",
        "5.01": "Changes in Control",
        "5.02": "Departure/Election of Directors/Officers",
        "5.03": "Amendment to Articles",
        "7.01": "Regulation FD Disclosure",
        "8.01": "Other Events",
    }

    def __init__(self, source: Source | None = None):
        if source is None:
            source = next(s for s in DEFAULT_SOURCES if s.source_id == "sec_edgar")

        # SEC requires descriptive User-Agent
        super().__init__(
            source,
            user_agent="SmartWheelEngine/2.0 (financial-research@example.com)",
        )

        # Cache company CIK -> ticker mapping
        self._cik_to_ticker: dict[str, str] = {}
        self._ticker_to_cik: dict[str, str] = {}
        self._company_names: dict[str, str] = {}

    async def _load_company_tickers(self) -> None:
        """Load SEC company tickers mapping."""
        if self._cik_to_ticker:
            return  # Already loaded

        success, data = await self.fetch_json(self.COMPANY_TICKERS_URL)

        if not success:
            logger.warning(f"Failed to load SEC company tickers: {data}")
            return

        try:
            # Data format: {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}, ...}
            for entry in data.values():
                cik = str(entry.get("cik_str", "")).zfill(10)
                ticker = entry.get("ticker", "").upper()
                title = entry.get("title", "")

                if cik and ticker:
                    self._cik_to_ticker[cik] = ticker
                    self._ticker_to_cik[ticker] = cik
                    self._company_names[cik] = title

            logger.info(f"Loaded {len(self._cik_to_ticker)} SEC company mappings")
        except Exception as e:
            logger.error(f"Error parsing SEC company tickers: {e}")

    async def fetch_latest(
        self,
        since: datetime | None = None,
        limit: int = 100,
        filing_types: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> list[Article]:
        """
        Fetch latest SEC filings.

        Args:
            since: Only fetch filings after this time
            limit: Maximum filings to fetch
            filing_types: Filter to specific types (default: 8-K)
            tickers: Filter to specific tickers

        Returns:
            List of Article objects for filings
        """
        await self._load_company_tickers()

        if filing_types is None:
            filing_types = ["8-K"]  # Default to material events

        articles = []

        # Fetch recent filings via RSS
        for filing_type in filing_types:
            type_articles = await self._fetch_filings_by_type(
                filing_type,
                since=since,
                limit=limit // len(filing_types),
                tickers=tickers,
            )
            articles.extend(type_articles)

        # Sort by date and limit
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles[:limit]

    async def _fetch_filings_by_type(
        self,
        filing_type: str,
        since: datetime | None = None,
        limit: int = 50,
        tickers: list[str] | None = None,
    ) -> list[Article]:
        """Fetch filings of a specific type."""
        # Use SEC RSS feed for recent filings
        params = {
            "action": "getcurrent",
            "type": filing_type,
            "company": "",
            "dateb": "",
            "owner": "include",
            "count": str(min(limit, 100)),
            "output": "atom",
        }

        url = self.FILINGS_RSS
        result = await self.fetch(url, params=params)

        if not result.success:
            logger.warning(f"Failed to fetch {filing_type} filings: {result.error}")
            return []

        articles = []
        try:
            # Parse Atom feed
            articles = self._parse_sec_atom_feed(
                result.content,
                filing_type,
                since,
                tickers,
            )
        except Exception as e:
            logger.error(f"Error parsing SEC feed: {e}")

        return articles[:limit]

    def _parse_sec_atom_feed(
        self,
        content: str,
        filing_type: str,
        since: datetime | None,
        tickers: list[str] | None,
    ) -> list[Article]:
        """Parse SEC Atom feed into articles."""
        from xml.etree import ElementTree as ET

        articles = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            root = ET.fromstring(content)

            for entry in root.findall("atom:entry", ns):
                try:
                    title = entry.findtext("atom:title", "", ns).strip()
                    link_elem = entry.find("atom:link", ns)
                    link = link_elem.get("href", "") if link_elem is not None else ""
                    updated = entry.findtext("atom:updated", "", ns)
                    summary = entry.findtext("atom:summary", "", ns)

                    if not title or not link:
                        continue

                    # Parse date
                    pub_date = self._parse_sec_date(updated)
                    if pub_date is None:
                        pub_date = datetime.utcnow()

                    # Filter by since
                    if since and pub_date < since:
                        continue

                    # Extract CIK and accession from URL
                    cik, accession = self._extract_cik_accession(link)

                    # Get ticker from CIK
                    ticker = self._cik_to_ticker.get(cik, "")
                    company_name = self._company_names.get(cik, "")

                    # Filter by tickers if specified
                    if tickers and ticker not in tickers:
                        continue

                    # Build article
                    article = Article(
                        article_id=Article.generate_id(link, self.source.source_id),
                        source_id=self.source.source_id,
                        canonical_url=link,
                        title=f"{ticker or 'Unknown'}: {filing_type} - {title[:100]}",
                        snippet=self._clean_text(summary)[:500] if summary else None,
                        published_at=pub_date,
                        ingested_at=datetime.utcnow(),
                        filing_type=filing_type,
                        cik=cik,
                        accession_number=accession,
                        categories=[CategoryType.SP500_CORPORATE],
                        tickers=[ticker] if ticker else [],
                        entities=self._build_filing_entities(ticker, company_name, filing_type),
                    )
                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing SEC entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Error parsing SEC Atom XML: {e}")

        return articles

    def _parse_sec_date(self, date_str: str) -> datetime | None:
        """Parse SEC date formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip().replace("Z", "+00:00"), fmt)
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        return None

    def _extract_cik_accession(self, url: str) -> tuple:
        """Extract CIK and accession number from SEC URL."""
        # URL format: /cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&...
        # Or: /Archives/edgar/data/320193/000032019324000123/...

        cik_match = re.search(r"CIK[=]?(\d+)", url, re.IGNORECASE)
        if cik_match:
            cik = cik_match.group(1).zfill(10)
        else:
            cik_match = re.search(r"/data/(\d+)/", url)
            cik = cik_match.group(1).zfill(10) if cik_match else ""

        acc_match = re.search(r"(\d{10}-\d{2}-\d{6})", url)
        accession = acc_match.group(1) if acc_match else ""

        return cik, accession

    def _clean_text(self, text: str) -> str:
        """Clean HTML and whitespace from text."""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _build_filing_entities(
        self,
        ticker: str,
        company_name: str,
        filing_type: str,
    ) -> list[Entity]:
        """Build entities for a filing."""
        entities = []

        if ticker:
            entities.append(
                Entity(
                    entity_id=f"ticker_{ticker}",
                    entity_type=EntityType.TICKER,
                    value=ticker,
                    ticker=ticker,
                    confidence=1.0,
                )
            )

        if company_name:
            entities.append(
                Entity(
                    entity_id=f"company_{ticker or 'unknown'}",
                    entity_type=EntityType.COMPANY,
                    value=company_name,
                    ticker=ticker,
                    confidence=1.0,
                )
            )

        return entities

    async def fetch_company_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[Article]:
        """
        Fetch recent filings for a specific company.

        Args:
            ticker: Stock ticker (e.g., "AAPL")
            filing_types: Types to fetch (default: all relevant)
            limit: Maximum filings

        Returns:
            List of Article objects
        """
        await self._load_company_tickers()

        cik = self._ticker_to_cik.get(ticker.upper())
        if not cik:
            logger.warning(f"Unknown ticker: {ticker}")
            return []

        # Fetch company submissions
        url = f"{self.SUBMISSIONS_API}/CIK{cik}.json"
        success, data = await self.fetch_json(url)

        if not success:
            logger.warning(f"Failed to fetch filings for {ticker}: {data}")
            return []

        articles = []
        if filing_types is None:
            filing_types = list(self.RELEVANT_FILING_TYPES.keys())

        try:
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            descriptions = filings.get("primaryDocument", [])

            company_name = data.get("name", "")

            for i in range(min(len(forms), limit * 2)):
                form = forms[i]
                if form not in filing_types:
                    continue

                filing_date = dates[i]
                accession = accessions[i].replace("-", "")
                doc = descriptions[i] if i < len(descriptions) else ""

                # Build URL
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

                pub_date = datetime.strptime(filing_date, "%Y-%m-%d")

                article = Article(
                    article_id=Article.generate_id(url, self.source.source_id),
                    source_id=self.source.source_id,
                    canonical_url=url,
                    title=f"{ticker}: {form} Filed {filing_date}",
                    published_at=pub_date,
                    ingested_at=datetime.utcnow(),
                    filing_type=form,
                    cik=cik,
                    accession_number=accessions[i],
                    categories=[CategoryType.SP500_CORPORATE],
                    tickers=[ticker],
                    entities=self._build_filing_entities(ticker, company_name, form),
                )
                articles.append(article)

                if len(articles) >= limit:
                    break

        except Exception as e:
            logger.error(f"Error parsing company filings: {e}")

        return articles

    async def health_check(self) -> bool:
        """Check if SEC EDGAR is accessible."""
        result = await self.fetch(self.COMPANY_TICKERS_URL)
        return result.success and result.status_code == 200
