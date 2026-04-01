"""
Entity Extraction and Tickerization

Extracts:
- Companies (with ticker resolution)
- People (executives, analysts, policymakers)
- Countries/Regions
- Products/Technologies
- Financial instruments

Uses local AI for extraction, with optional cloud upgrade for accuracy.
"""

import logging
import re
from typing import Any

from financial_news.models import Article, Entity, TopicCategory

logger = logging.getLogger(__name__)


# Common ticker mappings (expandable)
TICKER_MAPPINGS = {
    # Tech Giants
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    # Finance
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C",
    "blackrock": "BLK",
    "berkshire hathaway": "BRK.B",
    # Other majors
    "walmart": "WMT",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "procter & gamble": "PG",
    "coca-cola": "KO",
    "disney": "DIS",
    "nike": "NKE",
    # Semiconductors
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
}

# Key people patterns
PERSON_PATTERNS = [
    # Central bankers
    (r"fed chair(?:man)?\s+([\w\s]+)", "central_banker"),
    (r"powell", "central_banker"),
    (r"yellen", "government_official"),
    (r"lagarde", "central_banker"),
    (r"kuroda", "central_banker"),
    (r"bailey", "central_banker"),
    # CEO patterns
    (r"ceo\s+([\w\s]+)", "executive"),
    (r"([\w\s]+),?\s+ceo\s+of", "executive"),
    # Analysts
    (r"analyst\s+([\w\s]+)\s+at", "analyst"),
]

# Topic keywords for classification
TOPIC_KEYWORDS = {
    TopicCategory.MACRO_RATES: [
        "interest rate",
        "fed",
        "fomc",
        "rate hike",
        "rate cut",
        "monetary policy",
        "treasury yield",
        "bond yield",
        "rate decision",
    ],
    TopicCategory.MACRO_INFLATION: [
        "inflation",
        "cpi",
        "pce",
        "price pressure",
        "deflation",
        "stagflation",
    ],
    TopicCategory.EARNINGS: [
        "earnings",
        "eps",
        "revenue",
        "quarterly",
        "guidance",
        "profit",
        "beat",
        "miss",
        "outlook",
        "results",
    ],
    TopicCategory.M_AND_A: [
        "merger",
        "acquisition",
        "acquire",
        "takeover",
        "buyout",
        "deal",
        "combination",
        "spinoff",
        "divestiture",
    ],
    TopicCategory.TECH_AI: [
        "ai",
        "artificial intelligence",
        "machine learning",
        "llm",
        "gpt",
        "neural network",
        "deep learning",
        "chatgpt",
        "openai",
    ],
    TopicCategory.TECH_SEMIS: [
        "semiconductor",
        "chip",
        "gpu",
        "cpu",
        "wafer",
        "fab",
        "foundry",
        "nvidia",
        "amd",
        "intel",
        "tsmc",
    ],
    TopicCategory.COMMODITIES_OIL: [
        "oil",
        "crude",
        "brent",
        "wti",
        "opec",
        "petroleum",
        "energy",
        "natural gas",
        "lng",
        "barrel",
    ],
    TopicCategory.GEOPOLITICS: [
        "tariff",
        "sanction",
        "trade war",
        "geopolitical",
        "conflict",
        "diplomacy",
        "embargo",
        "tension",
    ],
    TopicCategory.CHINA: [
        "china",
        "chinese",
        "beijing",
        "shanghai",
        "ccp",
        "yuan",
        "renminbi",
        "prc",
    ],
    TopicCategory.CRYPTO: [
        "bitcoin",
        "crypto",
        "ethereum",
        "blockchain",
        "defi",
        "nft",
        "digital asset",
        "stablecoin",
        "binance",
        "coinbase",
    ],
}


class EntityExtractor:
    """
    Extracts entities from article text and resolves to financial identifiers.

    Uses rule-based extraction with optional LLM enhancement.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ):
        """
        Initialize entity extractor.

        Args:
            use_llm: Whether to use LLM for enhanced extraction
            llm_client: Optional LLM client (from local_agent)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

        # Build reverse lookup for tickers
        self._company_patterns = self._build_company_patterns()

    def _build_company_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Build regex patterns for company name matching"""
        patterns = []
        for company, ticker in TICKER_MAPPINGS.items():
            # Case insensitive, word boundary matching
            pattern = re.compile(rf"\b{re.escape(company)}\b", re.IGNORECASE)
            patterns.append((pattern, ticker))
        return patterns

    async def extract_entities(self, article: Article) -> list[Entity]:
        """
        Extract all entities from an article.

        Args:
            article: Article to process

        Returns:
            List of extracted entities
        """
        entities = []
        text = f"{article.title} {article.snippet or ''}"

        # Extract companies
        companies = self._extract_companies(text)
        entities.extend(companies)

        # Extract people
        people = self._extract_people(text)
        entities.extend(people)

        # Extract countries/regions
        regions = self._extract_regions(text)
        entities.extend(regions)

        # Extract explicit tickers (e.g., "$AAPL" or "(NASDAQ: AAPL)")
        tickers = self._extract_explicit_tickers(text)
        for ticker in tickers:
            if ticker not in [e.ticker for e in entities if e.ticker]:
                entities.append(
                    Entity(
                        name=ticker,
                        entity_type="company",
                        ticker=ticker,
                        confidence=1.0,
                    )
                )

        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _extract_companies(self, text: str) -> list[Entity]:
        """Extract company mentions with ticker resolution"""
        entities = []

        for pattern, ticker in self._company_patterns:
            matches = pattern.findall(text)
            if matches:
                # Use the original text match as name
                name = (
                    matches[0]
                    if isinstance(matches[0], str)
                    else pattern.pattern.replace(r"\b", "").replace("\\", "")
                )
                entities.append(
                    Entity(
                        name=name.title(),
                        entity_type="company",
                        ticker=ticker,
                        confidence=0.9,
                    )
                )

        return entities

    def _extract_people(self, text: str) -> list[Entity]:
        """Extract person mentions"""
        entities = []

        for pattern_str, person_type in PERSON_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, str) and len(match) > 2:
                    entities.append(
                        Entity(
                            name=match.strip().title(),
                            entity_type=person_type,
                            confidence=0.7,
                        )
                    )

        return entities

    def _extract_regions(self, text: str) -> list[Entity]:
        """Extract country/region mentions"""
        regions = {
            "united states": "US",
            "u.s.": "US",
            "usa": "US",
            "china": "CN",
            "japan": "JP",
            "germany": "DE",
            "uk": "GB",
            "britain": "GB",
            "france": "FR",
            "canada": "CA",
            "australia": "AU",
            "india": "IN",
            "brazil": "BR",
            "russia": "RU",
            "european union": "EU",
            "eurozone": "EU",
        }

        entities = []
        text_lower = text.lower()

        for region_name, _code in regions.items():
            if region_name in text_lower:
                entities.append(
                    Entity(
                        name=region_name.title(),
                        entity_type="country",
                        confidence=0.95,
                    )
                )

        return entities

    def _extract_explicit_tickers(self, text: str) -> list[str]:
        """Extract explicit ticker mentions like $AAPL or (NASDAQ: AAPL)"""
        tickers = []

        # $TICKER pattern
        dollar_pattern = re.compile(r"\$([A-Z]{1,5})\b")
        tickers.extend(dollar_pattern.findall(text.upper()))

        # (EXCHANGE: TICKER) pattern
        exchange_pattern = re.compile(
            r"\((?:NYSE|NASDAQ|AMEX|TSX|LSE):\s*([A-Z]{1,5})\)", re.IGNORECASE
        )
        tickers.extend(exchange_pattern.findall(text.upper()))

        return list(set(tickers))

    def classify_topics(self, article: Article) -> list[TopicCategory]:
        """
        Classify article into topic categories.

        Args:
            article: Article to classify

        Returns:
            List of applicable topic categories
        """
        topics = []
        text = f"{article.title} {article.snippet or ''}".lower()

        for topic, keywords in TOPIC_KEYWORDS.items():
            # Count keyword matches
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count >= 2:  # Require at least 2 keyword matches
                topics.append(topic)
            elif match_count == 1 and len(keywords) <= 3:
                # For smaller keyword sets, one match is enough
                topics.append(topic)

        return topics

    def get_tickers(self, entities: list[Entity]) -> list[str]:
        """Extract unique tickers from entities"""
        tickers = []
        for entity in entities:
            if entity.ticker and entity.ticker not in tickers:
                tickers.append(entity.ticker)
        return tickers

    async def process_article(self, article: Article) -> Article:
        """
        Process article to extract entities and classify topics.

        Modifies article in place and returns it.
        """
        # Extract entities
        entities = await self.extract_entities(article)
        article.entities = entities

        # Extract tickers
        article.tickers = self.get_tickers(entities)

        # Classify topics if not already set
        if not article.topics:
            article.topics = self.classify_topics(article)

        return article

    async def process_batch(self, articles: list[Article]) -> list[Article]:
        """Process multiple articles"""
        processed = []
        for article in articles:
            processed.append(await self.process_article(article))
        return processed
