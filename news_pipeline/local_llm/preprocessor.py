"""
Local LLM Preprocessor

Uses local LLM (Ollama) for cheap preprocessing tasks.
Falls back to rule-based processing if no LLM available.

Tasks:
- Filter irrelevant/low-quality news
- Categorize headlines
- Extract and validate tickers
- Basic deduplication

NOT for:
- Verification (use Claude with search)
- Analysis (use Claude/ChatGPT)
- Editorial (use Claude)
"""

import json
import logging
from dataclasses import dataclass

import aiohttp

from news_pipeline.scrapers.base import NewsCategory, NewsItem

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing a news item."""

    keep: bool  # Whether to keep for further processing
    category: NewsCategory
    tickers: list[str]
    relevance_score: float  # 0-1
    reason: str = ""


class LocalPreprocessor:
    """
    Local LLM-based preprocessor.

    Uses Ollama for lightweight processing.
    Falls back to rules if Ollama unavailable.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",  # Good balance of speed/quality
        use_llm: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            ollama_url: Ollama API endpoint
            model: Model to use
            use_llm: Whether to use LLM (False = rules only)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.use_llm = use_llm
        self._llm_available = False

    async def check_llm_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        self._llm_available = any(self.model.split(":")[0] in m for m in models)
                        return self._llm_available
        except Exception as e:
            logger.debug(f"[LocalLLM] Ollama not available: {e}")

        self._llm_available = False
        return False

    async def preprocess_batch(
        self,
        items: list[NewsItem],
        filter_threshold: float = 0.3,
    ) -> list[NewsItem]:
        """
        Preprocess a batch of news items.

        Args:
            items: Raw news items
            filter_threshold: Minimum relevance score to keep

        Returns:
            Filtered and enriched items
        """
        if self.use_llm:
            await self.check_llm_available()

        results = []

        for item in items:
            if self._llm_available:
                result = await self._preprocess_with_llm(item)
            else:
                result = self._preprocess_with_rules(item)

            if result.keep and result.relevance_score >= filter_threshold:
                # Update item with preprocessing results
                item.category = result.category
                item.tickers = result.tickers
                results.append(item)

        logger.info(
            f"[Preprocessor] Kept {len(results)}/{len(items)} items (LLM: {self._llm_available})"
        )

        return results

    async def _preprocess_with_llm(self, item: NewsItem) -> PreprocessingResult:
        """Preprocess using local LLM."""
        prompt = f"""Analyze this financial news headline for trading relevance.

HEADLINE: "{item.headline}"
SOURCE: {item.source_name}

Return JSON only:
{{
    "relevant": true/false,
    "category": "fed/earnings/oil/geopolitics/macro/crypto/tech/other",
    "tickers": ["AAPL", "NVDA"],
    "relevance_score": 0.0-1.0,
    "reason": "brief explanation"
}}

Rules:
- relevant=true if it could move markets
- Extract valid US stock tickers only
- Higher score for Fed, earnings, macro events"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 200,
                        },
                    },
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_llm_response(data.get("response", ""))

        except Exception as e:
            logger.debug(f"[LocalLLM] Request failed: {e}")

        # Fall back to rules
        return self._preprocess_with_rules(item)

    def _parse_llm_response(self, response: str) -> PreprocessingResult:
        """Parse LLM response into PreprocessingResult."""
        try:
            # Try to extract JSON
            response = response.strip()

            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            data = json.loads(response)

            # Map category string to enum
            category_map = {
                "fed": NewsCategory.FED,
                "earnings": NewsCategory.EARNINGS,
                "oil": NewsCategory.OIL,
                "geopolitics": NewsCategory.GEOPOLITICS,
                "macro": NewsCategory.MACRO,
                "crypto": NewsCategory.CRYPTO,
                "tech": NewsCategory.TECH,
            }

            return PreprocessingResult(
                keep=data.get("relevant", True),
                category=category_map.get(
                    data.get("category", "other").lower(),
                    NewsCategory.OTHER,
                ),
                tickers=data.get("tickers", []),
                relevance_score=float(data.get("relevance_score", 0.5)),
                reason=data.get("reason", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"[LocalLLM] Parse error: {e}")
            return PreprocessingResult(
                keep=True,
                category=NewsCategory.OTHER,
                tickers=[],
                relevance_score=0.5,
                reason="LLM parse failed",
            )

    def _preprocess_with_rules(self, item: NewsItem) -> PreprocessingResult:
        """Preprocess using rule-based logic (fallback)."""
        headline_lower = item.headline.lower()

        # Check relevance
        irrelevant_keywords = [
            "horoscope",
            "celebrity",
            "sports",
            "weather",
            "entertainment",
            "lifestyle",
            "recipe",
        ]
        if any(kw in headline_lower for kw in irrelevant_keywords):
            return PreprocessingResult(
                keep=False,
                category=NewsCategory.OTHER,
                tickers=[],
                relevance_score=0.0,
                reason="Irrelevant content",
            )

        # Categorize
        category = self._categorize_headline(headline_lower)

        # Extract tickers
        tickers = self._extract_tickers(item.headline)

        # Calculate relevance score
        relevance = 0.5
        if category in (NewsCategory.FED, NewsCategory.EARNINGS, NewsCategory.MACRO):
            relevance = 0.8
        if tickers:
            relevance += 0.1
        if item.source_type.value == "official":
            relevance += 0.1

        return PreprocessingResult(
            keep=True,
            category=category,
            tickers=tickers,
            relevance_score=min(1.0, relevance),
            reason="Rule-based",
        )

    def _categorize_headline(self, headline: str) -> NewsCategory:
        """Categorize headline using keywords."""
        if any(kw in headline for kw in ["fed", "fomc", "powell", "rate"]):
            return NewsCategory.FED
        if any(kw in headline for kw in ["earnings", "revenue", "profit", "eps"]):
            return NewsCategory.EARNINGS
        if any(kw in headline for kw in ["oil", "crude", "opec", "energy"]):
            return NewsCategory.OIL
        if any(kw in headline for kw in ["china", "russia", "war", "sanctions"]):
            return NewsCategory.GEOPOLITICS
        if any(kw in headline for kw in ["gdp", "jobs", "inflation", "cpi"]):
            return NewsCategory.MACRO
        if any(kw in headline for kw in ["bitcoin", "crypto", "ethereum"]):
            return NewsCategory.CRYPTO
        return NewsCategory.OTHER

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock tickers from text."""
        import re

        exclude = {
            "CEO",
            "CFO",
            "IPO",
            "ETF",
            "NYSE",
            "SEC",
            "FED",
            "GDP",
            "CPI",
            "PPI",
            "USA",
            "FDA",
            "THE",
            "AND",
        }

        pattern = r"\b([A-Z]{2,5})\b"
        matches = re.findall(pattern, text)

        return [m for m in matches if m not in exclude][:5]
