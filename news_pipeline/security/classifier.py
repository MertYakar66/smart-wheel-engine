"""
Data Sensitivity Classifier

Classifies content into sensitivity tiers:
- Tier A: Public data, safe to send anywhere
- Tier B: Contains sanitizable private info
- Tier C: Contains private data, never send externally
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DataSensitivity(Enum):
    """Data sensitivity tiers."""

    TIER_A = "public"  # Safe to send to any provider
    TIER_B = "sanitizable"  # Can be sanitized before sending
    TIER_C = "private"  # Never send externally


@dataclass
class ClassificationResult:
    """Result of sensitivity classification."""

    tier: DataSensitivity
    confidence: float  # 0-1
    detected_patterns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_safe_external(self) -> bool:
        """Check if content is safe to send externally."""
        return self.tier == DataSensitivity.TIER_A

    @property
    def requires_sanitization(self) -> bool:
        """Check if content needs sanitization before external use."""
        return self.tier == DataSensitivity.TIER_B

    @property
    def is_private(self) -> bool:
        """Check if content must stay local."""
        return self.tier == DataSensitivity.TIER_C


class SensitivityClassifier:
    """
    Classifies content sensitivity for routing decisions.

    Uses pattern matching and heuristics to detect:
    - PII (emails, phone numbers, SSNs)
    - Credentials (API keys, passwords, tokens)
    - Financial data (account numbers, positions)
    - Internal references (internal URLs, employee IDs)
    """

    # Tier C patterns - never send externally
    TIER_C_PATTERNS = {
        "api_key": r"(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*['\"]?[\w\-]{20,}",
        "password": r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        "bearer_token": r"Bearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+",
        "aws_key": r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}",
        "private_key": r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
        "internal_url": r"https?://(?:internal|intranet|private|localhost|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)[^\s]+",
        "account_number": r"(?:account|acct)\s*#?\s*[:=]?\s*\d{8,}",
        "routing_number": r"(?:routing|aba)\s*#?\s*[:=]?\s*\d{9}\b",
    }

    # Tier B patterns - can be sanitized
    TIER_B_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ticker_position": r"\b(long|short|position|shares|holding)\s+\d+\s*(?:shares?|contracts?)?\s+(?:of\s+)?[A-Z]{1,5}\b",
        "portfolio_value": r"(?:portfolio|position|holding|balance).*?\$[\d,]+(?:\.\d{2})?",
        "employee_id": r"(?:employee|emp|staff)\s*(?:id|#|number)?\s*[:=]?\s*[A-Z0-9]{5,}",
        "user_name": r"(?:user(?:name)?|login)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{3,}",
        "date_of_birth": r"\b(?:dob|birth(?:date)?)\s*[:=]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        "address": r"\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd)\b",
    }

    # Tier A keywords - public financial news
    TIER_A_INDICATORS = {
        "fed",
        "fomc",
        "powell",
        "yellen",
        "treasury",
        "inflation",
        "cpi",
        "gdp",
        "unemployment",
        "earnings",
        "revenue",
        "quarterly",
        "guidance",
        "forecast",
        "analyst",
        "upgrade",
        "downgrade",
        "target price",
        "market cap",
        "ipo",
        "merger",
        "acquisition",
        "sec filing",
        "10-k",
        "10-q",
        "8-k",
        "breaking",
        "report",
        "reuters",
        "bloomberg",
        "cnbc",
        "wsj",
    }

    def __init__(self, strict_mode: bool = True):
        """
        Initialize classifier.

        Args:
            strict_mode: If True, err on side of caution (higher tier when uncertain)
        """
        self.strict_mode = strict_mode
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._tier_c_compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.TIER_C_PATTERNS.items()
        }
        self._tier_b_compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.TIER_B_PATTERNS.items()
        }

    def classify(self, content: str) -> ClassificationResult:
        """
        Classify content sensitivity.

        Args:
            content: Text content to classify

        Returns:
            ClassificationResult with tier and detected patterns
        """
        detected = []
        recommendations = []

        # Check Tier C patterns first (most sensitive)
        for name, pattern in self._tier_c_compiled.items():
            if pattern.search(content):
                detected.append(f"tier_c:{name}")

        if detected:
            return ClassificationResult(
                tier=DataSensitivity.TIER_C,
                confidence=0.95,
                detected_patterns=detected,
                recommendations=["Process locally only", "Never send to external providers"],
            )

        # Check Tier B patterns
        for name, pattern in self._tier_b_compiled.items():
            if pattern.search(content):
                detected.append(f"tier_b:{name}")
                recommendations.append(f"Sanitize {name} before external use")

        if detected:
            return ClassificationResult(
                tier=DataSensitivity.TIER_B,
                confidence=0.85,
                detected_patterns=detected,
                recommendations=recommendations,
            )

        # Check for public news indicators
        content_lower = content.lower()
        public_indicators = sum(1 for kw in self.TIER_A_INDICATORS if kw in content_lower)

        if public_indicators >= 2:
            return ClassificationResult(
                tier=DataSensitivity.TIER_A,
                confidence=0.9,
                detected_patterns=[f"public_indicators:{public_indicators}"],
                recommendations=["Safe for external providers"],
            )

        # Default handling
        if self.strict_mode:
            # In strict mode, unknown content is Tier B
            return ClassificationResult(
                tier=DataSensitivity.TIER_B,
                confidence=0.5,
                detected_patterns=["unknown_content"],
                recommendations=["Review content manually", "Consider sanitization"],
            )
        else:
            return ClassificationResult(
                tier=DataSensitivity.TIER_A,
                confidence=0.6,
                detected_patterns=["assumed_public"],
                recommendations=["Low confidence classification"],
            )

    def classify_news_item(
        self, headline: str, snippet: str = "", source: str = ""
    ) -> ClassificationResult:
        """
        Classify a news item for processing.

        News items are typically Tier A (public), but may contain
        user-added annotations that need sanitization.

        Args:
            headline: News headline
            snippet: Optional snippet/summary
            source: Source name

        Returns:
            ClassificationResult
        """
        combined = f"{headline}\n{snippet}\n{source}"
        result = self.classify(combined)

        # News from known sources is more likely public
        known_sources = {"reuters", "bloomberg", "cnbc", "wsj", "fed", "sec", "treasury"}
        if any(src in source.lower() for src in known_sources):
            if result.tier == DataSensitivity.TIER_B and result.confidence < 0.7:
                # Downgrade to Tier A if from known source with low confidence
                return ClassificationResult(
                    tier=DataSensitivity.TIER_A,
                    confidence=0.8,
                    detected_patterns=result.detected_patterns + ["known_source_override"],
                    recommendations=["From trusted public source"],
                )

        return result

    def classify_prompt(self, prompt: str, context: str = "") -> ClassificationResult:
        """
        Classify a prompt before sending to external model.

        Prompts may contain user context that needs sanitization.

        Args:
            prompt: The prompt text
            context: Additional context

        Returns:
            ClassificationResult
        """
        combined = f"{prompt}\n{context}"
        return self.classify(combined)


# Module-level convenience functions
_default_classifier: SensitivityClassifier | None = None


def get_classifier() -> SensitivityClassifier:
    """Get the default classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SensitivityClassifier()
    return _default_classifier


def classify_content(content: str) -> ClassificationResult:
    """Classify content using default classifier."""
    return get_classifier().classify(content)
