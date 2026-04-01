"""
Prompt Sanitizer

Removes or redacts sensitive information from prompts before
sending to external model providers.

Supports:
- PII redaction (emails, phones, names)
- Credential removal
- Position/portfolio masking
- Custom redaction rules
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from news_pipeline.security.classifier import DataSensitivity, SensitivityClassifier

logger = logging.getLogger(__name__)


class RedactionStyle(Enum):
    """How to handle sensitive data."""

    REMOVE = "remove"  # Delete entirely
    MASK = "mask"  # Replace with [REDACTED]
    GENERALIZE = "generalize"  # Replace with generic version
    HASH = "hash"  # Replace with hash (reversible locally)


@dataclass
class SanitizationResult:
    """Result of sanitizing content."""

    original: str
    sanitized: str
    was_modified: bool
    redactions: list[dict] = field(default_factory=list)
    sensitivity_tier: DataSensitivity = DataSensitivity.TIER_A

    @property
    def redaction_count(self) -> int:
        """Number of redactions made."""
        return len(self.redactions)

    def get_redaction_map(self) -> dict[str, str]:
        """Get mapping of placeholders to original values."""
        return {r["placeholder"]: r["original"] for r in self.redactions if "placeholder" in r}


class Sanitizer:
    """
    Sanitizes content for external transmission.

    Removes or masks sensitive information while preserving
    the semantic meaning needed for AI processing.
    """

    # Redaction patterns with replacement strategies
    SANITIZATION_RULES = {
        # Tier C - Must redact
        "api_key": {
            "pattern": r"(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*['\"]?([\w\-]{20,})",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
        },
        "password": {
            "pattern": r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
        },
        "bearer_token": {
            "pattern": r"(Bearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
        },
        "aws_key": {
            "pattern": r"((?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16})",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
        },
        "private_key": {
            "pattern": r"(-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.+?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----)",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
            "flags": re.DOTALL,
        },
        "ssn": {
            "pattern": r"\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b",
            "style": RedactionStyle.MASK,
            "replacement": "[SSN-REDACTED]",
        },
        "credit_card": {
            "pattern": r"\b((?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}))\b",
            "style": RedactionStyle.MASK,
            "replacement": "[CARD-REDACTED]",
        },
        "account_number": {
            "pattern": r"(?:account|acct)\s*#?\s*[:=]?\s*(\d{8,})",
            "style": RedactionStyle.MASK,
            "replacement": "[ACCOUNT-XXXX]",
        },
        "routing_number": {
            "pattern": r"(?:routing|aba)\s*#?\s*[:=]?\s*(\d{9})\b",
            "style": RedactionStyle.MASK,
            "replacement": "[ROUTING-XXXX]",
        },
        # Tier B - Sanitize for external use
        "email": {
            "pattern": r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[email]",
        },
        "phone": {
            "pattern": r"\b((?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[phone]",
        },
        "ip_address": {
            "pattern": r"\b((?:\d{1,3}\.){3}\d{1,3})\b",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[IP]",
        },
        "position_details": {
            "pattern": r"\b((?:long|short)\s+\d+\s*(?:shares?|contracts?)?\s+(?:of\s+)?[A-Z]{1,5})\b",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[position in TICKER]",
        },
        "portfolio_value": {
            "pattern": r"((?:portfolio|position|holding|balance).*?\$[\d,]+(?:\.\d{2})?)",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[portfolio value]",
        },
        "date_of_birth": {
            "pattern": r"\b(?:dob|birth(?:date)?)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "style": RedactionStyle.MASK,
            "replacement": "[DOB-REDACTED]",
        },
        "address": {
            "pattern": r"\b(\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd))\b",
            "style": RedactionStyle.GENERALIZE,
            "replacement": "[address]",
        },
        "internal_url": {
            "pattern": r"(https?://(?:internal|intranet|private|localhost|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)[^\s]+)",
            "style": RedactionStyle.REMOVE,
            "replacement": "",
        },
    }

    def __init__(
        self,
        classifier: SensitivityClassifier | None = None,
        custom_rules: dict | None = None,
        preserve_structure: bool = True,
    ):
        """
        Initialize sanitizer.

        Args:
            classifier: Sensitivity classifier instance
            custom_rules: Additional sanitization rules
            preserve_structure: Keep semantic structure when redacting
        """
        self.classifier = classifier or SensitivityClassifier()
        self.preserve_structure = preserve_structure

        # Compile rules
        self._rules = {}
        for name, rule in self.SANITIZATION_RULES.items():
            flags = rule.get("flags", re.IGNORECASE)
            self._rules[name] = {
                "pattern": re.compile(rule["pattern"], flags),
                "style": rule["style"],
                "replacement": rule["replacement"],
            }

        # Add custom rules
        if custom_rules:
            for name, rule in custom_rules.items():
                flags = rule.get("flags", re.IGNORECASE)
                self._rules[name] = {
                    "pattern": re.compile(rule["pattern"], flags),
                    "style": rule.get("style", RedactionStyle.MASK),
                    "replacement": rule.get("replacement", f"[{name.upper()}-REDACTED]"),
                }

    def sanitize(
        self,
        content: str,
        target_tier: DataSensitivity = DataSensitivity.TIER_A,
    ) -> SanitizationResult:
        """
        Sanitize content for the target sensitivity tier.

        Args:
            content: Content to sanitize
            target_tier: Target tier (sanitize to this level)

        Returns:
            SanitizationResult with sanitized content and redaction info
        """
        # First classify the content
        classification = self.classifier.classify(content)

        # If already at or below target tier, no sanitization needed
        if classification.tier == DataSensitivity.TIER_A:
            return SanitizationResult(
                original=content,
                sanitized=content,
                was_modified=False,
                sensitivity_tier=DataSensitivity.TIER_A,
            )

        # If Tier C and target is external, we can't fully sanitize
        if classification.tier == DataSensitivity.TIER_C and target_tier == DataSensitivity.TIER_A:
            logger.warning(
                "[Sanitizer] Content contains Tier C data - full sanitization may lose context"
            )

        # Apply sanitization rules
        sanitized = content
        redactions = []

        for name, rule in self._rules.items():
            pattern = rule["pattern"]
            replacement = rule["replacement"]

            for match in pattern.finditer(sanitized):
                matched_sensitive = match.group(1) if match.lastindex else match.group(0)

                # Create placeholder
                if rule["style"] == RedactionStyle.HASH:
                    import hashlib

                    placeholder = f"[{name}:{hashlib.sha256(matched_sensitive.encode()).hexdigest()[:8]}]"
                elif rule["style"] == RedactionStyle.REMOVE:
                    placeholder = ""
                else:
                    placeholder = replacement

                redactions.append(
                    {
                        "rule": name,
                        "original": matched_sensitive,
                        "placeholder": placeholder,
                        "position": match.start(),
                    }
                )

            # Apply replacements
            sanitized = pattern.sub(replacement, sanitized)

        # Clean up multiple spaces/newlines from removals
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        sanitized = re.sub(r" {2,}", " ", sanitized)
        sanitized = sanitized.strip()

        # Re-classify sanitized content
        final_classification = self.classifier.classify(sanitized)

        return SanitizationResult(
            original=content,
            sanitized=sanitized,
            was_modified=bool(redactions),
            redactions=redactions,
            sensitivity_tier=final_classification.tier,
        )

    def sanitize_prompt(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str = "",
    ) -> tuple[str, str, str]:
        """
        Sanitize a complete prompt with context.

        Args:
            prompt: Main prompt
            context: Additional context
            system_prompt: System prompt

        Returns:
            Tuple of (sanitized_prompt, sanitized_context, sanitized_system)
        """
        results = [
            self.sanitize(prompt),
            self.sanitize(context) if context else SanitizationResult("", "", False),
            self.sanitize(system_prompt) if system_prompt else SanitizationResult("", "", False),
        ]

        return (
            results[0].sanitized,
            results[1].sanitized,
            results[2].sanitized,
        )

    def sanitize_news_item(self, headline: str, snippet: str = "") -> tuple[str, str]:
        """
        Sanitize a news item.

        News items are typically public, but may have user annotations.

        Args:
            headline: News headline
            snippet: Optional snippet

        Returns:
            Tuple of (sanitized_headline, sanitized_snippet)
        """
        h_result = self.sanitize(headline)
        s_result = self.sanitize(snippet) if snippet else SanitizationResult("", "", False)

        return h_result.sanitized, s_result.sanitized


# Module-level convenience functions
_default_sanitizer: Sanitizer | None = None


def get_sanitizer() -> Sanitizer:
    """Get the default sanitizer instance."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = Sanitizer()
    return _default_sanitizer


def sanitize_prompt(prompt: str, target_tier: DataSensitivity = DataSensitivity.TIER_A) -> SanitizationResult:
    """Sanitize a prompt using the default sanitizer."""
    return get_sanitizer().sanitize(prompt, target_tier)
