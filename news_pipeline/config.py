"""
Pipeline Configuration

Centralized configuration for the multi-model news pipeline.
Supports environment variables and sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    api_key: str | None = None
    model: str = ""
    base_url: str | None = None
    timeout: int = 60
    max_retries: int = 3
    enabled: bool = True


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    Loads from environment variables with sensible defaults.
    """

    # Provider configurations
    grok: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("XAI_API_KEY"),
            model=os.getenv("GROK_MODEL", "grok-beta"),
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            timeout=int(os.getenv("GROK_TIMEOUT", "90")),
        )
    )

    gemini: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "60")),
        )
    )

    chatgpt: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("CHATGPT_MODEL", "gpt-4o"),
            timeout=int(os.getenv("CHATGPT_TIMEOUT", "60")),
        )
    )

    claude: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            timeout=int(os.getenv("CLAUDE_TIMEOUT", "60")),
        )
    )

    # Pipeline settings
    min_verification_confidence: int = field(
        default_factory=lambda: int(os.getenv("MIN_CONFIDENCE", "6"))
    )
    max_concurrent_verifications: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT", "5"))
    )
    max_stories_per_run: int = field(default_factory=lambda: int(os.getenv("MAX_STORIES", "50")))

    # Publishing
    publish_endpoint: str | None = field(default_factory=lambda: os.getenv("PUBLISH_ENDPOINT"))
    publish_api_key: str | None = field(default_factory=lambda: os.getenv("PUBLISH_API_KEY"))

    # Database
    db_path: str = field(default_factory=lambda: os.getenv("NEWS_DB_PATH", "data/news_pipeline.db"))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file: bool = field(
        default_factory=lambda: os.getenv("LOG_TO_FILE", "false").lower() == "true"
    )
    log_file_path: str = field(
        default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/pipeline.log")
    )

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls()

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.grok.enabled and not self.grok.api_key:
            errors.append("XAI_API_KEY required for Grok discovery")

        if self.gemini.enabled and not self.gemini.api_key:
            errors.append("GOOGLE_API_KEY required for Gemini verification")

        if self.chatgpt.enabled and not self.chatgpt.api_key:
            errors.append("OPENAI_API_KEY required for ChatGPT formatting")

        if self.claude.enabled and not self.claude.api_key:
            errors.append("ANTHROPIC_API_KEY required for Claude editorial")

        if not 0 <= self.min_verification_confidence <= 10:
            errors.append("MIN_CONFIDENCE must be between 0 and 10")

        return errors

    def get_enabled_providers(self) -> list[str]:
        """Get list of enabled providers."""
        providers = []
        if self.grok.enabled:
            providers.append("grok")
        if self.gemini.enabled:
            providers.append("gemini")
        if self.chatgpt.enabled:
            providers.append("chatgpt")
        if self.claude.enabled:
            providers.append("claude")
        return providers
