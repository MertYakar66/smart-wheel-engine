"""
Browser Agent Type Definitions

Enums and dataclasses that don't require playwright.
These can be imported without the playwright dependency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ModelType(Enum):
    """Supported model types."""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    LOCAL = "local"


class SessionStatus(Enum):
    """Browser session status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATED = "authenticated"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNINITIALIZED = "uninitialized"


@dataclass
class ModelResponse:
    """Response from a model session."""

    success: bool
    content: str
    model: str
    latency_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
