"""
Security Module

Data sensitivity classification, prompt sanitization, and routing policies.

Tier System:
- Tier A: Public data (RSS headlines, public prices)
- Tier B: Sanitizable data (user queries, portfolio hints)
- Tier C: Private data (credentials, PII, internal notes)
"""

from news_pipeline.security.classifier import (
    DataSensitivity,
    SensitivityClassifier,
    classify_content,
)
from news_pipeline.security.routing_policy import (
    RoutingDecision,
    RoutingPolicy,
    get_routing_policy,
)
from news_pipeline.security.sanitizer import (
    SanitizationResult,
    Sanitizer,
    sanitize_prompt,
)

__all__ = [
    "DataSensitivity",
    "SensitivityClassifier",
    "classify_content",
    "Sanitizer",
    "SanitizationResult",
    "sanitize_prompt",
    "RoutingPolicy",
    "RoutingDecision",
    "get_routing_policy",
]
