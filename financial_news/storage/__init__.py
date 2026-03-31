"""
Storage Layer for Macro + SP500 Event Intelligence System

Provides:
- NewsDatabase: Primary storage with canonical schema
  - Sources with trust tiers
  - 8 core categories with classification rules
  - Event calendar for macro releases
  - Articles, stories, briefs
  - Run logging

- NewsStore: Legacy storage (deprecated, use NewsDatabase)
"""

from .database import NewsDatabase
from .news_store import NewsStore

__all__ = ["NewsDatabase", "NewsStore"]
