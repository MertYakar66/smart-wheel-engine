"""
Storage Layer for Financial News Platform

Provides persistent storage for:
- Articles (metadata only, no full text for licensed content)
- Stories (clustered news)
- Categories (saved queries)
- User profiles
"""

from .news_store import NewsStore

__all__ = ["NewsStore"]
