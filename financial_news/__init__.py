"""
Macro + SP500 Event Intelligence System

A scheduled macro and SP500 event intelligence platform:
- Official sources first (Fed, BLS, BEA, SEC, EIA, OPEC, Treasury)
- 8 focused categories (Fed, Inflation, Labor, Growth, Oil, Geopolitics, SP500, Regime)
- Event calendar with 2026 macro release schedules
- AM/PM briefs with event-aware mini-runs
- Story clustering and ranking by macro/SP500 relevance

Architecture:
1. schema.py - Canonical data models
2. storage/database.py - SQLite implementation
3. calendar/macro_calendar.py - 2026 event schedules
4. connectors/ - Official source connectors (Fed, BLS, BEA, SEC, EIA)
5. processing/ - Classification, clustering, ranking
6. scheduler.py - AM/PM runs + event triggers
"""

__version__ = "2.0.0"

# Core schema
from .schema import (
    # Enums
    SourceType,
    SourceProvider,
    CategoryType,
    EventType,
    EntityType,
    ImportanceLevel,
    BriefType,
    RunStatus,
    # Data models
    Source,
    Category,
    CategoryRule,
    ScheduledEvent,
    Entity,
    Article,
    Story,
    Brief,
    RunLog,
    UserWatchlist,
    # Defaults
    DEFAULT_SOURCES,
    DEFAULT_CATEGORIES,
    DEFAULT_CATEGORY_RULES,
)

# Storage
from .storage import NewsDatabase

# Calendar
from .calendar import MacroCalendar

__all__ = [
    # Enums
    "SourceType",
    "SourceProvider",
    "CategoryType",
    "EventType",
    "EntityType",
    "ImportanceLevel",
    "BriefType",
    "RunStatus",
    # Models
    "Source",
    "Category",
    "CategoryRule",
    "ScheduledEvent",
    "Entity",
    "Article",
    "Story",
    "Brief",
    "RunLog",
    "UserWatchlist",
    # Defaults
    "DEFAULT_SOURCES",
    "DEFAULT_CATEGORIES",
    "DEFAULT_CATEGORY_RULES",
    # Components
    "NewsDatabase",
    "MacroCalendar",
]
