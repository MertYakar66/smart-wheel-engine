"""
Macro + SP500 Event Intelligence System

A scheduled macro and SP500 event intelligence platform:
- Official sources first (Fed, BLS, BEA, SEC, EIA, OPEC, Treasury)
- 8 focused categories (Fed, Inflation, Labor, Growth, Oil, Geopolitics, SP500, Regime)
- Event calendar with 2026 macro release schedules
- AM/PM briefs with event-aware mini-runs
- Story clustering and ranking by macro/SP500 relevance
- Verification engine using Claude Code's WebSearch

Architecture:
1. schema.py - Canonical data models
2. storage/database.py - SQLite implementation
3. calendar/macro_calendar.py - 2026 event schedules
4. connectors/ - Official source connectors (Fed, BLS, BEA, SEC, EIA)
5. processing/ - Classification, clustering, ranking
6. scheduler.py - AM/PM runs + event triggers
7. verification_engine.py - Claude Code verification layer
"""

__version__ = "2.1.0"

# Core schema
# Calendar
from .calendar import MacroCalendar
from .schema import (
    DEFAULT_CATEGORIES,
    DEFAULT_CATEGORY_RULES,
    # Defaults
    DEFAULT_SOURCES,
    Article,
    Brief,
    BriefType,
    Category,
    CategoryRule,
    CategoryType,
    Entity,
    EntityType,
    EventType,
    ImportanceLevel,
    RunLog,
    RunStatus,
    ScheduledEvent,
    # Data models
    Source,
    SourceProvider,
    # Enums
    SourceType,
    Story,
    UserWatchlist,
)

# Storage
from .storage import NewsDatabase

# Verification Engine
from .verification_engine import (
    PushPayload,
    VerificationCandidate,
    VerificationEngine,
    VerificationResult,
    VerificationStatus,
    print_stats,
    print_verification_queries,
    process_verification_response,
    push_verified_stories,
    run_verification_cycle,
)

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
    # Verification
    "VerificationEngine",
    "VerificationCandidate",
    "VerificationResult",
    "VerificationStatus",
    "PushPayload",
    "run_verification_cycle",
    "process_verification_response",
    "push_verified_stories",
    "print_verification_queries",
    "print_stats",
]
