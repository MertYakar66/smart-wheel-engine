"""
Event Calendar for Macro + SP500 Event Intelligence System

Provides:
- 2026 FOMC meeting dates
- BLS release schedule (CPI, NFP, JOLTS)
- BEA release schedule (GDP, PCE)
- EIA petroleum report schedule
- Event-aware run scheduling
"""

from .macro_calendar import (
    MacroCalendar,
    get_cpi_2026,
    get_eia_petroleum_schedule,
    get_fomc_2026,
    get_gdp_2026,
    get_nfp_2026,
    get_pce_2026,
)

__all__ = [
    "MacroCalendar",
    "get_fomc_2026",
    "get_cpi_2026",
    "get_nfp_2026",
    "get_gdp_2026",
    "get_pce_2026",
    "get_eia_petroleum_schedule",
]
