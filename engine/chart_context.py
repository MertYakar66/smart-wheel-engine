"""
Chart context primitives for the TradingView visual layer.

Architectural role
------------------
TradingView is a *supplemental visual source*, not a data backend and not
a decision engine. The Smart Wheel Engine's hierarchy is:

    1. Engine ranks candidates         (WheelRunner.rank_candidates_by_ev)
    2. TradingView screenshot is gathered for the TOP candidates only
    3. Chart context is attached to the candidate dossier
    4. A chart reviewer may DOWNGRADE a trade based on visual context
       (e.g. "setup is broken — recent support violated")
    5. A chart reviewer MAY NOT UPGRADE a negative-EV trade. Ever.

This module defines the pure-data primitives used throughout that
pipeline. The actual browser automation and screenshot capture lives
in ``engine/tradingview_bridge.py``.

Hard constraints (from the TradingView strategic review)
--------------------------------------------------------
* The screenshot is a *visual* artifact. We deliberately do NOT parse
  numbers out of it. Numbers come from the engine's structured data
  path, never from the chart.
* A ChartContext has at most three things: the ticker, the timeframe,
  and a file path to the image. Everything else is optional metadata
  that must not be treated as authoritative.
* If chart context retrieval fails, the dossier layer must degrade
  gracefully — the engine's ranking stands on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol


Timeframe = Literal[
    "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D", "1W", "1M"
]


@dataclass
class ChartContext:
    """A single TradingView chart context artifact.

    The fields are split into three tiers by level of trust:

    * Authoritative (always safe to use downstream):
        - ``ticker``, ``timeframe``, ``captured_at``
        - ``screenshot_path`` — the image file itself

    * Supplemental (may be used by the chart reviewer, never as a
      replacement for structured numeric data):
        - ``visible_price`` — price shown on the chart at capture time,
          for cross-checking against the engine's structured spot price
        - ``visible_indicators`` — optional dict of values the browser
          automation happened to scrape from Pine indicator labels

    * Diagnostic (purely informational):
        - ``source`` — which provider produced this context
        - ``browser_url`` — canonical TradingView URL
        - ``notes`` — freeform text, e.g. "captured after retry"
    """

    # Authoritative
    ticker: str
    timeframe: Timeframe
    captured_at: datetime
    screenshot_path: Path | None = None

    # Supplemental
    visible_price: float | None = None
    visible_indicators: dict[str, float] = field(default_factory=dict)

    # Diagnostic
    source: str = "unknown"
    browser_url: str = ""
    notes: str = ""

    # Failure signal: when the bridge could not retrieve a chart context
    # the ``error`` field is populated. Dossier consumers must check
    # this first and treat a missing chart as a degraded — but still
    # tradeable — candidate.
    error: str = ""

    def is_ok(self) -> bool:
        return not self.error and self.screenshot_path is not None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "captured_at": self.captured_at.isoformat(),
            "screenshot_path": (
                str(self.screenshot_path) if self.screenshot_path else None
            ),
            "visible_price": self.visible_price,
            "visible_indicators": dict(self.visible_indicators),
            "source": self.source,
            "browser_url": self.browser_url,
            "notes": self.notes,
            "error": self.error,
            "ok": self.is_ok(),
        }


class ChartContextProvider(Protocol):
    """Protocol for any backend that can produce a ChartContext.

    Implementations include:

    * :class:`engine.tradingview_bridge.FilesystemChartProvider` — reads
      an existing screenshot off disk. Used for tests and for
      externally-captured workflows (e.g. a cronjob that runs Chromium
      and drops screenshots into a well-known directory).

    * :class:`engine.tradingview_bridge.PlaywrightChartProvider` — drives
      a headless Chromium session via Playwright, navigates to the TV
      chart URL, and captures a PNG. Lazy-imports playwright so the
      engine has no hard dependency on it; importing this provider when
      playwright is not installed returns a clearly-flagged error
      ChartContext rather than crashing the caller.
    """

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1D",
        *,
        as_of: datetime | None = None,
    ) -> ChartContext:
        ...
