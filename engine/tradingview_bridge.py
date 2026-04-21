"""
Thin TradingView bridge for screenshot / chart-context capture.

Design constraints (from the TradingView strategic review)
----------------------------------------------------------
* **Thin.** The bridge captures a URL + screenshot and stops there.
  It does not parse numbers out of the DOM. It does not scrape
  indicator values except what the caller explicitly requests as
  *supplemental* metadata.
* **Pluggable.** The bridge is an interface, not a concrete
  implementation. We ship two providers:
    - :class:`FilesystemChartProvider` — zero dependencies, reads a
      pre-captured screenshot off disk. This is the canonical
      implementation for Claude-terminal-driven workflows where the
      terminal captures the screenshot via its own browser tooling
      and drops it into a well-known directory.
    - :class:`PlaywrightChartProvider` — drives a headless Chromium
      via Playwright. Used when the engine has its own automation.
      Lazy-imports Playwright so the engine picks up no hard dep.
* **Graceful degradation.** If the bridge cannot produce a chart
  context for any reason (missing file, Playwright not installed,
  network timeout), it returns a :class:`~engine.chart_context.ChartContext`
  with the ``error`` field populated. The dossier layer treats that
  as a degraded-but-tradeable candidate.

Canonical TradingView URL format
--------------------------------
We hit ``https://www.tradingview.com/chart/?symbol=<EXCHANGE>:<TICKER>&interval=<tf>``.
The exchange prefix defaults to NASDAQ for tech and NYSE for
everything else — this is a rough heuristic and can be overridden per
ticker via the ``exchange_map`` constructor arg. TradingView will
resolve the symbol on its side so a wrong exchange prefix does not
crash the capture, it just shows a fallback chart.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from .chart_context import ChartContext, ChartContextProvider, Timeframe

logger = logging.getLogger(__name__)


# Rough heuristic for exchange prefix. Users can override via
# ``exchange_map`` on either provider. Defaults to NASDAQ for tech
# mega-caps and NYSE for the rest of the SP500. This is ONLY used to
# build canonical TradingView URLs — the engine's own data path is
# exchange-agnostic.
_DEFAULT_EXCHANGE_MAP = {
    "AAPL": "NASDAQ",
    "MSFT": "NASDAQ",
    "GOOGL": "NASDAQ",
    "GOOG": "NASDAQ",
    "AMZN": "NASDAQ",
    "META": "NASDAQ",
    "NVDA": "NASDAQ",
    "TSLA": "NASDAQ",
    "AVGO": "NASDAQ",
    "ADBE": "NASDAQ",
    "NFLX": "NASDAQ",
    "PYPL": "NASDAQ",
    "INTC": "NASDAQ",
    "AMD": "NASDAQ",
    "QCOM": "NASDAQ",
    "MU": "NASDAQ",
    "CSCO": "NASDAQ",
    "CMCSA": "NASDAQ",
    "SPY": "AMEX",
    "QQQ": "NASDAQ",
    "IWM": "AMEX",
}

_DEFAULT_TF_INTERVAL = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1D": "D",
    "1W": "W",
    "1M": "M",
}


def build_tradingview_url(
    ticker: str,
    timeframe: Timeframe = "1D",
    exchange_map: dict[str, str] | None = None,
) -> str:
    """Return the canonical TradingView chart URL for ``ticker``.

    >>> build_tradingview_url("AAPL", "1D")
    'https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL&interval=D'
    """
    em = {**_DEFAULT_EXCHANGE_MAP, **(exchange_map or {})}
    exchange = em.get(ticker.upper(), "NYSE")
    interval = _DEFAULT_TF_INTERVAL.get(timeframe, "D")
    symbol = f"{exchange}:{ticker.upper()}"
    # url-encode the colon
    return (
        f"https://www.tradingview.com/chart/?symbol={exchange}%3A{ticker.upper()}"
        f"&interval={interval}"
    )


# ---------------------------------------------------------------------
# 1. Filesystem provider (canonical for Claude-terminal workflows)
# ---------------------------------------------------------------------
class FilesystemChartProvider:
    """Chart provider that reads externally-captured screenshots off disk.

    This is the canonical provider for workflows where the terminal /
    agent captures TradingView screenshots via its own browser tooling
    and drops them into a well-known directory. The directory layout:

        <base_dir>/<TICKER>/<TIMEFRAME>.png

    E.g.::

        screenshots/
          AAPL/
            1D.png
            4h.png
          MSFT/
            1D.png

    When the caller requests a ticker/timeframe that does not exist on
    disk, the provider returns a ChartContext with
    ``error="screenshot_not_found"`` and ``screenshot_path=None``.

    The filesystem provider never crashes; it always returns a
    well-formed ChartContext.
    """

    def __init__(
        self,
        base_dir: Path | str,
        exchange_map: dict[str, str] | None = None,
        staleness_seconds: int = 24 * 3600,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.exchange_map = exchange_map
        self.staleness_seconds = staleness_seconds

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1D",
        *,
        as_of: datetime | None = None,
    ) -> ChartContext:
        now = as_of or datetime.utcnow()
        url = build_tradingview_url(ticker, timeframe, self.exchange_map)

        path = self.base_dir / ticker.upper() / f"{timeframe}.png"
        if not path.exists():
            return ChartContext(
                ticker=ticker.upper(),
                timeframe=timeframe,
                captured_at=now,
                screenshot_path=None,
                source="filesystem",
                browser_url=url,
                error="screenshot_not_found",
                notes=f"Expected at {path}",
            )

        # Staleness check: if the file mtime is older than
        # ``staleness_seconds`` the provider flags it with an error so
        # the dossier layer can decide whether to reject or accept.
        mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
        age_seconds = (now - mtime).total_seconds()
        if age_seconds > self.staleness_seconds:
            return ChartContext(
                ticker=ticker.upper(),
                timeframe=timeframe,
                captured_at=now,
                screenshot_path=path,
                source="filesystem",
                browser_url=url,
                error=f"stale_screenshot_{int(age_seconds)}s",
                notes=f"File mtime {mtime.isoformat()} > {self.staleness_seconds}s ago",
            )

        return ChartContext(
            ticker=ticker.upper(),
            timeframe=timeframe,
            captured_at=mtime,
            screenshot_path=path,
            source="filesystem",
            browser_url=url,
            notes="",
        )


# ---------------------------------------------------------------------
# 2. Playwright provider (optional heavy dep, lazy-imported)
# ---------------------------------------------------------------------
class PlaywrightChartProvider:
    """Chart provider that drives a headless Chromium via Playwright.

    Lazy-imports Playwright on the first call to :meth:`fetch`. If
    Playwright is not installed, every call returns a
    ``ChartContext(error="playwright_not_installed")`` so callers can
    fall back to the filesystem provider without a crash.

    Usage::

        provider = PlaywrightChartProvider(
            output_dir="screenshots/",
            viewport=(1280, 800),
            wait_seconds=5,
        )
        ctx = provider.fetch("AAPL", "1D")
    """

    def __init__(
        self,
        output_dir: Path | str,
        exchange_map: dict[str, str] | None = None,
        viewport: tuple[int, int] = (1280, 800),
        wait_seconds: float = 5.0,
        headless: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_map = exchange_map
        self.viewport = viewport
        self.wait_seconds = wait_seconds
        self.headless = headless
        self._pw_available: bool | None = None

    def _check_playwright(self) -> bool:
        if self._pw_available is not None:
            return self._pw_available
        try:
            import playwright.sync_api  # noqa: F401

            self._pw_available = True
        except ImportError:
            self._pw_available = False
        return self._pw_available

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1D",
        *,
        as_of: datetime | None = None,
    ) -> ChartContext:
        now = as_of or datetime.utcnow()
        url = build_tradingview_url(ticker, timeframe, self.exchange_map)

        if not self._check_playwright():
            return ChartContext(
                ticker=ticker.upper(),
                timeframe=timeframe,
                captured_at=now,
                screenshot_path=None,
                source="playwright",
                browser_url=url,
                error="playwright_not_installed",
                notes="pip install playwright && playwright install chromium",
            )

        # Hash URL for a stable filename in the output dir.
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        out_path = self.output_dir / f"{ticker.upper()}_{timeframe}_{digest}.png"

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=self.headless)
                page = browser.new_page(
                    viewport={"width": self.viewport[0], "height": self.viewport[1]}
                )
                page.goto(url, wait_until="networkidle")
                # TradingView lazy-loads a lot of DOM; give it time.
                page.wait_for_timeout(int(self.wait_seconds * 1000))
                page.screenshot(path=str(out_path), full_page=False)
                browser.close()
        except Exception as exc:  # noqa: BLE001 — any browser failure
            logger.warning("Playwright capture failed for %s: %s", ticker, exc)
            return ChartContext(
                ticker=ticker.upper(),
                timeframe=timeframe,
                captured_at=now,
                screenshot_path=None,
                source="playwright",
                browser_url=url,
                error=f"playwright_exception:{exc.__class__.__name__}",
                notes=str(exc)[:200],
            )

        return ChartContext(
            ticker=ticker.upper(),
            timeframe=timeframe,
            captured_at=now,
            screenshot_path=out_path,
            source="playwright",
            browser_url=url,
        )


# ---------------------------------------------------------------------
# 3. Composable provider with automatic fallback
# ---------------------------------------------------------------------
class ChainedChartProvider:
    """Try providers in order; return the first non-error ChartContext.

    Useful for Claude-terminal workflows where the terminal drops
    screenshots to disk and the engine also has Playwright available
    as a fallback — use this to try the filesystem first (fast, zero
    browser overhead) and fall back to Playwright only when the file
    is missing or stale.
    """

    def __init__(self, providers: list[ChartContextProvider]) -> None:
        self.providers = providers

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1D",
        *,
        as_of: datetime | None = None,
    ) -> ChartContext:
        last_error_ctx: ChartContext | None = None
        for provider in self.providers:
            ctx = provider.fetch(ticker, timeframe, as_of=as_of)
            if ctx.is_ok():
                return ctx
            last_error_ctx = ctx
        # Every provider errored — return the last one so the caller
        # can inspect the most recent failure reason.
        if last_error_ctx is not None:
            return last_error_ctx
        return ChartContext(
            ticker=ticker.upper(),
            timeframe=timeframe,
            captured_at=as_of or datetime.utcnow(),
            screenshot_path=None,
            source="chained",
            error="no_providers_configured",
        )


# ---------------------------------------------------------------------
# 4. Convenience factory
# ---------------------------------------------------------------------
def build_default_provider(
    screenshots_dir: Path | str = "screenshots",
    enable_playwright_fallback: bool = False,
) -> ChartContextProvider:
    """Build the canonical provider for the Smart Wheel engine.

    By default this is just a :class:`FilesystemChartProvider` pointing
    at ``screenshots/``. When ``enable_playwright_fallback=True``, a
    Playwright provider is chained after the filesystem one.
    """
    fs = FilesystemChartProvider(base_dir=screenshots_dir)
    if not enable_playwright_fallback:
        return fs
    pw = PlaywrightChartProvider(output_dir=Path(screenshots_dir) / "_playwright")
    return ChainedChartProvider([fs, pw])
