"""
Hard event-lockout gate for the Smart Wheel Engine.

What this module does
---------------------
The wheel strategy's single largest source of tail losses is entering
into a scheduled news event — earnings, FOMC, CPI, NFP. The EV engine
already de-weights near these events via ``regime_multiplier``, but a
soft down-weight is not enough: for some events the *entire distribution*
is mis-specified (jump regimes, IV-crush dynamics) and no probabilistic
model trained on non-event data can price the post-event move correctly.

This module provides a **hard** lockout gate that the EV engine calls
before ranking. Any candidate whose holding period touches a scheduled
high-impact event within a configured buffer is filtered out with a
structured reason string, surfaced to the trader as
``reason="event_lockout:earnings"`` (or similar).

The gate is deliberately *separate* from the softer regime multiplier
— by design, traders cannot bypass it with a "high EV" verdict, because
the gate runs BEFORE EV is computed.

Usage
-----
::

    from engine.event_gate import EventGate, ScheduledEvent
    gate = EventGate(earnings_buffer_days=5, macro_buffer_days=1)
    gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 4, 30)))
    gate.add_event(ScheduledEvent("*", "fomc", date(2026, 5, 1)))

    block, reason = gate.is_blocked(
        ticker="AAPL",
        trade_start=date(2026, 4, 25),
        trade_end=date(2026, 5, 20),
    )
    # block=True, reason="earnings @ 2026-04-30 (within 5-day buffer)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal


EventKind = Literal[
    "earnings",
    "fomc",
    "cpi",
    "nfp",
    "pce",
    "ecb",
    "boe",
    "dividend",
    "split",
    "custom",
]


@dataclass
class ScheduledEvent:
    """A scheduled market-moving event.

    ``ticker`` may be a specific underlying (``"AAPL"``) or the
    wildcard ``"*"`` to indicate a macro event that affects every
    candidate.
    """

    ticker: str
    kind: EventKind
    event_date: date
    note: str = ""


@dataclass
class EventGate:
    """Hard event lockout gate for wheel candidate pre-filtering."""

    earnings_buffer_days: int = 5
    macro_buffer_days: int = 1
    dividend_buffer_days: int = 1
    split_buffer_days: int = 3
    events: list[ScheduledEvent] = field(default_factory=list)

    # ------------------------------------------------------------------
    def add_event(self, event: ScheduledEvent) -> None:
        self.events.append(event)

    def add_events(self, events: list[ScheduledEvent]) -> None:
        self.events.extend(events)

    def clear(self) -> None:
        self.events.clear()

    # ------------------------------------------------------------------
    def _buffer_for(self, kind: EventKind) -> int:
        if kind == "earnings":
            return self.earnings_buffer_days
        if kind in ("fomc", "cpi", "nfp", "pce", "ecb", "boe"):
            return self.macro_buffer_days
        if kind == "dividend":
            return self.dividend_buffer_days
        if kind == "split":
            return self.split_buffer_days
        return 0

    def _event_applies_to(self, event: ScheduledEvent, ticker: str) -> bool:
        return event.ticker == "*" or event.ticker.upper() == ticker.upper()

    def _event_touches_window(
        self,
        event: ScheduledEvent,
        trade_start: date,
        trade_end: date,
    ) -> bool:
        buf = self._buffer_for(event.kind)
        window_start = trade_start - timedelta(days=buf)
        window_end = trade_end + timedelta(days=buf)
        return window_start <= event.event_date <= window_end

    # ------------------------------------------------------------------
    def is_blocked(
        self,
        ticker: str,
        trade_start: date,
        trade_end: date,
    ) -> tuple[bool, str]:
        """Return (blocked, reason_string).

        When multiple events match, the first-sorted (by date) is
        returned as the reason.
        """
        hits: list[ScheduledEvent] = []
        for ev in self.events:
            if not self._event_applies_to(ev, ticker):
                continue
            if self._event_touches_window(ev, trade_start, trade_end):
                hits.append(ev)
        if not hits:
            return False, ""
        hits.sort(key=lambda e: e.event_date)
        first = hits[0]
        buf = self._buffer_for(first.kind)
        return True, (
            f"event_lockout:{first.kind}@{first.event_date.isoformat()} "
            f"(±{buf}d buffer)"
        )

    def filter_candidates(
        self,
        candidates: list[dict],
        trade_start_key: str = "trade_date",
        trade_end_key: str = "expiration",
        ticker_key: str = "ticker",
    ) -> tuple[list[dict], list[dict]]:
        """Partition a list of candidate dicts into (kept, blocked).

        Each blocked candidate gets an added ``event_lockout_reason``
        field so callers can log or display why it was dropped.
        """
        kept: list[dict] = []
        blocked: list[dict] = []
        for c in candidates:
            ticker = c.get(ticker_key, "")
            ts = c.get(trade_start_key)
            te = c.get(trade_end_key)
            if ts is None or te is None:
                kept.append(c)
                continue
            # Normalise to date if it's a datetime
            ts = ts.date() if hasattr(ts, "date") else ts
            te = te.date() if hasattr(te, "date") else te
            blk, reason = self.is_blocked(ticker, ts, te)
            if blk:
                c = {**c, "event_lockout_reason": reason}
                blocked.append(c)
            else:
                kept.append(c)
        return kept, blocked

    # ------------------------------------------------------------------
    @classmethod
    def from_bloomberg_calendar(
        cls,
        earnings_df,
        macro_df=None,
        dividends_df=None,
        earnings_buffer_days: int = 5,
        macro_buffer_days: int = 1,
        dividend_buffer_days: int = 1,
    ) -> "EventGate":
        """Build a gate from standard Bloomberg-shape DataFrames.

        ``earnings_df`` must have columns ``ticker`` and
        ``announcement_date``. ``macro_df`` must have columns ``event``
        and ``date``. ``dividends_df`` must have ``ticker`` and
        ``ex_date``. Any of them may be ``None``.
        """
        gate = cls(
            earnings_buffer_days=earnings_buffer_days,
            macro_buffer_days=macro_buffer_days,
            dividend_buffer_days=dividend_buffer_days,
        )

        if earnings_df is not None and len(earnings_df) > 0:
            for _, row in earnings_df.iterrows():
                ts = row.get("announcement_date")
                if ts is None:
                    continue
                d = ts.date() if hasattr(ts, "date") else ts
                gate.add_event(
                    ScheduledEvent(
                        ticker=str(row.get("ticker", "")),
                        kind="earnings",
                        event_date=d,
                    )
                )

        if macro_df is not None and len(macro_df) > 0:
            for _, row in macro_df.iterrows():
                ts = row.get("date")
                if ts is None:
                    continue
                d = ts.date() if hasattr(ts, "date") else ts
                ev_kind = str(row.get("event", "custom")).lower()
                if ev_kind not in (
                    "fomc",
                    "cpi",
                    "nfp",
                    "pce",
                    "ecb",
                    "boe",
                    "custom",
                ):
                    ev_kind = "custom"
                gate.add_event(
                    ScheduledEvent(ticker="*", kind=ev_kind, event_date=d)
                )

        if dividends_df is not None and len(dividends_df) > 0:
            for _, row in dividends_df.iterrows():
                ts = row.get("ex_date")
                if ts is None:
                    continue
                d = ts.date() if hasattr(ts, "date") else ts
                gate.add_event(
                    ScheduledEvent(
                        ticker=str(row.get("ticker", "")),
                        kind="dividend",
                        event_date=d,
                    )
                )

        return gate
