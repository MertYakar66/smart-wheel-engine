"use client";

import { TerminalPanel, TerminalDivider, TerminalBadge } from "./panel";
import type { CalendarEvent } from "@/types";

interface MacroPanelProps {
  events: CalendarEvent[];
  loading: boolean;
  /** Held-book underlyings whose earnings the page queried the engine for —
   *  names the empty state can honestly reference. */
  heldTickers?: string[];
  /** One-shot highlight when the command line targets this panel. */
  flash?: boolean;
}

const EVENT_ICONS: Record<string, string> = {
  fomc: "🏛",
  cpi: "📊",
  jobs: "👥",
  gdp: "📈",
  earnings: "💰",
};

// Local-day string "YYYY-MM-DD" for today, used as a stable string
// comparison baseline — avoids UTC/local-time skew in filter and daysUntil.
function localTodayStr(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Days until a date-only string, using a consistent local-day baseline. */
function daysUntil(dateStr: string): number {
  // Parse as local noon to avoid UTC-midnight→local-day-shift in negative-UTC
  // offsets (e.g. "2026-06-17" → Jun 16 20:00 EDT). Same pattern as
  // calendar/page.tsx formatDate.
  const target = new Date(dateStr + "T12:00:00");
  const today = new Date(localTodayStr() + "T12:00:00");
  return Math.round((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
}

function EventRow({ event }: { event: CalendarEvent }) {
  const days = daysUntil(event.eventDate);
  return (
    <div className="flex items-center justify-between py-[2px] hover:bg-terminal-border/30">
      <div className="flex min-w-0 flex-1 items-center gap-1.5">
        <span className="text-[10px]">{EVENT_ICONS[event.eventType] || "•"}</span>
        {event.ticker && (
          <span className="shrink-0 text-terminal-amber">{event.ticker}</span>
        )}
        <span className="truncate text-terminal-text">
          {event.description || event.eventType.toUpperCase()}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="text-[10px] text-terminal-dim">
          {/* Local noon: avoids the UTC-midnight one-day-early shift. */}
          {new Date(event.eventDate + "T12:00:00").toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })}
        </span>
        <TerminalBadge
          variant={days <= 3 ? "red" : days <= 7 ? "amber" : "default"}
        >
          {days}d
        </TerminalBadge>
      </div>
    </div>
  );
}

export function MacroPanel({
  events,
  loading,
  heldTickers = [],
  flash,
}: MacroPanelProps) {
  // Use the same local-day baseline as daysUntil so today-local events are
  // never filtered into 'past' before local midnight.
  const now = localTodayStr();
  const upcoming = events
    .filter((e) => e.eventDate >= now)
    .sort((a, b) => a.eventDate.localeCompare(b.eventDate));

  // Earnings (per-name, event-lockout relevant) vs macro (FOMC/CPI/jobs/GDP)
  // are different decisions for a wheel book — render them distinctly.
  const earnings = upcoming.filter((e) => e.eventType === "earnings").slice(0, 6);
  const macro = upcoming.filter((e) => e.eventType !== "earnings").slice(0, 6);

  return (
    <TerminalPanel title="Events" tag="ECON" flash={flash}>
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Loading events...
        </div>
      ) : (
        <>
          {/* Macro events (FOMC etc.) */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ MACRO
          </div>
          {macro.length === 0 ? (
            <div className="py-1 text-[10px] italic text-terminal-dim">
              No macro events from the engine calendar
            </div>
          ) : (
            macro.map((event) => <EventRow key={event.eventId} event={event} />)
          )}

          <TerminalDivider />

          {/* Earnings inside the held book — the engine hard-locks entries
              inside the earnings buffer, so absence here is itself a signal
              worth stating honestly. */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ EARNINGS{heldTickers.length > 0 ? " (HELD BOOK)" : ""}
          </div>
          {earnings.length === 0 ? (
            <div className="py-1 text-[10px] italic text-terminal-dim">
              {heldTickers.length > 0
                ? `No earnings events from the engine for ${heldTickers.join(", ")}`
                : "No earnings events from the engine calendar"}
            </div>
          ) : (
            earnings.map((event) => (
              <EventRow key={event.eventId} event={event} />
            ))
          )}
        </>
      )}
    </TerminalPanel>
  );
}
