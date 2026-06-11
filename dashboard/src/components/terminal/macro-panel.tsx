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

function daysUntil(dateStr: string): number {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  const target = new Date(dateStr);
  target.setHours(0, 0, 0, 0);
  return Math.ceil((target.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
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
          {new Date(event.eventDate).toLocaleDateString("en-US", {
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
  const now = new Date().toISOString().split("T")[0];
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
