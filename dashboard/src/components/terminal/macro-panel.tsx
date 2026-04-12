"use client";

import { TerminalPanel, TerminalDivider, TerminalBadge } from "./panel";
import type { CalendarEvent } from "@/types";

interface MacroPanelProps {
  events: CalendarEvent[];
  loading: boolean;
}

const EVENT_ICONS: Record<string, string> = {
  fomc: "🏛",
  cpi: "📊",
  jobs: "👥",
  gdp: "📈",
  earnings: "💰",
};

const EVENT_COLORS: Record<string, "amber" | "blue" | "green" | "red"> = {
  fomc: "amber",
  cpi: "blue",
  jobs: "green",
  gdp: "blue",
  earnings: "green",
};

function daysUntil(dateStr: string): number {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  const target = new Date(dateStr);
  target.setHours(0, 0, 0, 0);
  return Math.ceil((target.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
}

export function MacroPanel({ events, loading }: MacroPanelProps) {
  const now = new Date().toISOString().split("T")[0];
  const upcoming = events
    .filter((e) => e.eventDate >= now)
    .sort((a, b) => a.eventDate.localeCompare(b.eventDate))
    .slice(0, 8);

  const past = events
    .filter((e) => e.eventDate < now)
    .sort((a, b) => b.eventDate.localeCompare(a.eventDate))
    .slice(0, 4);

  return (
    <TerminalPanel title="Macro Calendar" tag="ECON">
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Loading events...
        </div>
      ) : (
        <>
          {/* Upcoming */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ UPCOMING
          </div>
          {upcoming.length === 0 ? (
            <div className="py-2 text-center text-terminal-dim">
              No upcoming events
            </div>
          ) : (
            upcoming.map((event) => {
              const days = daysUntil(event.eventDate);
              return (
                <div
                  key={event.eventId}
                  className="flex items-center justify-between py-[2px] hover:bg-terminal-border/30"
                >
                  <div className="flex items-center gap-1.5 min-w-0 flex-1">
                    <span className="text-[10px]">
                      {EVENT_ICONS[event.eventType] || "•"}
                    </span>
                    <span className="text-terminal-text truncate">
                      {event.description || event.eventType.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[10px] text-terminal-dim">
                      {new Date(event.eventDate).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      })}
                    </span>
                    <TerminalBadge
                      variant={
                        days <= 3
                          ? "red"
                          : days <= 7
                          ? "amber"
                          : "default"
                      }
                    >
                      {days}d
                    </TerminalBadge>
                  </div>
                </div>
              );
            })
          )}

          <TerminalDivider />

          {/* Recent/Past */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ RECENT
          </div>
          {past.length === 0 ? (
            <div className="py-1 text-terminal-dim text-[10px]">
              No recent events
            </div>
          ) : (
            past.map((event) => (
              <div
                key={event.eventId}
                className="flex items-center justify-between py-[1px] opacity-60"
              >
                <span className="text-terminal-dim truncate text-[11px]">
                  {EVENT_ICONS[event.eventType]}{" "}
                  {event.description || event.eventType.toUpperCase()}
                </span>
                <span className="text-[10px] text-terminal-dim shrink-0">
                  {new Date(event.eventDate).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  })}
                </span>
              </div>
            ))
          )}

          <TerminalDivider />

          {/* Key indicators — compact labels so values don't clip at narrow widths */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ KEY INDICATORS
          </div>
          <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px]">
            <div className="flex justify-between gap-1 min-w-0">
              <span className="text-terminal-dim shrink-0">FFR</span>
              <span className="text-terminal-amber tabular-nums truncate">5.25-5.50%</span>
            </div>
            <div className="flex justify-between gap-1 min-w-0">
              <span className="text-terminal-dim shrink-0">CPI</span>
              <span className="text-terminal-text tabular-nums truncate">3.1%</span>
            </div>
            <div className="flex justify-between gap-1 min-w-0">
              <span className="text-terminal-dim shrink-0">UNEMP</span>
              <span className="text-terminal-green tabular-nums truncate">3.7%</span>
            </div>
            <div className="flex justify-between gap-1 min-w-0">
              <span className="text-terminal-dim shrink-0">GDP</span>
              <span className="text-terminal-green tabular-nums truncate">+3.3%</span>
            </div>
          </div>
        </>
      )}
    </TerminalPanel>
  );
}
