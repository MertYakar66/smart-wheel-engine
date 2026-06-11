"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, DollarSign, TrendingUp, BarChart3, Briefcase } from "lucide-react";
import type { CalendarEvent } from "@/types";

const EVENT_TYPE_CONFIG: Record<
  string,
  { label: string; icon: React.ElementType; color: string }
> = {
  fomc: { label: "FOMC", icon: DollarSign, color: "text-blue-500" },
  cpi: { label: "CPI Release", icon: TrendingUp, color: "text-orange-500" },
  earnings: { label: "Earnings", icon: BarChart3, color: "text-green-500" },
  jobs: { label: "Jobs Report", icon: Briefcase, color: "text-purple-500" },
  gdp: { label: "GDP", icon: TrendingUp, color: "text-red-500" },
};

const MACRO_TYPES = new Set(["fomc", "cpi", "jobs", "gdp"]);

// Events carry their provenance so projected fallback dates are never
// presented as engine truth.
type SourcedEvent = CalendarEvent & { source: "engine" | "db" | "projected" };

// Static FOMC schedule — used ONLY when the engine calendar is
// unreachable, always badged "Projected". (The old page rendered a full
// hardcoded macro list as primary data; its CPI/jobs dates had already
// gone stale.)
const FALLBACK_EVENTS: CalendarEvent[] = [
  { eventId: "fomc-proj-4", eventType: "fomc", ticker: null, eventDate: "2026-06-17", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  { eventId: "fomc-proj-5", eventType: "fomc", ticker: null, eventDate: "2026-07-29", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-proj-6", eventType: "fomc", ticker: null, eventDate: "2026-09-16", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  { eventId: "fomc-proj-7", eventType: "fomc", ticker: null, eventDate: "2026-11-04", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-proj-8", eventType: "fomc", ticker: null, eventDate: "2026-12-16", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
];

interface EngineCalendarEvent {
  eventId: string;
  eventType: string;
  ticker: string | null;
  eventDate: string;
  description: string | null;
}

export default function CalendarPage() {
  const [events, setEvents] = useState<SourcedEvent[]>([]);
  const [heldSymbols, setHeldSymbols] = useState<string[]>([]);
  const [engineUp, setEngineUp] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const collected: SourcedEvent[] = [];
      let engineOk = false;

      const [dbRes, engineRes, posRes] = await Promise.allSettled([
        fetch("/api/events"),
        fetch("/api/engine?action=calendar&days=180"),
        fetch("/api/portfolio/positions"),
      ]);

      if (dbRes.status === "fulfilled" && dbRes.value.ok) {
        try {
          const data: CalendarEvent[] = await dbRes.value.json();
          collected.push(...data.map((e) => ({ ...e, source: "db" as const })));
        } catch {
          /* malformed DB payload — skip */
        }
      }

      if (engineRes.status === "fulfilled" && engineRes.value.ok) {
        try {
          const data = await engineRes.value.json();
          const engineEvents: EngineCalendarEvent[] = Array.isArray(data?.events)
            ? data.events
            : [];
          collected.push(
            ...engineEvents.map((e) => ({
              eventId: e.eventId,
              eventType: e.eventType as CalendarEvent["eventType"],
              ticker: e.ticker,
              eventDate: e.eventDate,
              description: e.description,
              source: "engine" as const,
            }))
          );
          engineOk = true;
        } catch {
          /* fall through to projected fallback */
        }
      }

      // Per-held-ticker earnings from the engine calendar. Returns nothing
      // while the engine's earnings CSV frontier is stale — the section
      // below states that explicitly instead of pretending a clear week.
      let held: string[] = [];
      if (posRes.status === "fulfilled" && posRes.value.ok) {
        try {
          const data = await posRes.value.json();
          const holdings: { sym?: string }[] = Array.isArray(data?.holdings)
            ? data.holdings
            : [];
          held = [...new Set(holdings.map((h) => h.sym).filter(Boolean))] as string[];
        } catch {
          /* no held book — earnings section shows its empty state */
        }
      }
      setHeldSymbols(held);

      if (engineOk && held.length > 0) {
        const perTicker = await Promise.allSettled(
          held.map((sym) =>
            fetch(`/api/engine?action=calendar&ticker=${encodeURIComponent(sym)}&days=120`)
          )
        );
        for (const res of perTicker) {
          if (res.status !== "fulfilled" || !res.value.ok) continue;
          try {
            const data = await res.value.json();
            const evts: EngineCalendarEvent[] = Array.isArray(data?.events)
              ? data.events
              : [];
            collected.push(
              ...evts
                .filter((e) => e.eventType === "earnings")
                .map((e) => ({
                  eventId: e.eventId,
                  eventType: e.eventType as CalendarEvent["eventType"],
                  ticker: e.ticker,
                  eventDate: e.eventDate,
                  description: e.description,
                  source: "engine" as const,
                }))
            );
          } catch {
            /* skip this ticker */
          }
        }
      }

      if (!engineOk) {
        collected.push(
          ...FALLBACK_EVENTS.map((e) => ({ ...e, source: "projected" as const }))
        );
      }

      // Dedup by eventId (engine macro events can repeat across the
      // per-ticker calls; DB rows may mirror engine rows).
      const seen = new Set<string>();
      const unique = collected.filter((e) => {
        if (seen.has(e.eventId)) return false;
        seen.add(e.eventId);
        return true;
      });

      setEvents(unique);
      setEngineUp(engineOk);
      setLoading(false);
    }
    load();
  }, []);

  const todayStr = new Date().toISOString().split("T")[0];

  const upcoming = events
    .filter((e) => e.eventDate >= todayStr)
    .sort((a, b) => a.eventDate.localeCompare(b.eventDate));
  const macroEvents = upcoming.filter((e) => MACRO_TYPES.has(e.eventType));
  const earningsEvents = upcoming.filter((e) => e.eventType === "earnings");

  const pastEvents = events
    .filter((e) => e.eventDate < todayStr)
    .sort((a, b) => b.eventDate.localeCompare(a.eventDate))
    .slice(0, 10);

  function formatDate(dateStr: string): string {
    return new Date(dateStr + "T12:00:00").toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  function daysUntil(dateStr: string): number {
    const target = new Date(dateStr + "T00:00:00");
    const today = new Date(todayStr + "T00:00:00");
    return Math.ceil((target.getTime() - today.getTime()) / 86400000);
  }

  function sourceBadge(source: SourcedEvent["source"]) {
    if (source === "engine") {
      return (
        <Badge variant="outline" className="border-green-300 text-green-600 text-[10px] uppercase">
          Engine
        </Badge>
      );
    }
    if (source === "projected") {
      return (
        <Badge variant="outline" className="border-amber-300 text-amber-600 text-[10px] uppercase">
          Projected
        </Badge>
      );
    }
    return (
      <Badge variant="outline" className="text-[10px] uppercase">
        Manual
      </Badge>
    );
  }

  function EventRow({ event }: { event: SourcedEvent }) {
    const config = EVENT_TYPE_CONFIG[event.eventType];
    const Icon = config?.icon || Calendar;
    const days = daysUntil(event.eventDate);

    return (
      <div className="flex items-center gap-4 rounded-lg border border-zinc-100 p-3 dark:border-zinc-800">
        <div className={`${config?.color || "text-zinc-400"}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <p className="text-sm font-medium">
              {event.ticker ? `${event.ticker} — ` : ""}
              {event.description || config?.label || event.eventType}
            </p>
            {sourceBadge(event.source)}
          </div>
          <p className="text-xs text-zinc-400">{formatDate(event.eventDate)}</p>
        </div>
        <div className="text-right">
          <Badge variant={days <= 7 ? "destructive" : "secondary"}>
            {days === 0 ? "Today" : days === 1 ? "Tomorrow" : `${days} days`}
          </Badge>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
            Event Calendar
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading events…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          Event Calendar
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Macro events from the engine calendar
          {heldSymbols.length > 0 && <> · earnings checked for held names ({heldSymbols.join(", ")})</>}
          {!engineUp && (
            <span className="ml-1 text-amber-600 dark:text-amber-500">
              · engine offline — showing projected FOMC schedule only
            </span>
          )}
        </p>
      </div>

      {/* Macro & Policy */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Macro &amp; Policy
          </CardTitle>
          <CardDescription>
            {macroEvents.length > 0
              ? `${macroEvents.length} upcoming events`
              : "No upcoming macro events"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {macroEvents.length > 0 ? (
            <div className="space-y-3">
              {macroEvents.map((event) => (
                <EventRow key={event.eventId} event={event} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-zinc-500">
              {engineUp
                ? "The engine calendar returned no macro events for the next 180 days."
                : "Engine offline and no projected dates in range."}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Earnings — held book */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Earnings — Held Book
          </CardTitle>
          <CardDescription>
            Engine earnings dates for symbols in the live book
          </CardDescription>
        </CardHeader>
        <CardContent>
          {earningsEvents.length > 0 ? (
            <div className="space-y-3">
              {earningsEvents.map((event) => (
                <EventRow key={event.eventId} event={event} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-zinc-500">
              {!engineUp
                ? "Engine offline — earnings dates unavailable."
                : heldSymbols.length === 0
                  ? "No held positions found, so no earnings to check."
                  : "No earnings dates returned for held names within 120 days. Note: this may reflect the engine's earnings-data frontier rather than a clear calendar."}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Past Events */}
      {pastEvents.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {pastEvents.map((event) => {
                const config = EVENT_TYPE_CONFIG[event.eventType];
                return (
                  <div
                    key={event.eventId}
                    className="flex items-center gap-3 py-2 text-zinc-500"
                  >
                    <Badge variant="outline" className="text-xs">
                      {config?.label || event.eventType}
                    </Badge>
                    <span className="text-sm">
                      {event.ticker ? `${event.ticker} — ` : ""}
                      {event.description}
                    </span>
                    <span className="ml-auto text-xs">{formatDate(event.eventDate)}</span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
