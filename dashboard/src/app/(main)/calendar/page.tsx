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

// Static calendar events (Fed schedule + known releases)
const STATIC_EVENTS: CalendarEvent[] = [
  // 2026 FOMC meetings (projected schedule)
  { eventId: "fomc-1", eventType: "fomc", ticker: null, eventDate: "2026-01-28", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-2", eventType: "fomc", ticker: null, eventDate: "2026-03-18", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  { eventId: "fomc-3", eventType: "fomc", ticker: null, eventDate: "2026-05-06", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-4", eventType: "fomc", ticker: null, eventDate: "2026-06-17", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  { eventId: "fomc-5", eventType: "fomc", ticker: null, eventDate: "2026-07-29", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-6", eventType: "fomc", ticker: null, eventDate: "2026-09-16", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  { eventId: "fomc-7", eventType: "fomc", ticker: null, eventDate: "2026-11-04", description: "FOMC Meeting Day 2 - Rate Decision" },
  { eventId: "fomc-8", eventType: "fomc", ticker: null, eventDate: "2026-12-16", description: "FOMC Meeting Day 2 - Rate Decision + SEP" },
  // CPI releases (approximate)
  { eventId: "cpi-1", eventType: "cpi", ticker: null, eventDate: "2026-01-14", description: "December 2025 CPI Report" },
  { eventId: "cpi-2", eventType: "cpi", ticker: null, eventDate: "2026-02-12", description: "January 2026 CPI Report" },
  { eventId: "cpi-3", eventType: "cpi", ticker: null, eventDate: "2026-03-12", description: "February 2026 CPI Report" },
  { eventId: "cpi-4", eventType: "cpi", ticker: null, eventDate: "2026-04-14", description: "March 2026 CPI Report" },
  { eventId: "cpi-5", eventType: "cpi", ticker: null, eventDate: "2026-05-13", description: "April 2026 CPI Report" },
  { eventId: "cpi-6", eventType: "cpi", ticker: null, eventDate: "2026-06-10", description: "May 2026 CPI Report" },
  // Jobs reports
  { eventId: "jobs-1", eventType: "jobs", ticker: null, eventDate: "2026-01-09", description: "December 2025 Employment Situation" },
  { eventId: "jobs-2", eventType: "jobs", ticker: null, eventDate: "2026-02-06", description: "January 2026 Employment Situation" },
  { eventId: "jobs-3", eventType: "jobs", ticker: null, eventDate: "2026-03-06", description: "February 2026 Employment Situation" },
  { eventId: "jobs-4", eventType: "jobs", ticker: null, eventDate: "2026-04-03", description: "March 2026 Employment Situation" },
  // GDP releases
  { eventId: "gdp-1", eventType: "gdp", ticker: null, eventDate: "2026-01-30", description: "Q4 2025 GDP Advance Estimate" },
  { eventId: "gdp-2", eventType: "gdp", ticker: null, eventDate: "2026-04-29", description: "Q1 2026 GDP Advance Estimate" },
];

export default function CalendarPage() {
  const [events, setEvents] = useState<CalendarEvent[]>(STATIC_EVENTS);

  // Sort events by date, filter to upcoming
  const upcomingEvents = events
    .filter((e) => new Date(e.eventDate) >= new Date(new Date().toISOString().split("T")[0]))
    .sort((a, b) => a.eventDate.localeCompare(b.eventDate));

  const pastEvents = events
    .filter((e) => new Date(e.eventDate) < new Date(new Date().toISOString().split("T")[0]))
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
    const today = new Date(new Date().toISOString().split("T")[0] + "T00:00:00");
    return Math.ceil((target.getTime() - today.getTime()) / 86400000);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          Macro Calendar
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          FOMC meetings, CPI releases, jobs reports, GDP estimates
        </p>
      </div>

      {/* Upcoming Events */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Upcoming Events
          </CardTitle>
          <CardDescription>
            {upcomingEvents.length} events scheduled
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {upcomingEvents.map((event) => {
              const config = EVENT_TYPE_CONFIG[event.eventType];
              const Icon = config?.icon || Calendar;
              const days = daysUntil(event.eventDate);

              return (
                <div
                  key={event.eventId}
                  className="flex items-center gap-4 rounded-lg border border-zinc-100 p-3 dark:border-zinc-800"
                >
                  <div className={`${config?.color || "text-zinc-400"}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">
                      {event.description}
                    </p>
                    <p className="text-xs text-zinc-400">
                      {formatDate(event.eventDate)}
                    </p>
                  </div>
                  <div className="text-right">
                    <Badge
                      variant={days <= 7 ? "destructive" : "secondary"}
                    >
                      {days === 0
                        ? "Today"
                        : days === 1
                          ? "Tomorrow"
                          : `${days} days`}
                    </Badge>
                    {event.ticker && (
                      <p className="mt-1 text-xs text-zinc-400">
                        {event.ticker}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
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
                    <span className="text-sm">{event.description}</span>
                    <span className="ml-auto text-xs">
                      {formatDate(event.eventDate)}
                    </span>
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
