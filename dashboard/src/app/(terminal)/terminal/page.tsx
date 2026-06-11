"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { StatusBar } from "@/components/terminal/status-bar";
import { MarketOverview } from "@/components/terminal/market-overview";
import { NewsPanel } from "@/components/terminal/news-panel";
import { OptionsPanel } from "@/components/terminal/options-panel";
import { LiveBookPanel } from "@/components/terminal/live-book-panel";
import { WatchlistPanel } from "@/components/terminal/watchlist-panel";
import { MacroPanel } from "@/components/terminal/macro-panel";
import { CommandLine } from "@/components/terminal/command-line";
import { ChatPanel } from "@/components/terminal/chat-panel";
import type { StoryCard, CalendarEvent } from "@/types";
import { TradingViewLinkRow } from "@/components/terminal/tradingview-link-panel";
import { TickerAnalysisPanel } from "@/components/terminal/ticker-analysis-panel";
import { DealerPositioningPanel } from "@/components/terminal/dealer-positioning-panel";
import { PanelErrorBoundary } from "@/components/terminal/panel-error-boundary";
import { CrossPageNav, WheelhouseHeader } from "@/components/shell/wheelhouse-header";
import { useEngineData, useLiveBook } from "@/hooks/useEngineData";

// Panels the command line can highlight (one-shot flash).
type FlashTarget = "news" | "options" | "calendar" | "book" | "market";

// ─── Main Terminal Dashboard ───────────────────────────────────────────

export default function TerminalPage() {
  // Options engine data (connected to smart-wheel-engine)
  const engineData = useEngineData();

  // Live IBKR book (read-only /api/portfolio proxy)
  const book = useLiveBook();

  // Stories state (connected to real API)
  const [stories, setStories] = useState<StoryCard[]>([]);
  const [storiesLoading, setStoriesLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);

  // Watchlist state (connected to real API)
  const [watchlist, setWatchlist] = useState<
    { ticker: string; price: number | null; changePct: number | null; addedAt: string }[]
  >([]);
  const [watchlistLoading, setWatchlistLoading] = useState(true);

  // Events state (connected to real API)
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(true);

  // Alerts
  const [alertCount, setAlertCount] = useState(0);

  // Ollama status
  const [ollamaStatus, setOllamaStatus] = useState<"connected" | "disconnected" | "checking">("checking");

  // Command history
  const [commandHistory, setCommandHistory] = useState<string[]>([]);

  // Chat query from command line
  const [chatQuery, setChatQuery] = useState<string | undefined>();

  // Selected story for detail
  const [selectedStory, setSelectedStory] = useState<StoryCard | null>(null);

  // Selected ticker for the symbol workbench
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

  // One-shot panel highlight for command-line focus commands
  const [flashPanel, setFlashPanel] = useState<FlashTarget | null>(null);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const triggerFlash = useCallback((target: FlashTarget) => {
    if (flashTimer.current) clearTimeout(flashTimer.current);
    setFlashPanel(target);
    // Clear after the 0.5s CSS animation so a repeat command re-triggers it.
    flashTimer.current = setTimeout(() => setFlashPanel(null), 700);
  }, []);
  useEffect(() => {
    return () => {
      if (flashTimer.current) clearTimeout(flashTimer.current);
    };
  }, []);

  // Data epoch for the panel error boundaries: a healthy poll (engine or
  // book) un-latches any panel stuck on a transient malformed payload.
  const dataEpoch = `${engineData.lastUpdated?.getTime() ?? 0}-${
    book.lastUpdated?.getTime() ?? 0
  }`;

  // ─── Data fetching ─────────────────────────────────────────────────

  const fetchStories = useCallback(async () => {
    setStoriesLoading(true);
    try {
      const res = await fetch("/api/stories?limit=30");
      if (res.ok) {
        const data = await res.json();
        setStories(data);
      }
    } catch (err) {
      console.error("Failed to fetch stories:", err);
    } finally {
      setStoriesLoading(false);
    }
  }, []);

  const fetchWatchlist = useCallback(async () => {
    setWatchlistLoading(true);
    try {
      const res = await fetch("/api/watchlist");
      if (res.ok) {
        const data = await res.json();
        setWatchlist(data);
      }
    } catch (err) {
      console.error("Failed to fetch watchlist:", err);
    } finally {
      setWatchlistLoading(false);
    }
  }, []);

  // Held-book underlyings, stable key so the events fetch re-runs only when
  // the actual set of names changes (not on every poll-driven array identity).
  const heldKey = book.heldTickers.join(",");

  const fetchEvents = useCallback(async () => {
    setEventsLoading(true);
    try {
      const held = heldKey ? heldKey.split(",") : [];
      // DB events + engine macro calendar + per-held-name engine calendar
      // (the engine's earnings branch needs a ticker param to be reachable).
      const [dbRes, engineRes, ...tickerRes] = await Promise.all([
        fetch("/api/events").catch(() => null),
        fetch("/api/engine?action=calendar&days=60").catch(() => null),
        ...held.map((t) =>
          fetch(`/api/engine?action=calendar&days=45&ticker=${encodeURIComponent(t)}`).catch(
            () => null
          )
        ),
      ]);

      const allEvents: CalendarEvent[] = [];

      if (dbRes?.ok) {
        const data = await dbRes.json();
        allEvents.push(...data);
      }

      for (const res of [engineRes, ...tickerRes]) {
        if (!res?.ok) continue;
        const data = await res.json();
        const engineEvents = (data.events || []).map((e: { eventId: string; eventType: string; ticker: string | null; eventDate: string; description: string; daysUntil: number }) => ({
          eventId: e.eventId,
          eventType: e.eventType,
          ticker: e.ticker,
          eventDate: e.eventDate,
          description: e.description,
          daysUntil: e.daysUntil,
        }));
        allEvents.push(...engineEvents);
      }

      // Deduplicate by eventId (the per-ticker calls repeat the FOMC rows)
      const seen = new Set<string>();
      const unique = allEvents.filter((e) => {
        if (seen.has(e.eventId)) return false;
        seen.add(e.eventId);
        return true;
      });

      setEvents(unique);
    } catch (err) {
      console.error("Failed to fetch events:", err);
    } finally {
      setEventsLoading(false);
    }
  }, [heldKey]);

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch("/api/alerts");
      if (res.ok) {
        const data = await res.json();
        setAlertCount(data.length);
      }
    } catch {
      // silent
    }
  }, []);

  const checkOllama = useCallback(async () => {
    try {
      // Proxy through the same-origin engine route. A direct browser fetch to
      // Ollama (localhost:11434) is CORS-blocked and always reports
      // "disconnected" even when Ollama is healthy; the engine checks Ollama
      // server-side — and it is the engine that actually calls Ollama for memos.
      const res = await fetch("/api/engine?action=ollama_status", {
        signal: AbortSignal.timeout(4000),
      });
      if (!res.ok) {
        setOllamaStatus("disconnected");
        return;
      }
      const data = await res.json();
      const connected = Boolean(
        data?.available ||
          data?.connected ||
          data?.ok ||
          data?.status === "connected" ||
          data?.status === "ok"
      );
      setOllamaStatus(connected ? "connected" : "disconnected");
    } catch {
      setOllamaStatus("disconnected");
    }
  }, []);

  // Initial data load (events re-fetch when the held-book names change)
  useEffect(() => {
    fetchStories();
    fetchWatchlist();
    fetchAlerts();
    checkOllama();
  }, [fetchStories, fetchWatchlist, fetchAlerts, checkOllama]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // ─── Handlers ──────────────────────────────────────────────────────

  const handleIngest = async () => {
    setIngesting(true);
    try {
      await fetch("/api/ingest", { method: "POST" });
      await fetchStories();
    } catch (err) {
      console.error("Ingestion failed:", err);
    } finally {
      setIngesting(false);
    }
  };

  const handleAddTicker = async (ticker: string) => {
    try {
      await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.toUpperCase() }),
      });
      await fetchWatchlist();
    } catch (err) {
      console.error("Failed to add ticker:", err);
    }
  };

  const handleRemoveTicker = async (ticker: string) => {
    try {
      await fetch(`/api/watchlist?ticker=${ticker}`, { method: "DELETE" });
      await fetchWatchlist();
    } catch (err) {
      console.error("Failed to remove ticker:", err);
    }
  };

  // ─── Command handler ──────────────────────────────────────────────
  // Every command advertised in the command line's HELP has a case here —
  // no advertised command may silently no-op or fall through to the ticker
  // matcher (NEWS/AGENT used to open TradingView for fake symbols).

  const handleCommand = (cmd: string) => {
    setCommandHistory((prev) => [...prev, cmd]);

    const parts = cmd.split(/\s+/);
    const action = parts[0];
    const arg = parts.slice(1).join(" ");

    switch (action) {
      case "REFRESH":
      case "INGEST":
        handleIngest();
        break;
      case "NEWS":
        triggerFlash("news");
        break;
      case "OPTIONS":
        triggerFlash("options");
        break;
      case "CALENDAR":
        triggerFlash("calendar");
        break;
      case "BOOK":
        triggerFlash("book");
        break;
      case "MARKET":
        triggerFlash("market");
        break;
      case "WATCH":
        if (arg) handleAddTicker(arg);
        break;
      case "UNWATCH":
        if (arg) handleRemoveTicker(arg);
        break;
      case "QUOTE":
        // Read-only: open the symbol workbench (engine EOD read) — QUOTE no
        // longer mutates the watchlist DB as a side effect.
        if (arg) setSelectedTicker(arg.toUpperCase());
        break;
      case "RESEARCH":
        if (arg) {
          setChatQuery(arg);
        }
        break;
      case "CLEAR":
        setCommandHistory([]);
        break;
      case "STORY":
        if (arg) {
          // Open story by index (1-based)
          const idx = parseInt(arg) - 1;
          if (stories[idx]) {
            window.open(`/story/${stories[idx].storyId}`, "_blank");
          }
        } else if (selectedStory) {
          window.open(`/story/${selectedStory.storyId}`, "_blank");
        }
        break;
      case "IMPACT":
        if (selectedStory) {
          setChatQuery(`Explain the market impact of: "${selectedStory.canonicalTitle}". Who is exposed? What mechanism drives the impact? Over what time horizon?`);
        }
        break;
      case "ENGINE":
        // Refresh engine data
        engineData.refresh();
        break;
      case "CHART":
      case "ANALYZE":
        // Open the symbol workbench (analysis + dealer read + TV handoff)
        if (arg) setSelectedTicker(arg.toUpperCase());
        break;
      case "BACK":
      case "CLOSE":
        // Close the workbench and return to the dashboard
        setSelectedTicker(null);
        break;
      case "HELP":
        // Help is shown via the command line component
        break;
      default:
        // If it looks like a ticker symbol, open the workbench
        if (/^[A-Z]{1,5}$/.test(action)) {
          setSelectedTicker(action);
        }
        break;
    }
  };

  const handleSelectStory = (story: StoryCard) => {
    setSelectedStory(story);
    setChatQuery(`Analyze this financial news: "${story.canonicalTitle}"`);
  };

  // ─── Render ────────────────────────────────────────────────────────

  const liveNav =
    book.summary?.source === "live" ? book.summary.netLiq : null;

  return (
    <div className="flex h-screen flex-col bg-pf-bg font-mono">
      {/* Shared Wheelhouse chrome — branding + cross-page tabs */}
      <PanelErrorBoundary label="Header" resetKey={dataEpoch}>
        <WheelhouseHeader
          page="Terminal"
          maxW="max-w-none"
          right={
            <>
              {engineData.regime.regime && engineData.regime.regime !== "---" && (
                <span className="text-xs uppercase tracking-wider text-terminal-dim">
                  {engineData.regime.regime}
                </span>
              )}
              {engineData.regime.vix > 0 && (
                <span className="text-sm font-semibold tabular-nums text-terminal-text">
                  VIX {engineData.regime.vix.toFixed(1)}
                </span>
              )}
            </>
          }
          status={
            <>
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  engineData.connected ? "bg-pf-ok" : "bg-terminal-dim"
                }`}
              />
              <span className={engineData.connected ? "text-pf-ok" : "text-terminal-dim"}>
                {engineData.connected ? "Engine live" : "Engine offline"}
              </span>
            </>
          }
        >
          <CrossPageNav active="Terminal" />
        </WheelhouseHeader>
      </PanelErrorBoundary>

      {/* Status Bar — real reads only (VIX complex, NAV, frontier, feed) */}
      <PanelErrorBoundary label="Status Bar" resetKey={dataEpoch}>
        <StatusBar
          alertCount={alertCount}
          ollamaStatus={ollamaStatus}
          storyCount={storiesLoading ? null : stories.length}
          vix={engineData.regime.vix}
          vix3m={engineData.regime.vix3m}
          contango={engineData.regime.contango}
          regime={engineData.regime.regime}
          dataFrontier={engineData.dataFrontier}
          nav={liveNav}
        />
      </PanelErrorBoundary>

      {/* Main Grid — each panel gets its own error boundary so one panel's
          throw (e.g. a malformed engine payload) degrades that panel alone
          instead of taking down the whole terminal + crash-looping the poll.
          resetKey un-latches a tripped boundary on the next healthy poll. */}
      {selectedTicker ? (
        /* Symbol Workbench: compact TV handoff row + engine substance */
        <div className="flex-1 grid grid-cols-[1fr_350px] gap-[1px] bg-terminal-border p-[1px] overflow-hidden">
          <div className="flex min-h-0 flex-col gap-[1px]">
            <TradingViewLinkRow
              symbol={selectedTicker}
              onClose={() => setSelectedTicker(null)}
            />
            <div className="grid min-h-0 flex-1 grid-rows-2 gap-[1px]">
              <PanelErrorBoundary label="Engine Read" resetKey={dataEpoch}>
                <TickerAnalysisPanel
                  ticker={selectedTicker}
                  dataFrontier={engineData.dataFrontier}
                />
              </PanelErrorBoundary>
              <PanelErrorBoundary label="Dealer Positioning" resetKey={dataEpoch}>
                <DealerPositioningPanel ticker={selectedTicker} />
              </PanelErrorBoundary>
            </div>
          </div>
          <div className="grid grid-rows-2 gap-[1px]">
            <PanelErrorBoundary label="Options Engine" resetKey={dataEpoch}>
              <OptionsPanel
                trades={engineData.trades}
                regime={engineData.regime}
                connected={engineData.connected}
              />
            </PanelErrorBoundary>
            <PanelErrorBoundary label="Research" resetKey={dataEpoch}>
              <ChatPanel initialQuery={chatQuery} />
            </PanelErrorBoundary>
          </div>
        </div>
      ) : (
        /* Default View: full terminal dashboard */
        <div className="flex-1 grid grid-cols-3 grid-rows-2 gap-[1px] bg-terminal-border p-[1px] overflow-hidden">
          {/* Row 1 */}
          <PanelErrorBoundary label="Market / Vol" resetKey={dataEpoch}>
            <MarketOverview
              regime={engineData.regime}
              dataFrontier={engineData.dataFrontier}
              connected={engineData.connected}
              flash={flashPanel === "market"}
            />
          </PanelErrorBoundary>
          <PanelErrorBoundary label="Options Engine" resetKey={dataEpoch}>
            <OptionsPanel
              trades={engineData.trades}
              regime={engineData.regime}
              connected={engineData.connected}
              flash={flashPanel === "options"}
            />
          </PanelErrorBoundary>
          <PanelErrorBoundary label="Live Book" resetKey={dataEpoch}>
            <LiveBookPanel
              summary={book.summary}
              legs={book.legs}
              loading={book.loading}
              error={book.error}
              flash={flashPanel === "book"}
            />
          </PanelErrorBoundary>

          {/* Row 2 */}
          <PanelErrorBoundary label="News" resetKey={dataEpoch}>
            <NewsPanel
              stories={stories}
              loading={storiesLoading}
              onRefresh={handleIngest}
              refreshing={ingesting}
              onSelectStory={handleSelectStory}
              flash={flashPanel === "news"}
            />
          </PanelErrorBoundary>
          <PanelErrorBoundary label="Watchlist" resetKey={dataEpoch}>
            <WatchlistPanel
              items={watchlist}
              loading={watchlistLoading}
              onRefresh={fetchWatchlist}
            />
          </PanelErrorBoundary>
          <div className="grid grid-rows-2 gap-[1px]">
            <PanelErrorBoundary label="Events" resetKey={dataEpoch}>
              <MacroPanel
                events={events}
                loading={eventsLoading}
                heldTickers={book.heldTickers}
                flash={flashPanel === "calendar"}
              />
            </PanelErrorBoundary>
            <PanelErrorBoundary label="Research" resetKey={dataEpoch}>
              <ChatPanel initialQuery={chatQuery} />
            </PanelErrorBoundary>
          </div>
        </div>
      )}

      {/* Command Line */}
      <PanelErrorBoundary label="Command Line" resetKey={dataEpoch}>
        <CommandLine onCommand={handleCommand} history={commandHistory} />
      </PanelErrorBoundary>
    </div>
  );
}
