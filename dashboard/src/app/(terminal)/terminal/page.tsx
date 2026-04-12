"use client";

import { useState, useEffect, useCallback } from "react";
import { StatusBar } from "@/components/terminal/status-bar";
import { MarketOverview } from "@/components/terminal/market-overview";
import { NewsPanel } from "@/components/terminal/news-panel";
import { OptionsPanel } from "@/components/terminal/options-panel";
import { AgentPanel } from "@/components/terminal/agent-panel";
import { WatchlistPanel } from "@/components/terminal/watchlist-panel";
import { MacroPanel } from "@/components/terminal/macro-panel";
import { CommandLine } from "@/components/terminal/command-line";
import { ChatPanel } from "@/components/terminal/chat-panel";
import type {
  StoryCard,
  CalendarEvent,
  MarketIndex,
  AgentStatus,
  AgentTask,
} from "@/types";
import { ChartPanel } from "@/components/terminal/chart-panel";
import { useEngineData } from "@/hooks/useEngineData";

// ─── Placeholder data for systems not yet connected ────────────────────

const PLACEHOLDER_INDICES: MarketIndex[] = [
  { symbol: "SPX", name: "S&P 500", price: 5234.18, changePct: 0.42, change: 21.87 },
  { symbol: "NDX", name: "Nasdaq 100", price: 18432.51, changePct: 0.67, change: 122.74 },
  { symbol: "DJI", name: "Dow Jones", price: 39821.44, changePct: 0.31, change: 123.15 },
  { symbol: "RUT", name: "Russell 2000", price: 2087.63, changePct: -0.14, change: -2.92 },
];

const PLACEHOLDER_FUTURES: MarketIndex[] = [
  { symbol: "ES", name: "E-mini S&P", price: 5240.50, changePct: 0.52, change: 27.25 },
  { symbol: "NQ", name: "E-mini Nasdaq", price: 18450.25, changePct: 0.71, change: 130.50 },
  { symbol: "YM", name: "E-mini Dow", price: 39850.0, changePct: 0.38, change: 150.0 },
  { symbol: "RTY", name: "E-mini Russell", price: 2090.10, changePct: -0.08, change: -1.67 },
];

const PLACEHOLDER_COMMODITIES: MarketIndex[] = [
  { symbol: "CL", name: "Crude Oil", price: 78.42, changePct: -1.23, change: -0.98 },
  { symbol: "GC", name: "Gold", price: 2045.80, changePct: 0.18, change: 3.68 },
  { symbol: "SI", name: "Silver", price: 23.15, changePct: 0.52, change: 0.12 },
  { symbol: "NG", name: "Nat Gas", price: 2.87, changePct: -2.41, change: -0.07 },
];

// Options engine data is now fetched via useEngineData hook

const PLACEHOLDER_AGENT_STATUS: AgentStatus = {
  online: true,
  model: "Qwen2.5-VL 7B",
  vramUsage: 14.2,
  vramTotal: 16,
  ramUsage: 9.1,
  activeTabs: 3,
  tasksCompleted: 47,
  uptime: "4h 32m",
};

const PLACEHOLDER_AGENT_TASKS: AgentTask[] = [
  { id: "1", description: "Scanning SEC 10-K filings for AAPL", status: "running", startedAt: "2026-03-02T10:30:00Z" },
  { id: "2", description: "Fetching FOMC meeting minutes", status: "queued" },
  { id: "3", description: "Monitoring earnings calendar updates", status: "queued" },
  { id: "4", description: "RSS feed refresh cycle", status: "completed", startedAt: "2026-03-02T10:15:00Z", completedAt: "2026-03-02T10:18:00Z" },
  { id: "5", description: "Collecting VIX term structure data", status: "completed", startedAt: "2026-03-02T10:00:00Z", completedAt: "2026-03-02T10:04:00Z" },
  { id: "6", description: "Analyzing market regime signals", status: "completed", startedAt: "2026-03-02T09:45:00Z", completedAt: "2026-03-02T09:52:00Z" },
];

// ─── Main Terminal Dashboard ───────────────────────────────────────────

export default function TerminalPage() {
  // Options engine data (connected to smart-wheel-engine)
  const engineData = useEngineData();

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

  // Retrieval providers
  const [retrievalProviders, setRetrievalProviders] = useState<string[]>(["rss"]);

  // Command history
  const [commandHistory, setCommandHistory] = useState<string[]>([]);

  // Chat query from command line
  const [chatQuery, setChatQuery] = useState<string | undefined>();

  // Selected story for detail
  const [selectedStory, setSelectedStory] = useState<StoryCard | null>(null);

  // Selected ticker for chart panel
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

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

  const fetchEvents = useCallback(async () => {
    setEventsLoading(true);
    try {
      // Fetch from both database and engine calendar
      const [dbRes, engineRes] = await Promise.all([
        fetch("/api/events").catch(() => null),
        fetch("/api/engine?action=calendar&days=60").catch(() => null),
      ]);

      const allEvents: CalendarEvent[] = [];

      if (dbRes?.ok) {
        const data = await dbRes.json();
        allEvents.push(...data);
      }

      if (engineRes?.ok) {
        const data = await engineRes.json();
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

      // Deduplicate by eventId
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
  }, []);

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
      const url = process.env.NEXT_PUBLIC_OLLAMA_URL || "http://localhost:11434";
      const res = await fetch(`${url}/api/version`, { signal: AbortSignal.timeout(3000) });
      setOllamaStatus(res.ok ? "connected" : "disconnected");
    } catch {
      setOllamaStatus("disconnected");
    }
  }, []);

  // Initial data load
  useEffect(() => {
    fetchStories();
    fetchWatchlist();
    fetchEvents();
    fetchAlerts();
    checkOllama();
  }, [fetchStories, fetchWatchlist, fetchEvents, fetchAlerts, checkOllama]);

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
      case "WATCH":
        if (arg) handleAddTicker(arg);
        break;
      case "UNWATCH":
        if (arg) handleRemoveTicker(arg);
        break;
      case "QUOTE":
        if (arg) {
          // Add to watchlist temporarily to see price
          handleAddTicker(arg);
        }
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
        // Open chart for ticker: CHART AAPL
        if (arg) setSelectedTicker(arg.toUpperCase());
        break;
      case "ANALYZE":
        // Same as CHART
        if (arg) setSelectedTicker(arg.toUpperCase());
        break;
      case "BACK":
      case "CLOSE":
        // Close chart and return to dashboard
        setSelectedTicker(null);
        break;
      case "HELP":
        // Help is shown via the command line component
        break;
      default:
        // If it looks like a ticker symbol, open chart
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

  return (
    <div className="flex h-screen flex-col bg-terminal-bg font-mono">
      {/* Status Bar */}
      <StatusBar
        indices={PLACEHOLDER_INDICES.slice(0, 4).map((i) => ({
          symbol: i.symbol,
          price: i.price,
          changePct: i.changePct,
        }))}
        alertCount={alertCount}
        ollamaStatus={ollamaStatus}
        agentStatus={PLACEHOLDER_AGENT_STATUS.online ? "online" : "offline"}
        retrievalProviders={retrievalProviders}
        vix={engineData.regime.vix}
        regime={engineData.regime.regime}
      />

      {/* Main Grid */}
      {selectedTicker ? (
        /* Chart View: shows when a ticker is selected */
        <div className="flex-1 grid grid-cols-[1fr_350px] gap-[1px] bg-terminal-border p-[1px] overflow-hidden">
          <ChartPanel
            key={selectedTicker}
            ticker={selectedTicker}
            onClose={() => setSelectedTicker(null)}
          />
          <div className="grid grid-rows-2 gap-[1px]">
            <OptionsPanel
              trades={engineData.trades}
              regime={engineData.regime}
              portfolio={engineData.portfolio}
              connected={engineData.connected}
            />
            <ChatPanel initialQuery={chatQuery} />
          </div>
        </div>
      ) : (
        /* Default View: full terminal dashboard */
        <div className="flex-1 grid grid-cols-3 grid-rows-2 gap-[1px] bg-terminal-border p-[1px] overflow-hidden">
          {/* Row 1 */}
          <MarketOverview
            indices={PLACEHOLDER_INDICES}
            futures={PLACEHOLDER_FUTURES}
            commodities={PLACEHOLDER_COMMODITIES}
            loading={false}
          />
          <OptionsPanel
            trades={engineData.trades}
            regime={engineData.regime}
            portfolio={engineData.portfolio}
            connected={engineData.connected}
          />
          <AgentPanel
            status={PLACEHOLDER_AGENT_STATUS}
            tasks={PLACEHOLDER_AGENT_TASKS}
            connected={false}
          />

          {/* Row 2 */}
          <NewsPanel
            stories={stories}
            loading={storiesLoading}
            onRefresh={handleIngest}
            refreshing={ingesting}
            onSelectStory={handleSelectStory}
          />
          <WatchlistPanel
            items={watchlist}
            loading={watchlistLoading}
            onRefresh={fetchWatchlist}
            onAddTicker={handleAddTicker}
          />
          <div className="grid grid-rows-2 gap-[1px]">
            <MacroPanel events={events} loading={eventsLoading} />
            <ChatPanel initialQuery={chatQuery} />
          </div>
        </div>
      )}

      {/* Command Line */}
      <CommandLine onCommand={handleCommand} history={commandHistory} />
    </div>
  );
}
