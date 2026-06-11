"use client";

import { useEffect, useState, useCallback, useRef, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  RefreshCw,
  ExternalLink,
  Clock,
  Layers,
  Filter,
  AlertTriangle,
  CheckCircle,
  GitBranch,
  Zap,
  TrendingUp,
  BarChart3,
  X,
} from "lucide-react";
import type { StoryCard } from "@/types";

const SECTORS = [
  "All",
  "Technology",
  "Financials",
  "Energy",
  "Healthcare",
  "Consumer",
  "Macro",
  "General",
];

const IMPACT_TAG_COLORS: Record<string, string> = {
  rates: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  oil: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  fx: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  regulation: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  earnings: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  demand: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300",
  supply_chain: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  geopolitical: "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300",
  monetary_policy: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
  labor: "bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300",
};

const STATUS_BADGE: Record<string, { icon: typeof Clock; label: string; className: string }> = {
  developing: { icon: Clock, label: "Developing", className: "border-blue-300 text-blue-600" },
  evolving: { icon: GitBranch, label: "Evolving", className: "border-amber-300 text-amber-600" },
  resolved: { icon: CheckCircle, label: "Resolved", className: "border-green-300 text-green-600" },
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

interface LastRunInfo {
  startedAt: string;
  headlinesIngested: number | null;
  status: string | null;
}

function FeedPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // Category filter arrives via /feed?category=<id> (the /top "View All"
  // links); it is served by the categories endpoint, not the stories one.
  const categoryId = searchParams.get("category");

  const [stories, setStories] = useState<StoryCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [selectedSector, setSelectedSector] = useState("All");
  const [universeOnly, setUniverseOnly] = useState(false);
  const [universeUnavailable, setUniverseUnavailable] = useState(false);
  const [categoryName, setCategoryName] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<LastRunInfo | null>(null);
  const [streamLive, setStreamLive] = useState(false);

  const fetchLastRun = useCallback(async () => {
    try {
      const res = await fetch("/api/schedule");
      if (!res.ok) return;
      const data = await res.json();
      if (data?.startedAt) {
        setLastRun({
          startedAt: data.startedAt,
          headlinesIngested: data.headlinesIngested ?? null,
          status: data.status ?? null,
        });
      } else {
        setLastRun(null);
      }
    } catch {
      /* schedule info is decorative — leave whatever we had */
    }
  }, []);

  const fetchStories = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        if (categoryId) {
          const res = await fetch(
            `/api/categories?id=${encodeURIComponent(categoryId)}&stories=true`
          );
          if (res.ok) {
            const data = await res.json();
            setStories(data.stories || []);
            setCategoryName(data.category?.name || categoryId);
          } else {
            setStories([]);
            setCategoryName(null);
          }
          setUniverseUnavailable(false);
        } else {
          const params = new URLSearchParams({ limit: "50" });
          if (selectedSector !== "All") {
            params.set("sector", selectedSector);
          }
          if (universeOnly) {
            params.set("universe", "true");
          }
          const res = await fetch(`/api/stories?${params}`);
          const data = await res.json();
          setStories(data);
          setUniverseUnavailable(
            universeOnly &&
              res.headers.get("x-universe-filter") === "unavailable"
          );
        }
      } catch (err) {
        console.error("Failed to fetch stories:", err);
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [selectedSector, universeOnly, categoryId]
  );

  useEffect(() => {
    fetchStories();
    fetchLastRun();
  }, [fetchStories, fetchLastRun]);

  // Live headline stream (SSE): silently refresh when new stories land so
  // scheduled ingestion shows up without a manual reload.
  const fetchStoriesRef = useRef(fetchStories);
  fetchStoriesRef.current = fetchStories;
  useEffect(() => {
    const es = new EventSource("/api/stream");
    const onStory = () => fetchStoriesRef.current(true);
    es.addEventListener("story", onStory);
    es.onopen = () => setStreamLive(true);
    es.onerror = () => setStreamLive(false);
    return () => {
      es.removeEventListener("story", onStory);
      es.close();
    };
  }, []);

  const handleIngest = async () => {
    setIngesting(true);
    try {
      const res = await fetch("/api/ingest", { method: "POST" });
      const data = await res.json();
      console.log("Ingestion result:", data);
      await Promise.all([fetchStories(), fetchLastRun()]);
    } catch (err) {
      console.error("Ingestion failed:", err);
    } finally {
      setIngesting(false);
    }
  };

  const clearCategory = () => {
    setCategoryName(null);
    router.replace("/feed");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold text-zinc-900 dark:text-zinc-50">
            News Feed
            {streamLive && (
              <span
                className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-green-600"
                title="Live headline stream connected"
              >
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
                live
              </span>
            )}
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            {stories.length} stories from financial news sources
            {stories.some((s) => s.exposureRelevance !== undefined && s.exposureRelevance! > 0) && (
              <span className="ml-1 text-blue-600">(ranked by your exposures)</span>
            )}
            {" · "}
            {lastRun ? (
              <>
                last ingest {timeAgo(lastRun.startedAt)}
                {lastRun.headlinesIngested != null && (
                  <> · {lastRun.headlinesIngested} headlines</>
                )}
                {lastRun.status === "failed" && (
                  <span className="text-red-500"> · failed</span>
                )}
              </>
            ) : (
              <span className="text-amber-600">no ingestion has run yet</span>
            )}
          </p>
        </div>
        <Button
          onClick={handleIngest}
          disabled={ingesting}
          variant="outline"
          size="sm"
        >
          <RefreshCw
            className={`mr-2 h-4 w-4 ${ingesting ? "animate-spin" : ""}`}
          />
          {ingesting ? "Ingesting..." : "Ingest Now"}
        </Button>
      </div>

      {/* Category filter chip (from /top "View All") */}
      {categoryId ? (
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-zinc-400" />
          <Badge variant="secondary" className="flex items-center gap-1.5 pr-1">
            Category: {categoryName || categoryId}
            <button
              onClick={clearCategory}
              aria-label="Clear category filter"
              className="rounded-full p-0.5 hover:bg-zinc-300/50 dark:hover:bg-zinc-600/50"
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
        </div>
      ) : (
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <Filter className="h-4 w-4 text-zinc-400" />
          {SECTORS.map((sector) => (
            <Button
              key={sector}
              variant={selectedSector === sector ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedSector(sector)}
              className="whitespace-nowrap"
            >
              {sector}
            </Button>
          ))}
          <span className="mx-1 h-4 w-px bg-zinc-200 dark:bg-zinc-700" />
          <Button
            variant={universeOnly ? "default" : "outline"}
            size="sm"
            aria-pressed={universeOnly}
            onClick={() => setUniverseOnly((v) => !v)}
            className="whitespace-nowrap"
            title="Only stories tagged with a validated S&P 500 (or held-book) ticker"
          >
            S&amp;P universe only
          </Button>
          {universeUnavailable && (
            <span className="whitespace-nowrap text-xs text-amber-600">
              filter unavailable — engine offline
            </span>
          )}
        </div>
      )}

      {/* Story Cards */}
      {loading ? (
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-3/4" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-full" />
                <Skeleton className="mt-2 h-4 w-2/3" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : stories.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-12 text-center">
            <p className="text-zinc-500 dark:text-zinc-400">
              {categoryId
                ? "No stories matched this category yet."
                : universeOnly
                  ? "No stories with validated S&P-universe tickers yet. Disable the universe filter or ingest more headlines."
                  : "No stories yet. Scheduled ingestion runs at 06:30/18:30 ET (plus market-hours refreshes), or click \"Ingest Now\"."}
            </p>
            <Button onClick={handleIngest} disabled={ingesting} size="sm">
              <RefreshCw
                className={`mr-2 h-4 w-4 ${ingesting ? "animate-spin" : ""}`}
              />
              {ingesting ? "Ingesting..." : "Ingest Now"}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {stories.map((story) => {
            const statusInfo = STATUS_BADGE[story.storyStatus || "developing"] || STATUS_BADGE.developing;
            const StatusIcon = statusInfo.icon;

            return (
              <Card
                key={story.storyId}
                className="transition-shadow hover:shadow-md"
              >
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <Link href={`/story/${story.storyId}`}>
                        <CardTitle className="text-base leading-snug hover:text-blue-600 cursor-pointer">
                          {story.canonicalTitle}
                        </CardTitle>
                      </Link>
                    </div>
                    <div className="flex shrink-0 items-center gap-2 text-xs text-zinc-400">
                      {story.exposureRelevance !== undefined && story.exposureRelevance > 0 && (
                        <Badge variant="outline" className="border-blue-300 text-blue-600">
                          <TrendingUp className="mr-1 h-3 w-3" />
                          {story.exposureRelevance.toFixed(1)}
                        </Badge>
                      )}
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {timeAgo(story.createdAt)}
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Why It Matters (if available) */}
                  {story.whyItMatters && (
                    <div className="rounded-md bg-blue-50 px-3 py-2 text-xs text-blue-800 dark:bg-blue-950/30 dark:text-blue-300">
                      <span className="font-semibold">Why it matters:</span> {story.whyItMatters}
                    </div>
                  )}

                  {story.summary && !story.whyItMatters && (
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">
                      {story.summary}
                    </p>
                  )}

                  {/* Impact tags */}
                  {story.impactTags && story.impactTags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {story.impactTags.map((tag) => (
                        <span
                          key={tag}
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${IMPACT_TAG_COLORS[tag] || "bg-zinc-100 text-zinc-800"}`}
                        >
                          <Zap className="mr-0.5 h-2.5 w-2.5" />
                          {tag.replace("_", " ")}
                        </span>
                      ))}
                      {story.impactHorizon && (
                        <span className="inline-flex items-center rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                          {story.impactHorizon}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Entity tags + status badges */}
                  <div className="flex flex-wrap gap-1.5">
                    {story.entities
                      .filter((e) => e.entityType === "ticker")
                      .map((e, idx) => (
                        <Link
                          key={`${e.entityValue}-${idx}`}
                          href={`/ticker/${e.entityValue}`}
                        >
                          <Badge variant="ticker" className="cursor-pointer">
                            ${e.entityValue}
                          </Badge>
                        </Link>
                      ))}
                    {story.sector && (
                      <Badge variant="secondary">{story.sector}</Badge>
                    )}
                    {story.sourceCount > 1 && (
                      <Badge variant="outline">
                        <Layers className="mr-1 h-3 w-3" />
                        {story.sourceCount} sources
                      </Badge>
                    )}
                    {story.storyStatus && story.storyStatus !== "developing" && (
                      <Badge variant="outline" className={statusInfo.className}>
                        <StatusIcon className="mr-1 h-3 w-3" />
                        {statusInfo.label}
                      </Badge>
                    )}
                    {story.contradictionFlag && (
                      <Badge variant="destructive" className="text-xs">
                        <AlertTriangle className="mr-1 h-3 w-3" />
                        Contradicting
                      </Badge>
                    )}
                    {story.corroborationScore !== undefined && story.corroborationScore > 0.5 && (
                      <Badge variant="outline" className="border-green-300 text-green-600 text-xs">
                        <CheckCircle className="mr-1 h-3 w-3" />
                        Corroborated
                      </Badge>
                    )}
                  </div>

                  {/* Sources */}
                  {story.sources.length > 0 && (
                    <div className="flex flex-wrap gap-2 text-xs">
                      {story.sources.slice(0, 3).map((source, idx) => (
                        <a
                          key={idx}
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200"
                        >
                          <ExternalLink className="h-3 w-3" />
                          {source.publisher}
                        </a>
                      ))}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div className="flex gap-2 pt-1">
                    <Link href={`/story/${story.storyId}`}>
                      <Button variant="ghost" size="sm" className="text-xs">
                        View Story
                      </Button>
                    </Link>
                    <Link
                      href={`/research?story=${story.storyId}&title=${encodeURIComponent(story.canonicalTitle)}`}
                    >
                      <Button variant="ghost" size="sm" className="text-xs">
                        <BarChart3 className="mr-1 h-3 w-3" />
                        Research
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

// useSearchParams requires a Suspense boundary for prerendering.
export default function FeedPage() {
  return (
    <Suspense
      fallback={
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      }
    >
      <FeedPageInner />
    </Suspense>
  );
}
