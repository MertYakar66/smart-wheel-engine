"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  RefreshCw,
  ExternalLink,
  Clock,
  Zap,
  AlertTriangle,
  Layers,
  TrendingUp,
  BarChart3,
  Newspaper,
  Sun,
  Moon,
  ChevronRight,
} from "lucide-react";
import type { StoryCard } from "@/types";

// ─── Bloomberg-style TOP page ─────────────────────────────────────────
// Curated command center with:
//  - Breaking strip (First Word)
//  - Top stories (ranked by impact + exposure)
//  - Category spotlight sections
//  - Morning/Evening briefing link

interface CategoryWithStories {
  category: {
    categoryId: string;
    name: string;
    color: string;
    icon: string;
    storyCount: number;
  };
  stories: StoryCard[];
}

interface BriefingSummary {
  briefingId: string;
  briefingType: string;
  title: string;
  generatedAt: string;
  totalStories: number;
}

const IMPACT_TAG_COLORS: Record<string, string> = {
  rates: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  oil: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  fx: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  regulation: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  earnings: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  geopolitical: "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300",
  monetary_policy: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
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

function MiniStoryCard({ story }: { story: StoryCard }) {
  return (
    <Link href={`/story/${story.storyId}`}>
      <div className="rounded-lg border border-zinc-200 p-3 transition-all hover:shadow-md hover:border-zinc-300 dark:border-zinc-700 dark:hover:border-zinc-600 cursor-pointer">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-sm font-medium text-zinc-900 dark:text-zinc-100 leading-snug line-clamp-2">
            {story.canonicalTitle}
          </h3>
          <span className="text-[10px] text-zinc-400 whitespace-nowrap">{timeAgo(story.createdAt)}</span>
        </div>

        {story.whyItMatters && (
          <p className="mt-1 text-xs text-blue-700 dark:text-blue-400 line-clamp-2">
            {story.whyItMatters}
          </p>
        )}

        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
          {story.entities
            .filter((e) => e.entityType === "ticker")
            .slice(0, 3)
            .map((e, idx) => (
              <Badge key={`${e.entityValue}-${idx}`} variant="ticker" className="text-[10px] px-1.5 py-0">
                ${e.entityValue}
              </Badge>
            ))}
          {story.impactTags?.slice(0, 2).map((tag) => (
            <span
              key={tag}
              className={`inline-flex items-center rounded-full px-1.5 py-0 text-[9px] font-medium ${IMPACT_TAG_COLORS[tag] || "bg-zinc-100 text-zinc-600"}`}
            >
              {tag.replace("_", " ")}
            </span>
          ))}
          {story.sourceCount > 1 && (
            <span className="text-[10px] text-zinc-400">
              <Layers className="inline h-2.5 w-2.5 mr-0.5" />
              {story.sourceCount}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

export default function TopPage() {
  const [topStories, setTopStories] = useState<StoryCard[]>([]);
  const [breakingStories, setBreakingStories] = useState<StoryCard[]>([]);
  const [categories, setCategories] = useState<{ categoryId: string; name: string; color: string; icon: string; storyCount: number }[]>([]);
  const [categoryStories, setCategoryStories] = useState<Record<string, StoryCard[]>>({});
  const [briefing, setBriefing] = useState<BriefingSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch top stories (exposure-ranked)
      const storiesRes = await fetch("/api/stories?limit=20");
      const storiesData = await storiesRes.json();
      setTopStories(storiesData);

      // Breaking = stories from last 2 hours with high impact
      const breaking = storiesData.filter((s: StoryCard) => {
        const age = Date.now() - new Date(s.createdAt).getTime();
        return age < 2 * 60 * 60 * 1000 && s.impactScore > 1;
      });
      setBreakingStories(breaking.slice(0, 5));

      // Fetch categories
      const catRes = await fetch("/api/categories");
      const catData = await catRes.json();
      setCategories(catData);

      // Fetch stories for top 4 categories
      const topCats = catData
        .filter((c: { storyCount: number }) => c.storyCount > 0)
        .slice(0, 4);

      const catStoriesMap: Record<string, StoryCard[]> = {};
      for (const cat of topCats) {
        try {
          const res = await fetch(`/api/categories?id=${cat.categoryId}&stories=true`);
          const data = await res.json();
          catStoriesMap[cat.categoryId] = data.stories?.slice(0, 4) || [];
        } catch {
          catStoriesMap[cat.categoryId] = [];
        }
      }
      setCategoryStories(catStoriesMap);

      // Fetch latest briefing
      try {
        const briefRes = await fetch("/api/briefings");
        if (briefRes.ok) {
          const briefData = await briefRes.json();
          setBriefing(briefData);
        }
      } catch {
        // No briefing yet
      }
    } catch (err) {
      console.error("Failed to fetch TOP data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetch("/api/ingest", { method: "POST" });
      await fetchData();
    } catch (err) {
      console.error("Refresh failed:", err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleGenerateBriefing = async (type: "morning" | "evening") => {
    try {
      const res = await fetch("/api/briefings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type }),
      });
      if (res.ok) {
        const data = await res.json();
        setBriefing(data);
      }
    } catch (err) {
      console.error("Briefing generation failed:", err);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  const isMarketHours = (() => {
    const now = new Date();
    const hour = now.getHours();
    return hour >= 9 && hour < 16;
  })();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
            TOP
          </h1>
          <p className="text-sm text-zinc-500">
            {topStories.length} stories · {categories.length} categories
            {isMarketHours && <span className="ml-2 text-green-600">Market Open</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleGenerateBriefing("morning")}
          >
            <Sun className="mr-1 h-3 w-3" />
            Morning Brief
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleGenerateBriefing("evening")}
          >
            <Moon className="mr-1 h-3 w-3" />
            Evening Brief
          </Button>
          <Button
            onClick={handleRefresh}
            disabled={refreshing}
            variant="outline"
            size="sm"
          >
            <RefreshCw className={`mr-1 h-3 w-3 ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </div>

      {/* Breaking Strip (First Word) */}
      {breakingStories.length > 0 && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 dark:bg-red-950/20 dark:border-red-800">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="h-4 w-4 text-red-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-red-600">Breaking</span>
          </div>
          <div className="space-y-1">
            {breakingStories.map((story) => (
              <Link key={story.storyId} href={`/story/${story.storyId}`}>
                <div className="flex items-center gap-2 py-1 hover:bg-red-100/50 dark:hover:bg-red-900/20 rounded px-1 cursor-pointer">
                  <span className="text-red-500 text-xs">●</span>
                  <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100 flex-1 truncate">
                    {story.canonicalTitle}
                  </span>
                  <span className="text-[10px] text-zinc-400 whitespace-nowrap">
                    {timeAgo(story.createdAt)}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Briefing Card */}
      {briefing && (
        <Card className="border-blue-200 bg-blue-50/30 dark:border-blue-800 dark:bg-blue-950/20">
          <CardContent className="py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Newspaper className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-blue-900 dark:text-blue-200">{briefing.title}</p>
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  {briefing.totalStories} stories · Generated {timeAgo(briefing.generatedAt)}
                </p>
              </div>
            </div>
            <Link href="/feed">
              <Button variant="ghost" size="sm" className="text-blue-600">
                View <ChevronRight className="ml-1 h-3 w-3" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* Top Stories Grid */}
      <div>
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
          Top Stories
        </h2>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {topStories.slice(0, 9).map((story) => (
            <MiniStoryCard key={story.storyId} story={story} />
          ))}
        </div>
      </div>

      {/* Category Sections (Bloomberg MYN-style) */}
      {categories
        .filter((c) => categoryStories[c.categoryId]?.length > 0)
        .slice(0, 4)
        .map((cat) => (
          <div key={cat.categoryId}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: cat.color || "#6B7280" }}
                />
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                  {cat.name}
                </h2>
                <Badge variant="outline" className="text-xs">
                  {cat.storyCount}
                </Badge>
              </div>
              <Link href={`/feed?category=${cat.categoryId}`}>
                <Button variant="ghost" size="sm" className="text-xs">
                  View All <ChevronRight className="ml-1 h-3 w-3" />
                </Button>
              </Link>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {(categoryStories[cat.categoryId] || []).map((story) => (
                <MiniStoryCard key={story.storyId} story={story} />
              ))}
            </div>
          </div>
        ))}

      {/* Quick Links */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Link href="/feed">
          <Card className="transition-shadow hover:shadow-md cursor-pointer">
            <CardContent className="py-4 text-center">
              <Newspaper className="h-5 w-5 mx-auto mb-1 text-zinc-600" />
              <p className="text-sm font-medium">Full Feed</p>
            </CardContent>
          </Card>
        </Link>
        <Link href="/watchlist">
          <Card className="transition-shadow hover:shadow-md cursor-pointer">
            <CardContent className="py-4 text-center">
              <TrendingUp className="h-5 w-5 mx-auto mb-1 text-zinc-600" />
              <p className="text-sm font-medium">Watchlist</p>
            </CardContent>
          </Card>
        </Link>
        <Link href="/research">
          <Card className="transition-shadow hover:shadow-md cursor-pointer">
            <CardContent className="py-4 text-center">
              <BarChart3 className="h-5 w-5 mx-auto mb-1 text-zinc-600" />
              <p className="text-sm font-medium">Research</p>
            </CardContent>
          </Card>
        </Link>
        <Link href="/terminal">
          <Card className="transition-shadow hover:shadow-md cursor-pointer">
            <CardContent className="py-4 text-center">
              <Zap className="h-5 w-5 mx-auto mb-1 text-zinc-600" />
              <p className="text-sm font-medium">Terminal</p>
            </CardContent>
          </Card>
        </Link>
      </div>
    </div>
  );
}
