"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Star,
  StarOff,
  TrendingUp,
  TrendingDown,
  ExternalLink,
  FileText,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { Quote, StoryCard, FilingSummary } from "@/types";

export default function TickerPage() {
  const params = useParams();
  const symbol = (params.symbol as string).toUpperCase();

  const [quote, setQuote] = useState<Quote | null>(null);
  const [stories, setStories] = useState<StoryCard[]>([]);
  const [filing, setFiling] = useState<FilingSummary | null>(null);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const [quoteRes, storiesRes, watchlistRes] = await Promise.allSettled([
        fetch(`/api/market?ticker=${symbol}`),
        fetch(`/api/stories?ticker=${symbol}`),
        fetch("/api/watchlist"),
      ]);

      if (quoteRes.status === "fulfilled" && quoteRes.value.ok) {
        setQuote(await quoteRes.value.json());
      }
      if (storiesRes.status === "fulfilled" && storiesRes.value.ok) {
        setStories(await storiesRes.value.json());
      }
      if (watchlistRes.status === "fulfilled" && watchlistRes.value.ok) {
        const wl = await watchlistRes.value.json();
        setInWatchlist(
          wl.some((w: { ticker: string }) => w.ticker === symbol)
        );
      }
      setLoading(false);
    }
    load();
  }, [symbol]);

  const toggleWatchlist = async () => {
    if (inWatchlist) {
      await fetch(`/api/watchlist?ticker=${symbol}`, { method: "DELETE" });
      setInWatchlist(false);
    } else {
      await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: symbol }),
      });
      setInWatchlist(true);
    }
  };

  // Mock price history for chart (in production, this would come from market data service)
  const priceHistory = quote
    ? Array.from({ length: 30 }, (_, i) => ({
        date: new Date(
          Date.now() - (29 - i) * 86400000
        ).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        price: +(
          quote.price *
          (1 + (Math.random() - 0.5) * 0.1)
        ).toFixed(2),
      }))
    : [];

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
            {symbol}
          </h1>
          {quote && (
            <div className="mt-1 flex items-center gap-3">
              <span className="text-3xl font-semibold">
                ${quote.price.toFixed(2)}
              </span>
              <Badge
                variant={quote.changePct >= 0 ? "positive" : "negative"}
                className="flex items-center gap-1"
              >
                {quote.changePct >= 0 ? (
                  <TrendingUp className="h-3 w-3" />
                ) : (
                  <TrendingDown className="h-3 w-3" />
                )}
                {quote.changePct >= 0 ? "+" : ""}
                {quote.changePct.toFixed(2)}%
              </Badge>
            </div>
          )}
        </div>
        <Button
          onClick={toggleWatchlist}
          variant={inWatchlist ? "default" : "outline"}
          size="sm"
        >
          {inWatchlist ? (
            <>
              <StarOff className="mr-2 h-4 w-4" />
              Remove from Watchlist
            </>
          ) : (
            <>
              <Star className="mr-2 h-4 w-4" />
              Add to Watchlist
            </>
          )}
        </Button>
      </div>

      {/* Price Chart */}
      {quote && priceHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Price History (30D)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11 }}
                    stroke="#71717a"
                  />
                  <YAxis
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 11 }}
                    stroke="#71717a"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#18181b",
                      border: "1px solid #3f3f46",
                      borderRadius: "8px",
                      color: "#fafafa",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Related News */}
      <Card>
        <CardHeader>
          <CardTitle>Related News</CardTitle>
          <CardDescription>
            Stories mentioning {symbol}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {stories.length === 0 ? (
            <p className="text-sm text-zinc-500">
              No stories found for {symbol}. Try refreshing feeds on the Feed
              page.
            </p>
          ) : (
            <div className="space-y-3">
              {stories.map((story) => (
                <div
                  key={story.storyId}
                  className="border-b border-zinc-100 pb-3 last:border-0 dark:border-zinc-800"
                >
                  <p className="text-sm font-medium">{story.canonicalTitle}</p>
                  {story.summary && (
                    <p className="mt-1 text-xs text-zinc-500">
                      {story.summary}
                    </p>
                  )}
                  <div className="mt-1 flex gap-2">
                    {story.sources.slice(0, 2).map((s, i) => (
                      <a
                        key={i}
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600"
                      >
                        <ExternalLink className="h-3 w-3" />
                        {s.publisher}
                      </a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* SEC Filing */}
      <Card>
        <CardHeader>
          <CardTitle>SEC Filings</CardTitle>
          <CardDescription>Latest filings from EDGAR</CardDescription>
        </CardHeader>
        <CardContent>
          {filing ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-zinc-400" />
                <span className="text-sm font-medium">{filing.title}</span>
              </div>
              <p className="text-xs text-zinc-500">{filing.summary}</p>
              <a
                href={filing.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-500 hover:underline"
              >
                View on EDGAR
              </a>
            </div>
          ) : (
            <p className="text-sm text-zinc-500">
              No recent filings loaded. Filing data is fetched on demand from
              SEC EDGAR.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Research Link */}
      <div className="pt-2">
        <Link href={`/research?ticker=${symbol}`}>
          <Button variant="outline">
            Research {symbol} in Chat
          </Button>
        </Link>
      </div>
    </div>
  );
}
