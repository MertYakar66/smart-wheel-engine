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

// /api/market labels each quote with its provenance; EOD quotes carry the
// close date and may lack a change% (single close available).
type LabeledQuote = Omit<Quote, "changePct"> & {
  changePct: number | null;
  source?: "live" | "snapshot" | "eod";
  asOf?: string;
};

export default function TickerPage() {
  const params = useParams();
  const symbol = (params.symbol as string).toUpperCase();

  const [quote, setQuote] = useState<LabeledQuote | null>(null);
  const [stories, setStories] = useState<StoryCard[]>([]);
  // EDGAR filing data is fetched on demand (not yet wired); the SEC card shows
  // an honest empty state until then.
  const [filing] = useState<FilingSummary | null>(null);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [loading, setLoading] = useState(true);
  const [priceHistory, setPriceHistory] = useState<
    { date: string; price: number }[]
  >([]);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const [quoteRes, storiesRes, watchlistRes, chartRes] =
        await Promise.allSettled([
          fetch(`/api/market?ticker=${symbol}`),
          fetch(`/api/stories?ticker=${symbol}`),
          fetch("/api/watchlist"),
          fetch(
            `/api/engine?action=chart&chart_type=bollinger&ticker=${symbol}&days=30`
          ),
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
      // Real 30-day daily closes from the engine OHLCV chart endpoint. If the
      // engine is unavailable we show an honest empty state below — never a
      // fabricated/random-walk series.
      if (chartRes.status === "fulfilled" && chartRes.value.ok) {
        try {
          const json = await chartRes.value.json();
          const rows: { date?: string; close?: number }[] = Array.isArray(
            json?.data
          )
            ? json.data
            : [];
          setPriceHistory(
            rows
              .filter(
                (r) =>
                  typeof r.close === "number" &&
                  Number.isFinite(r.close) &&
                  Boolean(r.date)
              )
              .map((r) => ({ date: r.date as string, price: r.close as number }))
          );
        } catch {
          setPriceHistory([]);
        }
      } else {
        setPriceHistory([]);
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
          {quote ? (
            <div className="mt-1 flex items-center gap-3">
              <span className="text-3xl font-semibold">
                ${quote.price.toFixed(2)}
              </span>
              {quote.changePct != null && (
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
              )}
              {/* Provenance: EOD closes and cached snapshots must never
                  pass as a live quote. */}
              {quote.source === "eod" && (
                <Badge variant="outline" className="border-amber-300 text-amber-600">
                  EOD · as of {quote.asOf}
                </Badge>
              )}
              {quote.source === "snapshot" && (
                <Badge variant="outline" className="border-zinc-300 text-zinc-500">
                  cached quote
                </Badge>
              )}
            </div>
          ) : (
            <p className="mt-1 text-sm text-zinc-500">
              No price available — engine and quote providers unreachable.
            </p>
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

      {/* Price Chart — real daily closes from the engine OHLCV feed. */}
      <Card>
        <CardHeader>
          <CardTitle>Price History (30D)</CardTitle>
          <CardDescription>Daily closes from the engine OHLCV feed</CardDescription>
        </CardHeader>
        <CardContent>
          {priceHistory.length > 0 ? (
            <div className="h-64">
              {/* Numeric height (matches h-64 = 256px) so recharts has a positive
                  dimension on the first paint, before its ResizeObserver runs —
                  avoids the "width(-1)/height(-1) ... greater than 0" warning. */}
              <ResponsiveContainer width="100%" height={256}>
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
          ) : (
            <p className="text-sm text-zinc-500">
              No price history available. Start the engine API (
              <code>python engine_api.py</code>) to load daily closes, or open
              the full chart in the{" "}
              <Link href="/terminal" className="text-blue-500 hover:underline">
                Terminal
              </Link>
              .
            </p>
          )}
        </CardContent>
      </Card>

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
