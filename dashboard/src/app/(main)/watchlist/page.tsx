"use client";

import { useEffect, useState, useCallback } from "react";
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
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Plus,
  Trash2,
  TrendingUp,
  TrendingDown,
  RefreshCw,
} from "lucide-react";

interface WatchlistItem {
  ticker: string;
  addedAt: string;
  alertThresholdPct: number;
  price: number | null;
  changePct: number | null;
}

export default function WatchlistPage() {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [newTicker, setNewTicker] = useState("");
  const [adding, setAdding] = useState(false);

  const fetchWatchlist = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/watchlist");
      const data = await res.json();
      setItems(data);
    } catch (err) {
      console.error("Failed to fetch watchlist:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  const addTicker = async () => {
    const ticker = newTicker.trim().toUpperCase();
    if (!ticker) return;

    setAdding(true);
    try {
      await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      setNewTicker("");
      await fetchWatchlist();
    } catch (err) {
      console.error("Failed to add ticker:", err);
    } finally {
      setAdding(false);
    }
  };

  const removeTicker = async (ticker: string) => {
    try {
      await fetch(`/api/watchlist?ticker=${ticker}`, {
        method: "DELETE",
      });
      setItems((prev) => prev.filter((i) => i.ticker !== ticker));
    } catch (err) {
      console.error("Failed to remove ticker:", err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
            Watchlist
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Track tickers and get alerts on related news
          </p>
        </div>
        <Button onClick={fetchWatchlist} variant="outline" size="sm">
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh Prices
        </Button>
      </div>

      {/* Add Ticker */}
      <Card>
        <CardContent className="flex items-center gap-3 pt-6">
          <Input
            placeholder="Enter ticker symbol (e.g. AAPL)"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && addTicker()}
            className="max-w-xs"
          />
          <Button onClick={addTicker} disabled={adding} size="sm">
            <Plus className="mr-2 h-4 w-4" />
            Add
          </Button>
        </CardContent>
      </Card>

      {/* Watchlist Items */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <Card key={i}>
              <CardContent className="pt-6">
                <Skeleton className="h-6 w-24" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : items.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-zinc-500 dark:text-zinc-400">
              Your watchlist is empty. Add tickers above to start tracking.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <Card
              key={item.ticker}
              className="transition-shadow hover:shadow-md"
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <Link href={`/ticker/${item.ticker}`}>
                    <CardTitle className="cursor-pointer text-lg hover:text-blue-600">
                      {item.ticker}
                    </CardTitle>
                  </Link>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeTicker(item.ticker)}
                    className="h-8 w-8 text-zinc-400 hover:text-red-500"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {item.price !== null ? (
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-semibold">
                      ${item.price.toFixed(2)}
                    </span>
                    {item.changePct !== null && (
                      <Badge
                        variant={
                          item.changePct >= 0 ? "positive" : "negative"
                        }
                        className="flex items-center gap-1"
                      >
                        {item.changePct >= 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : (
                          <TrendingDown className="h-3 w-3" />
                        )}
                        {item.changePct >= 0 ? "+" : ""}
                        {item.changePct.toFixed(2)}%
                      </Badge>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-zinc-400">
                    No price data. Set FINNHUB_API_KEY for live quotes.
                  </p>
                )}
                <p className="mt-2 text-xs text-zinc-400">
                  Alert threshold: {item.alertThresholdPct}%
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
