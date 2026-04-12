import { v4 as uuidv4 } from "uuid";
import { db } from "@/db";
import { marketSnapshots } from "@/db/schema";
import type { Quote } from "@/types";
import { eq, desc } from "drizzle-orm";

const FINNHUB_BASE = "https://finnhub.io/api/v1";

// Finnhub free tier: 60 calls/min, no API key required for basic quotes
// If you have a key, set FINNHUB_API_KEY env var
function getApiKey(): string {
  return process.env.FINNHUB_API_KEY || "";
}

export async function fetchQuoteFromFinnhub(
  ticker: string
): Promise<Quote | null> {
  try {
    const key = getApiKey();
    // Finnhub's quote endpoint now requires a key. Skip entirely if none
    // is configured so the watchlist doesn't wait 10s on a sure failure.
    if (!key) return null;
    const url = `${FINNHUB_BASE}/quote?symbol=${encodeURIComponent(ticker)}&token=${key}`;

    const res = await fetch(url, {
      headers: { "User-Agent": "FinanceNewsPlatform/1.0" },
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) return null;

    const data = await res.json();

    // Finnhub returns { c: current, d: change, dp: change%, h: high, l: low, o: open, pc: prev close, t: timestamp }
    if (!data.c || data.c === 0) return null;

    const now = new Date().toISOString();
    const quote: Quote = {
      ticker,
      price: data.c,
      changePct: data.dp || 0,
      volume: 0, // Volume not in basic quote endpoint
      capturedAt: now,
    };

    // Store snapshot
    await db.insert(marketSnapshots).values({
      snapshotId: uuidv4(),
      ticker,
      price: quote.price,
      changePct: quote.changePct,
      volume: quote.volume,
      capturedAt: now,
    });

    return quote;
  } catch {
    console.error(`Failed to fetch quote for ${ticker}`);
    return null;
  }
}

export async function getLatestSnapshot(
  ticker: string
): Promise<Quote | null> {
  const result = await db.query.marketSnapshots.findFirst({
    where: eq(marketSnapshots.ticker, ticker),
    orderBy: [desc(marketSnapshots.capturedAt)],
  });

  // Reject empty / zero-price rows so the caller falls through to a fresh
  // engine/Finnhub fetch instead of rendering a meaningless "$0".
  if (!result || !result.price || result.price <= 0) return null;
  // Also reject snapshots older than 24h — wheel scanning needs fresh data.
  const captured = new Date(result.capturedAt).getTime();
  if (Number.isFinite(captured) && Date.now() - captured > 24 * 3600 * 1000) {
    return null;
  }

  return {
    ticker: result.ticker,
    price: result.price,
    changePct: result.changePct || 0,
    volume: result.volume || 0,
    capturedAt: result.capturedAt,
  };
}

export async function fetchQuotesForWatchlist(): Promise<Quote[]> {
  const watchlistItems = await db.query.watchlists.findMany();
  const quotes: Quote[] = [];

  for (const item of watchlistItems) {
    const quote = await fetchQuoteFromFinnhub(item.ticker);
    if (quote) quotes.push(quote);
    // Rate limit: small delay between calls
    await new Promise((r) => setTimeout(r, 200));
  }

  return quotes;
}
