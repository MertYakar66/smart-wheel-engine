import { NextResponse } from "next/server";
import { db } from "@/db";
import { watchlists } from "@/db/schema";
import { eq } from "drizzle-orm";
import { fetchQuoteFromFinnhub, getLatestSnapshot } from "@/services/market-data";

const ENGINE_API = process.env.ENGINE_API_URL || "http://localhost:8787";

async function getEnginePrice(
  ticker: string
): Promise<{ price: number; changePct: number } | null> {
  try {
    // ONE cheap engine call per ticker: the last two OHLCV closes give both
    // the EOD spot and the 1-day change. (This used to also hit /api/analyze
    // — a full per-ticker analysis — per watchlist row, doubling the calls
    // for a number the ohlcv read already contains.)
    const res = await fetch(
      `${ENGINE_API}/api/chart/ohlcv?ticker=${ticker}&days=2`,
      { cache: "no-store", signal: AbortSignal.timeout(5000) }
    );
    if (!res.ok) return null;
    const body = await res.json();
    const rows = body?.data || [];
    const curr = rows[rows.length - 1]?.close;
    if (typeof curr !== "number" || curr <= 0) return null;
    const prev = rows[rows.length - 2]?.close;
    const changePct =
      typeof prev === "number" && prev > 0 ? ((curr - prev) / prev) * 100 : 0;
    return { price: curr, changePct };
  } catch {
    return null;
  }
}

export async function GET() {
  const items = await db.query.watchlists.findMany();

  const enriched = await Promise.all(
    items.map(async (item) => {
      // Resolution order: cached snapshot -> engine spot (fast, always on)
      // -> Finnhub (skipped unless FINNHUB_API_KEY is set).
      const quote =
        (await getLatestSnapshot(item.ticker)) ||
        (await getEnginePrice(item.ticker)) ||
        (await fetchQuoteFromFinnhub(item.ticker));
      return {
        ticker: item.ticker,
        addedAt: item.addedAt,
        alertThresholdPct: item.alertThresholdPct,
        price: quote?.price ?? null,
        changePct: quote?.changePct ?? null,
      };
    })
  );

  return NextResponse.json(enriched);
}

export async function POST(request: Request) {
  const body = await request.json();
  const { ticker, alertThresholdPct } = body;

  if (!ticker || typeof ticker !== "string") {
    return NextResponse.json(
      { error: "ticker is required" },
      { status: 400 }
    );
  }

  await db
    .insert(watchlists)
    .values({
      ticker: ticker.toUpperCase(),
      addedAt: new Date().toISOString(),
      alertThresholdPct: alertThresholdPct || 5,
    })
    .onConflictDoNothing();

  return NextResponse.json({ success: true, ticker: ticker.toUpperCase() });
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const ticker = url.searchParams.get("ticker");

  if (!ticker) {
    return NextResponse.json(
      { error: "ticker parameter required" },
      { status: 400 }
    );
  }

  await db.delete(watchlists).where(eq(watchlists.ticker, ticker.toUpperCase()));

  return NextResponse.json({ success: true });
}
