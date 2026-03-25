import { NextResponse } from "next/server";
import { db } from "@/db";
import { watchlists } from "@/db/schema";
import { eq } from "drizzle-orm";
import { fetchQuoteFromFinnhub, getLatestSnapshot } from "@/services/market-data";

export async function GET() {
  const items = await db.query.watchlists.findMany();

  const enriched = await Promise.all(
    items.map(async (item) => {
      const quote =
        (await getLatestSnapshot(item.ticker)) ||
        (await fetchQuoteFromFinnhub(item.ticker));
      return {
        ticker: item.ticker,
        addedAt: item.addedAt,
        alertThresholdPct: item.alertThresholdPct,
        price: quote?.price || null,
        changePct: quote?.changePct || null,
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
