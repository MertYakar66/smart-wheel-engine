import { NextResponse } from "next/server";
import { fetchQuoteFromFinnhub, getLatestSnapshot } from "@/services/market-data";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const ticker = url.searchParams.get("ticker");

  if (!ticker) {
    return NextResponse.json(
      { error: "ticker parameter required" },
      { status: 400 }
    );
  }

  // Try live quote first, fall back to cached
  const quote =
    (await fetchQuoteFromFinnhub(ticker)) ||
    (await getLatestSnapshot(ticker));

  if (!quote) {
    return NextResponse.json(
      { error: "No data available for ticker" },
      { status: 404 }
    );
  }

  return NextResponse.json(quote);
}
