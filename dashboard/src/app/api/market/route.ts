import { NextResponse } from "next/server";
import {
  fetchQuoteFromFinnhub,
  getLatestSnapshot,
  fetchEngineEodQuote,
} from "@/services/market-data";

// Quote resolution chain, each result labeled with its provenance so the
// UI can render "live" vs "cached" vs "EOD as of <date>" honestly:
//   1. Finnhub (realtime; only when FINNHUB_API_KEY is set)
//   2. cached snapshot (<24h old, written by a previous Finnhub hit)
//   3. engine EOD close (always available while the engine runs)

export async function GET(request: Request) {
  const url = new URL(request.url);
  const ticker = url.searchParams.get("ticker");

  if (!ticker) {
    return NextResponse.json(
      { error: "ticker parameter required" },
      { status: 400 }
    );
  }

  const live = await fetchQuoteFromFinnhub(ticker);
  if (live) {
    return NextResponse.json({ ...live, source: "live" });
  }

  const snapshot = await getLatestSnapshot(ticker);
  if (snapshot) {
    return NextResponse.json({ ...snapshot, source: "snapshot" });
  }

  const eod = await fetchEngineEodQuote(ticker);
  if (eod) {
    return NextResponse.json({
      ticker: eod.ticker,
      price: eod.price,
      changePct: eod.changePct,
      volume: eod.volume,
      capturedAt: eod.asOf,
      source: "eod",
      asOf: eod.asOf,
    });
  }

  return NextResponse.json(
    { error: "No data available for ticker" },
    { status: 404 }
  );
}
