import { NextResponse } from "next/server";

/**
 * API bridge to the smart-wheel-engine Python API server.
 *
 * The Python API runs on port 8787 (started via: python engine_api.py)
 * and serves all engine data: candidates, analysis, regime, committee, etc.
 *
 * GET /api/engine?action=STATUS|candidates|analyze|regime|committee|calendar
 */

const ENGINE_API = process.env.ENGINE_API_URL || "http://localhost:8787";

async function fetchEngine(path: string): Promise<unknown> {
  const res = await fetch(`${ENGINE_API}${path}`, {
    headers: { "Content-Type": "application/json" },
    // Don't cache engine data
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Engine API returned ${res.status}`);
  }
  return res.json();
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get("action") || "status";

  try {
    switch (action) {
      case "status": {
        const data = await fetchEngine("/api/status");
        return NextResponse.json(data);
      }

      case "candidates": {
        const limit = searchParams.get("limit") || "15";
        const minScore = searchParams.get("min_score") || "50";
        const data = await fetchEngine(
          `/api/candidates?limit=${limit}&min_score=${minScore}`
        );
        return NextResponse.json(data);
      }

      case "analyze": {
        const ticker = searchParams.get("ticker") || "AAPL";
        const data = await fetchEngine(`/api/analyze/${ticker}`);
        return NextResponse.json(data);
      }

      case "regime": {
        const ticker = searchParams.get("ticker") || "SPY";
        const data = await fetchEngine(`/api/regime?ticker=${ticker}`);
        return NextResponse.json(data);
      }

      case "committee": {
        const ticker = searchParams.get("ticker") || "AAPL";
        const data = await fetchEngine(`/api/committee?ticker=${ticker}`);
        return NextResponse.json(data);
      }

      case "calendar": {
        const ticker = searchParams.get("ticker") || "";
        const days = searchParams.get("days") || "30";
        const data = await fetchEngine(
          `/api/calendar?ticker=${ticker}&days=${days}`
        );
        return NextResponse.json(data);
      }

      case "portfolio": {
        const tickers = searchParams.get("tickers") || "AAPL,MSFT,JPM";
        const data = await fetchEngine(`/api/portfolio?tickers=${tickers}`);
        return NextResponse.json(data);
      }

      case "vix": {
        const data = await fetchEngine("/api/vix");
        return NextResponse.json(data);
      }

      case "fundamentals": {
        const ticker = searchParams.get("ticker") || "AAPL";
        const data = await fetchEngine(`/api/fundamentals?ticker=${ticker}`);
        return NextResponse.json(data);
      }

      case "screen": {
        const limit = searchParams.get("limit") || "20";
        const minScore = searchParams.get("min_score") || "50";
        const data = await fetchEngine(
          `/api/screen?min_score=${minScore}&limit=${limit}`
        );
        return NextResponse.json(data);
      }

      case "universe": {
        const data = await fetchEngine("/api/universe");
        return NextResponse.json(data);
      }

      default:
        return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("Engine API error:", message);
    return NextResponse.json(
      {
        error: "Engine unavailable",
        detail: message,
        hint: "Start the Python API server: python engine_api.py",
      },
      { status: 503 }
    );
  }
}
