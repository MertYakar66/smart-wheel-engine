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
        // The decision cockpit forwards the full PIT parameter set. Defaults
        // mirror the engine's own (min_score=0 so the heuristic filter never
        // gates the EV ranking; min_ev=0 = the tradeable proceed+review set).
        const limit = searchParams.get("limit") || "15";
        const minScore = searchParams.get("min_score") || "0";
        const dte = searchParams.get("dte") || "35";
        const delta = searchParams.get("delta") || "0.25";
        const minEv = searchParams.get("min_ev") || "0";
        const asOf = searchParams.get("as_of") || "";
        const universeLimit = searchParams.get("universe_limit") || "";
        const qs = new URLSearchParams({
          limit,
          min_score: minScore,
          dte,
          delta,
          min_ev: minEv,
        });
        if (asOf) qs.set("as_of", asOf);
        if (universeLimit) qs.set("universe_limit", universeLimit);
        const data = await fetchEngine(`/api/candidates?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "dossier": {
        // Mode-B dossier: EV + reviewer chain (R1-R11) + verdict per
        // candidate. Optional nav/holdings/puts_held engage the D17 R7/R8/R9/
        // R10 portfolio gates (concentration meters + tail soft-warns).
        const topN = searchParams.get("top_n") || "10";
        const dte = searchParams.get("dte") || "35";
        const delta = searchParams.get("delta") || "0.25";
        const minEv = searchParams.get("min_ev") || "0";
        const asOf = searchParams.get("as_of") || "";
        const timeframe = searchParams.get("timeframe") || "1D";
        const screenshotsDir = searchParams.get("screenshots_dir") || "";
        const universeLimit = searchParams.get("universe_limit") || "";
        const nav = searchParams.get("nav") || "";
        const holdings = searchParams.get("holdings") || "";
        const putsHeld = searchParams.get("puts_held") || "";
        const regimeMap = searchParams.get("regime_map") || "";
        const qs = new URLSearchParams({
          top_n: topN,
          dte,
          delta,
          min_ev: minEv,
          timeframe,
        });
        if (asOf) qs.set("as_of", asOf);
        if (screenshotsDir) qs.set("screenshots_dir", screenshotsDir);
        if (universeLimit) qs.set("universe_limit", universeLimit);
        if (nav) qs.set("nav", nav);
        if (holdings) qs.set("holdings", holdings);
        if (putsHeld) qs.set("puts_held", putsHeld);
        if (regimeMap) qs.set("regime_map", regimeMap);
        const data = await fetchEngine(`/api/tv/dossier?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "concentration": {
        // Concentration-cap preview — the operator surface where the ARMED
        // R9 (sector, 25% NAV) / R10 (single-name, 10% NAV) production caps
        // actually fire against an EV-ranked batch (engine PR #351).
        // Display-only: the engine reports ADMIT/REFUSE per candidate with
        // the tracker's structured refusal reason; nothing here re-ranks.
        const qs = new URLSearchParams({
          dte: searchParams.get("dte") || "35",
          delta: searchParams.get("delta") || "0.25",
          min_ev: searchParams.get("min_ev") || "0",
          initial_capital: searchParams.get("initial_capital") || "100000",
          top_n: searchParams.get("top_n") || "20",
        });
        for (const k of ["as_of", "universe_limit", "entry_date", "tickers"]) {
          const v = searchParams.get(k);
          if (v) qs.set(k, v);
        }
        const data = await fetchEngine(`/api/concentration_preview?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "dealer_positioning": {
        // Dealer GEX / put-call walls / gamma-flip read for one underlying
        // (audit V surface). Display-only context — the dealer multiplier
        // itself is applied engine-side and is clamped there.
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          dte: searchParams.get("dte") || "35",
        });
        const data = await fetchEngine(`/api/tv/dealer_positioning?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "analyze": {
        const ticker = searchParams.get("ticker") || "AAPL";
        const data = await fetchEngine(`/api/analyze/${encodeURIComponent(ticker)}`);
        return NextResponse.json(data);
      }

      case "regime": {
        const ticker = searchParams.get("ticker") || "SPY";
        const qs = new URLSearchParams({ ticker });
        const data = await fetchEngine(`/api/regime?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "committee": {
        const ticker = searchParams.get("ticker") || "AAPL";
        const qs = new URLSearchParams({ ticker });
        const data = await fetchEngine(`/api/committee?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "calendar": {
        const ticker = searchParams.get("ticker") || "";
        const days = searchParams.get("days") || "30";
        const qs = new URLSearchParams({ days });
        if (ticker) qs.set("ticker", ticker);
        const data = await fetchEngine(`/api/calendar?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "portfolio": {
        const tickers = searchParams.get("tickers") || "AAPL,MSFT,JPM";
        const qs = new URLSearchParams({ tickers });
        const data = await fetchEngine(`/api/portfolio?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "vix": {
        const data = await fetchEngine("/api/vix");
        return NextResponse.json(data);
      }

      case "fundamentals": {
        const qs = new URLSearchParams({ ticker: searchParams.get("ticker") || "AAPL" });
        const data = await fetchEngine(`/api/fundamentals?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "screen": {
        const qs = new URLSearchParams({
          min_score: searchParams.get("min_score") || "50",
          limit: searchParams.get("limit") || "20",
        });
        const data = await fetchEngine(`/api/screen?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "chart": {
        const chartType = searchParams.get("chart_type") || "bollinger";
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          days: searchParams.get("days") || "120",
        });
        const data = await fetchEngine(
          `/api/chart/${encodeURIComponent(chartType)}?${qs.toString()}`
        );
        return NextResponse.json(data);
      }

      case "strangle": {
        const qs = new URLSearchParams({ ticker: searchParams.get("ticker") || "AAPL" });
        const data = await fetchEngine(`/api/strangle?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "payoff": {
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          strategy: searchParams.get("strategy") || "csp",
          strike: searchParams.get("strike") || "",
          premium: searchParams.get("premium") || "",
          dte: searchParams.get("dte") || "45",
        });
        const data = await fetchEngine(`/api/payoff?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "expected_move": {
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          dte: searchParams.get("dte") || "45",
        });
        const data = await fetchEngine(`/api/expected_move?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "strikes": {
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          strategy: searchParams.get("strategy") || "csp",
          dte: searchParams.get("dte") || "45",
        });
        const data = await fetchEngine(`/api/strikes?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "iv_history": {
        const qs = new URLSearchParams({
          ticker: searchParams.get("ticker") || "AAPL",
          days: searchParams.get("days") || "252",
        });
        const data = await fetchEngine(`/api/iv_history?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "memo": {
        const qs = new URLSearchParams({ ticker: searchParams.get("ticker") || "AAPL" });
        const data = await fetchEngine(`/api/memo?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "summary": {
        const qs = new URLSearchParams({ ticker: searchParams.get("ticker") || "AAPL" });
        const data = await fetchEngine(`/api/summary?${qs.toString()}`);
        return NextResponse.json(data);
      }

      case "ollama_status": {
        const data = await fetchEngine("/api/ollama_status");
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
