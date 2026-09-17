// ─── Market Data Types ─────────────────────────────────────────────────

/** One resolved quote (`/api/market`, the watchlist): Finnhub, a <24h cached
 *  snapshot, or the engine's EOD close — the route labels the provenance. */
export interface Quote {
  ticker: string;
  price: number;
  changePct: number;
  volume: number;
  capturedAt: string;
}

// ─── Event Types ───────────────────────────────────────────────────────

export interface CalendarEvent {
  eventId: string;
  eventType: "earnings" | "fomc" | "cpi" | "jobs" | "gdp";
  ticker: string | null;
  eventDate: string;
  description: string | null;
}

// ─── Options Engine Types (smart-wheel-engine) ────────────────────────

/**
 * One EV-ranked candidate row from /api/engine?action=candidates.
 *
 * Every numeric is nullable: the engine omits fields on degraded payloads and
 * the terminal renders "—" for absent values rather than fabricating a 0.
 * Unit notes (they differ per field — see runtime audit):
 *   premium      per-SHARE dollars
 *   evDollars / evPerDay / maxLoss   per-CONTRACT dollars
 *   probProfit / CI bounds           0-1
 *   targetDelta  the delta the strike was SELECTED at — an input, not a
 *                measured Greek
 *   expiration   modeled as as_of + dte, not an exchange listing — show "~"
 */
export interface WheelTrade {
  ticker: string;
  strategy: "short_put" | "covered_call";
  strike: number | null;
  expiration: string | null;
  dte: number | null;
  premium: number | null;
  probProfit: number | null;
  probProfitCiLow: number | null;
  probProfitCiHigh: number | null;
  nScenarios: number | null;
  evDollars: number | null;
  evPerDay: number | null;
  maxLoss: number | null;
  iv: number | null;
  targetDelta: number | null;
  recommendation: string | null;
  distributionSource: string | null;
}

/**
 * /api/engine?action=regime payload. This endpoint is a VIX-band heuristic
 * (NOT the engine's 4-state HMM) — label it as such wherever rendered.
 * trendScore/confidence were fabricated server constants and no longer exist.
 */
export interface MarketRegime {
  regime:
    | "BULL"
    | "BEAR"
    | "NEUTRAL"
    | "HIGH_VOL"
    | "ELEVATED"
    | "LOW_VOL"
    | "---";
  vix: number;
  vixPercentile: number | null;
  contango: boolean | null;
  termStructure: string | null;
  vix3m: number | null;
  vix6m: number | null;
}

// ─── Live IBKR Book Types (read-only viewer over /api/portfolio/*) ────

export interface LiveBookSummary {
  asOf: string | null;
  netLiq: number | null;
  dayChangeUsd: number | null;
  dayChangePct: number | null;
  cash: number | null;
  unrealizedPnl: number | null;
  realizedYtd: number | null;
  premium30d: number | null;
  winRate: number | null;
  availableFunds: number | null;
  excessLiquidity: number | null;
  maintMargin: number | null;
  source: string | null;
}

export interface LiveBookLeg {
  sym: string;
  name: string;
  state: string;
  qty: number | null;
  mark: number | null;
  mktValue: number | null;
  uPnl: number | null;
  pctNavExact: number | null;
  breach: boolean;
  sector: string | null;
  /** Server-computed DTE (expiry minus snapshot as_of, clamped >=0).
   *  Null when the adapter did not supply the field. */
  dte: number | null;
  /** ISO expiry date (YYYY-MM-DD) from the adapter. Null if absent. */
  expiry: string | null;
  /** Option strike in local currency. Null if absent or not an option leg. */
  strike: number | null;
  /** Moneyness in local currency: (mark - strike) / strike. Null if absent. */
  moneyness: number | null;
}

// ─── Terminal Command Types ───────────────────────────────────────────

export interface TerminalCommand {
  command: string;
  description: string;
  action: string;
}
