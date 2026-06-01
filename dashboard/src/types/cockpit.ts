// ─── Decision-Cockpit Types ──────────────────────────────────────────────
//
// The real wire shapes emitted by the smart-wheel-engine HTTP API
// (`engine_api.py`), proxied through `/api/engine`. Field names match the
// engine's camelCase output exactly — do NOT invent fields the engine does
// not send. Where a field is absent from the wire it is typed `null` and the
// UI must degrade gracefully (the engine is the source of truth; the cockpit
// only visualizes what exists).
//
// Source of truth:
//   /api/candidates  -> engine_api.py::_handle_candidates  (the EngineCandidate)
//   /api/tv/dossier  -> engine_api.py::_handle_tv_dossier   (the Dossier)

/** One ranked short-put candidate from `/api/candidates`. */
export interface EngineCandidate {
  ticker: string;
  strategy: "short_put";
  spot: number;
  strike: number;
  premium: number;
  dte: number;
  /** Decimal IV (0.42 = 42%). */
  iv: number;

  // EV diagnostics — a RANKING SCORE, not a dollar forecast.
  evDollars: number;
  evPerDay: number;

  // P&L distribution (per contract, dollars). For a short put the upper
  // quartiles frequently pin at the "keep full premium" ceiling, so
  // pnlP25/P50/P75 can all be equal — the real risk lives in cvar5.
  pnlP25: number | null;
  pnlP50: number | null;
  pnlP75: number | null;
  cvar5: number | null; // 5% conditional tail loss (negative dollars)
  cvar99Evt: number | null;
  tailXi: number | null;
  heavyTail: boolean;
  omegaRatio: number | null;

  // Probabilities (0-1). probProfit is well-calibrated in the mid range
  // (~0.6-0.85) and OVER-confident in the top bin (>0.90): crisis-realized
  // ~0.57 vs ~0.96 forecast (heavy-verify I1).
  probProfit: number;
  probAssignment: number;

  fairValue: number | null;
  edgeVsFair: number | null;
  breakevenMovePct: number | null;
  distributionSource: string | null;

  // Dealer positioning overlay (advisory; clamped multiplier, never rescues).
  dealerRegime: string | null;
  dealerMultiplier: number | null;
  gexTotal: number | null;
  gammaFlipDistancePct: number | null;
  nearestPutWallStrike: number | null;
  nearestCallWallStrike: number | null;

  daysToEarnings: number | null;

  // Backward-compat / convenience aliases also emitted by the engine.
  spotPrice: number;
  expectedPnL: number;
  probability: number; // probProfit * 100
  maxLoss: number; // (strike - premium) * 100
  score: number | null;
  wheelScore: number | null;

  /** Engine recommendation label: proceed | review | skip. */
  recommendation: "proceed" | "review" | "skip";
  expiration: string;
}

export interface CandidatesResponse {
  trades: EngineCandidate[];
  count: number;
  authority: string;
  engine_version: string;
  universe_scanned: number;
  universe_total: number;
  params?: Record<string, unknown>;
  error?: string;
  detail?: string;
  hint?: string;
}

// ─── Dossier (reviewer chain) ─────────────────────────────────────────────

export type Verdict = "proceed" | "review" | "skip" | "blocked";

/** One candidate dossier from `/api/tv/dossier` (`CandidateDossier.to_dict`). */
export interface Dossier {
  ticker: string;
  /** The full EV row dict (snake_case keys: ev_dollars, prob_profit, cvar_5, …). */
  ev_row: Record<string, unknown>;
  chart_context: Record<string, unknown> | null;
  verdict: Verdict;
  /** e.g. "elevated_vol_top_bin" (R11), "sector_cap_breach" (R9), "chart_missing" (R2). */
  verdict_reason: string;
  review_notes: string[];
  has_chart: boolean;
  built_at?: string;
}

export interface DossierResponse {
  dossiers: Dossier[];
  count: number;
  verdict_counts: Record<Verdict, number>;
  universe_scanned: number;
  universe_total: number;
  params?: Record<string, unknown>;
  engine_version: string;
  error?: string;
  detail?: string;
  hint?: string;
}

// ─── VIX regime (`/api/vix`) ──────────────────────────────────────────────

export interface VixRegime {
  vix: number;
  vix_percentile?: number;
  term_structure?: string;
  vix_3m?: number;
  vix_6m?: number;
  error?: string;
}
