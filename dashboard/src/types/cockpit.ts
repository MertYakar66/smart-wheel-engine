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
  /** MEAN P&L of the worst-5% scenarios (dollars per contract). Usually
   *  negative (the modeled crash tail) but CAN be positive — observed live
   *  (ADI +922.37): even the worst-5% mean keeps a profit. Never render a
   *  positive cvar5 as a loss. */
  cvar5: number | null;
  cvar99Evt: number | null;
  tailXi: number | null;
  heavyTail: boolean;
  omegaRatio: number | null;

  // Probabilities (0-1). probProfit is well-calibrated in the mid range
  // (~0.6-0.85) and OVER-confident in the top bin (>0.90): crisis-realized
  // ~0.57 vs ~0.96 forecast (heavy-verify I1).
  probProfit: number;
  /**
   * Wilson 95% SAMPLING CI for prob_profit and the scenario count it rests on
   * (engine `n_scenarios` / `prob_profit_ci_low` / `prob_profit_ci_high`). This
   * is the sampling uncertainty of the k/N forward-scenario frequency — a wide
   * band (N is small, ~30-35 on the empirical path) means the point estimate
   * must not be read to 2 dp. Orthogonal to the top-bin calibration bias noted
   * above. Optional: absent on older engine payloads, null when N <= 0.
   */
  nScenarios?: number;
  probProfitCiLow?: number | null;
  probProfitCiHigh?: number | null;
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

  /** Engine recommendation label: proceed | review | skip. NOTE: this is the
   *  API's EV-floor + prob_profit ≥ 0.65 label, NOT the EnginePhaseReviewer
   *  verdict — the authoritative verdict comes from `/api/tv/dossier`. */
  recommendation: "proceed" | "review" | "skip";
  /** MODELED contract date: as_of + dte (the synthetic contract the EV math
   *  priced — the Bloomberg provider has no listed chain). Label "~"/"modeled";
   *  never present it as a listed expiration. */
  expiration: string;
  /** The delta the strike was SELECTED at (ranker delta_target) — a selection
   *  target, not a measured per-strike Greek. Absent on older payloads. */
  targetDelta?: number;
}

export interface CandidatesResponse {
  trades: EngineCandidate[];
  count: number;
  authority: string;
  engine_version: string;
  universe_scanned: number;
  universe_total: number;
  /** Ranker gate diagnostics serialized from frame.attrs["drops_summary"]
   *  ({ total_dropped, by_gate }) — null/absent on older engine payloads. */
  drops_summary?: { total_dropped: number; by_gate: Record<string, number> } | null;
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

// ─── Engine status (`/api/status`) ────────────────────────────────────────

export interface EngineStatus {
  status?: string;
  provider?: string;
  /** Freshest bar in the engine's data files (YYYY-MM-DD) — the honest
   *  "latest available" reference for as_of, never hardcode it. */
  data_frontier?: string;
  universe_size?: number;
  vix?: number;
  error?: string;
}

// ─── Concentration preview (`/api/concentration_preview`) ────────────────
//
// The operator surface where the ARMED R9/R10 production caps fire over the
// EV-ranked batch (engine PR #351). Sequential consume in EV-rank order:
// each admit changes the ephemeral book the next candidate is checked
// against. Display-only — refuse-only gates, no order routing.

export interface ConcentrationOutcome {
  ticker: string;
  evDollars: number | null;
  opened: boolean;
  /** Coarse ("tracker_rejected"); the structured reason lives in refusals[]. */
  refusalReason: string | null;
}

export interface ConcentrationRefusal {
  ticker: string;
  /** e.g. "single_name_breach" (R10) | "sector_cap_breach" (R9). */
  reason: string;
  nav?: number | null;
  navSource?: string | null;
  sector?: string | null;
  postOpenSectorPct?: number | null;
  sectorLimit?: number | null;
  postOpenNamePct?: number | null;
  nameLimitPct?: number | null;
}

export interface ConcentrationPreview {
  authority?: string;
  engine_version?: string;
  entry_date?: string;
  initial_capital?: number;
  caps?: { sector_cap_pct?: number; single_name_cap_pct?: number };
  consumed?: number;
  opened?: number;
  refused?: number;
  outcomes?: ConcentrationOutcome[];
  refusals?: ConcentrationRefusal[];
  universe_scanned?: number;
  universe_total?: number;
  error?: string;
  detail?: string;
}

// ─── Live-book lite shapes (`/api/portfolio/{summary,positions}`) ─────────
//
// Minimal slices of the D26 read-only viewer payloads the cockpit consumes
// to attach the REAL book to the dossier call (nav/holdings/puts_held) and
// to seed the concentration NAV. Observational only.

export interface PortfolioSummaryLite {
  netLiq?: number | null;
  /** Provenance: "live" | "demo" | … — always label it. */
  source?: string;
  error?: string;
}

export interface PortfolioLegLite {
  sym?: string;
  /** IBKR local symbol, e.g. "MRVL 10JUL26 297.5 P" for options. */
  name?: string;
  /** "shares" | "short_put" | "short_call" | "long_call" | … */
  state?: string;
  qty?: number;
  /** Server-computed option fields from build_positions_flat (D26 PR #403).
   *  Consume these first; fall back to regex-parsed name strings only when absent. */
  strike?: number | null;
  expiry?: string | null;
  dte?: number | null;
}

export interface PortfolioPositionsLite {
  legs?: PortfolioLegLite[];
  source?: string;
  error?: string;
}
