// ─── Data Provider Types ───────────────────────────────────────────────

export interface Headline {
  sourceId: string;
  url: string;
  publisher: string;
  headline: string;
  publishedAt: string;
  snippet: string;
}

export interface Quote {
  ticker: string;
  price: number;
  changePct: number;
  volume: number;
  capturedAt: string;
}

export interface FilingSummary {
  ticker: string;
  filingType: string;
  filingDate: string;
  url: string;
  title: string;
  summary: string;
}

export interface MacroDataPoint {
  date: string;
  value: number;
}

export interface MacroData {
  series: string;
  description: string;
  data: MacroDataPoint[];
}

// ─── Impact & Exposure Types ──────────────────────────────────────────

export type ImpactFactor = "rates" | "oil" | "fx" | "regulation" | "earnings" | "demand" | "supply_chain" | "geopolitical" | "monetary_policy" | "labor";

export type ImpactHorizon = "intraday" | "days" | "weeks" | "quarters";

export type StoryStatus = "developing" | "evolving" | "resolved";

export type Sentiment = "positive" | "negative" | "neutral" | "mixed";

export interface ExposureMechanism {
  factor: ImpactFactor;
  direction: "positive" | "negative" | "uncertain";
  confidence: number;
  description?: string;
}

export interface UserExposure {
  id: number;
  exposureType: "ticker" | "sector" | "factor" | "country" | "theme";
  exposureValue: string;
  weight: number;
}

// ─── Story Types ───────────────────────────────────────────────────────

export interface StoryCard {
  storyId: string;
  canonicalTitle: string;
  summary: string | null;
  impactScore: number;
  sector: string | null;
  sourceCount: number;
  createdAt: string;
  entities: EntityTag[];
  sources: SourceRef[];
  // Story Graph Intelligence fields
  impactTags?: ImpactFactor[];
  exposureMechanisms?: ExposureMechanism[];
  impactHorizon?: ImpactHorizon | null;
  storyStatus?: StoryStatus;
  contradictionFlag?: boolean;
  whyItMatters?: string | null;
  corroborationScore?: number;
  exposureRelevance?: number;
}

export interface StoryDetail extends StoryCard {
  timeline: StoryTimelineEvent[];
  allSources: SourceRefDetailed[];
  relatedStories: StoryCard[];
}

export interface StoryTimelineEvent {
  id: number;
  eventType: "created" | "source_added" | "merged" | "status_change" | "contradiction_detected";
  description: string;
  metadata?: Record<string, unknown>;
  occurredAt: string;
}

export interface EntityTag {
  entityType: "ticker" | "person" | "org" | "topic" | "factor";
  entityValue: string;
}

export interface SourceRef {
  publisher: string;
  url: string;
  headline: string;
}

export interface SourceRefDetailed extends SourceRef {
  sourceId: string;
  publishedAt: string | null;
  snippet: string | null;
  sentiment: Sentiment | null;
  geography: string | null;
  rightsRestricted: boolean;
  retrievalProvider: string;
}

// ─── Event Types ───────────────────────────────────────────────────────

export interface CalendarEvent {
  eventId: string;
  eventType: "earnings" | "fomc" | "cpi" | "jobs" | "gdp";
  ticker: string | null;
  eventDate: string;
  description: string | null;
}

// ─── Alert Types ───────────────────────────────────────────────────────

export interface AlertItem {
  alertId: string;
  storyId: string | null;
  ticker: string | null;
  triggerType: "news" | "price_move";
  triggeredAt: string;
  dismissed: boolean;
  storyTitle?: string;
}

// ─── Chat Types ────────────────────────────────────────────────────────

export interface ChatMessage {
  messageId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
}

export interface ChatSession {
  sessionId: string;
  title: string | null;
  createdAt: string;
  updatedAt: string;
}

// ─── Data Provider Interface ───────────────────────────────────────────

export interface DataProvider {
  fetchHeadlines(): Promise<Headline[]>;
  fetchQuote(ticker: string): Promise<Quote | null>;
  fetchFilingSummary(
    ticker: string,
    filingType: string
  ): Promise<FilingSummary | null>;
  fetchMacroData(series: string): Promise<MacroData | null>;
}

// ─── Retrieval Provider Interface ──────────────────────────────────────

export interface RetrievalResult {
  url: string;
  title: string;
  snippet: string;
  publisher: string;
  publishedAt: string;
  relevanceScore: number;
  rightsRestricted: boolean;
  geography?: string;
  metadata?: Record<string, unknown>;
}

export interface RetrievalQuery {
  query: string;
  domain?: "news" | "filings" | "research" | "market" | "all";
  filters?: {
    tickers?: string[];
    sectors?: string[];
    dateFrom?: string;
    dateTo?: string;
    publishers?: string[];
    countries?: string[];
  };
  limit?: number;
}

export interface RetrievalProvider {
  name: string;
  search(query: RetrievalQuery): Promise<RetrievalResult[]>;
  isAvailable(): Promise<boolean>;
}

// ─── Code Execution Types ──────────────────────────────────────────────

export interface CodeExecutionRequest {
  code: string;
  language: "python";
  timeout?: number;
}

export interface CodeExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  artifacts?: ExecutionArtifact[];
  executionTime: number;
}

export interface ExecutionArtifact {
  type: "chart" | "table" | "csv" | "json";
  name: string;
  data: string;
}

// ─── RSS Feed Config ───────────────────────────────────────────────────

export interface RSSFeedConfig {
  name: string;
  url: string;
  publisher: string;
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
}

// ─── Terminal Command Types ───────────────────────────────────────────

export interface TerminalCommand {
  command: string;
  description: string;
  action: string;
}
