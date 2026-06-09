// Mock portfolio data for the performance-viewer AESTHETICS round.
// Numbers mirror the operator's real IBKR book (2026-06-05 read) so the layout
// stresses the way the live data will. NO real fetching here — functionality
// (the /api/portfolio/* endpoints + IBKR snapshot feed) lands in a later round.

export type WheelState =
  // Wheel-aggregated states (per-underlying view)
  | "csp"
  | "assigned"
  | "cc"
  // Flat per-leg states (one row per raw position)
  | "shares"
  | "short_put"
  | "short_call"
  | "long_put"
  | "long_call";
export type Period = "1D" | "1W" | "1M" | "3M" | "YTD" | "1Y" | "All";

export const PERIODS: Period[] = ["1D", "1W", "1M", "3M", "YTD", "1Y", "All"];

// Fields the live IBKR feed may not derive (deltas need history; realized/
// premium/win-rate need a trade ledger) are nullable — the engine returns
// null and the UI renders "—" rather than a misleading 0. The balance-sheet
// fields are always present.
export interface Account {
  asOf: string;
  netLiq: number;
  dayChangeUsd: number | null;
  dayChangePct: number | null;
  cash: number;
  unrealizedPnl: number | null;
  realizedYtd: number | null;
  premium30d: number | null;
  winRate: number | null;
  availableFunds: number;
  excessLiquidity: number;
  maintMargin: number;
}

export const ACCOUNT: Account = {
  asOf: "2026-06-05T21:36:00Z",
  netLiq: 144507,
  dayChangeUsd: -1240,
  dayChangePct: -0.0085,
  cash: 105123,
  unrealizedPnl: -37014,
  realizedYtd: 8900,
  premium30d: 12400,
  winRate: 0.68,
  availableFunds: -9257,
  excessLiquidity: 4746,
  maintMargin: 184251,
};

// Total-return by period — pct (fraction) + the $ move over that window.
export const RETURNS: Record<Period, { pct: number; usd: number }> = {
  "1D": { pct: -0.0085, usd: -1240 },
  "1W": { pct: -0.019, usd: -2790 },
  "1M": { pct: -0.018, usd: -2693 },
  "3M": { pct: -0.048, usd: -7293 },
  YTD: { pct: -0.056, usd: -8593 },
  "1Y": { pct: 0.129, usd: 16507 },
  All: { pct: 0.445, usd: 44507 },
};

// Monthly equity curve (portfolio net-liq vs SPY-equivalent), 13 points.
export interface EquityPoint {
  m: string;
  port: number;
  spy: number;
  premium: number;
}
export const EQUITY: EquityPoint[] = [
  { m: "Jun '25", port: 128000, spy: 128000, premium: 7200 },
  { m: "Jul", port: 131500, spy: 130200, premium: 9100 },
  { m: "Aug", port: 129800, spy: 129000, premium: 6800 },
  { m: "Sep", port: 134200, spy: 132500, premium: 8400 },
  { m: "Oct", port: 139000, spy: 135800, premium: 10250 },
  { m: "Nov", port: 143600, spy: 138000, premium: 9700 },
  { m: "Dec", port: 148900, spy: 141200, premium: 11200 },
  { m: "Jan '26", port: 153100, spy: 144000, premium: 8900 },
  { m: "Feb", port: 156200, spy: 146800, premium: 12600 },
  { m: "Mar", port: 151800, spy: 148100, premium: 10800 },
  { m: "Apr", port: 149000, spy: 149500, premium: 9400 },
  { m: "May", port: 147200, spy: 151000, premium: 13100 },
  { m: "Jun", port: 144507, spy: 152300, premium: 12400 },
];

// How many trailing points each period window shows on the chart.
export const PERIOD_WINDOW: Record<Period, number> = {
  "1D": 2,
  "1W": 2,
  "1M": 3,
  "3M": 4,
  YTD: 6,
  "1Y": 13,
  All: 13,
};

export interface Holding {
  sym: string;
  name: string;
  state: WheelState;
  qty: number;
  mark: number;
  mktValue: number; // USD-equivalent, for the position value column / sorting
  uPnl: number;
  pctNav: number; // exposure as % of NAV (short-put notional for CSPs)
  breach: boolean; // single-name cap (10% NAV) breach
  sector: string;
  currency: "USD" | "CAD";
}

export const HOLDINGS: Holding[] = [
  { sym: "CLS", name: "Celestica", state: "csp", qty: -5, mark: 53.0, mktValue: -27000, uPnl: -24300, pctNav: 147, breach: true, sector: "Semiconductors", currency: "USD" },
  { sym: "MU", name: "Micron", state: "csp", qty: -1, mark: 116.62, mktValue: -11662, uPnl: -8093, pctNav: 67, breach: true, sector: "Semiconductors", currency: "USD" },
  { sym: "AMD", name: "Advanced Micro Devices", state: "csp", qty: -1, mark: 31.73, mktValue: -3173, uPnl: -2862, pctNav: 35, breach: true, sector: "Semiconductors", currency: "USD" },
  { sym: "MRVL", name: "Marvell Technology", state: "csp", qty: -1, mark: 48.62, mktValue: -4862, uPnl: -1330, pctNav: 21, breach: true, sector: "Semiconductors", currency: "USD" },
  { sym: "TSM", name: "Taiwan Semi (ADR)", state: "cc", qty: 100, mark: 412.5, mktValue: 41250, uPnl: -1390, pctNav: 29, breach: false, sector: "Semiconductors", currency: "USD" },
  { sym: "NVDA", name: "NVIDIA", state: "cc", qty: 100, mark: 205.39, mktValue: 20539, uPnl: -679, pctNav: 14, breach: false, sector: "Semiconductors", currency: "USD" },
  { sym: "WMT", name: "Walmart", state: "cc", qty: 100, mark: 118.3, mktValue: 11830, uPnl: -215, pctNav: 8, breach: false, sector: "Consumer Staples", currency: "USD" },
  { sym: "ENB", name: "Enbridge", state: "cc", qty: 100, mark: 56.34, mktValue: 5635, uPnl: -20, pctNav: 4, breach: false, sector: "Energy", currency: "CAD" },
  { sym: "CNQ", name: "Canadian Natural Res.", state: "assigned", qty: 100, mark: 45.8, mktValue: 4580, uPnl: -142, pctNav: 3, breach: false, sector: "Energy", currency: "CAD" },
];

// Hex (not CSS vars) so the badge can append an alpha suffix for the tint.
export const WHEEL_LABEL: Record<WheelState, { full: string; short: string; color: string }> = {
  csp: { full: "Cash-Secured Put", short: "CSP", color: "#56b6f5" },
  assigned: { full: "Assigned Stock", short: "ASSIGNED", color: "#b79cfb" },
  cc: { full: "Covered Call", short: "COV CALL", color: "#f5b544" },
  // Flat per-leg states (build_positions_flat) — one row per raw position.
  shares: { full: "Shares", short: "SHARES", color: "#b79cfb" },
  short_put: { full: "Short Put", short: "SHORT PUT", color: "#56b6f5" },
  short_call: { full: "Short Call", short: "SHORT CALL", color: "#f5b544" },
  long_put: { full: "Long Put", short: "LONG PUT", color: "#fb7185" },
  long_call: { full: "Long Call", short: "LONG CALL", color: "#4ade80" },
};

// Allocation donut — share of gross exposure by sector.
export const SECTORS = [
  { name: "Semiconductors", val: 82.8, color: "var(--color-pf-accent)" },
  { name: "Energy", val: 9.5, color: "var(--color-pf-csp)" },
  { name: "Consumer Staples", val: 7.7, color: "var(--color-pf-assigned)" },
];

export const CURRENCY = [
  { name: "USD", val: 91.8 },
  { name: "CAD", val: 8.2 },
];

// Risk radar — concentration caps (% of NAV) + which names breach.
export const SINGLE_NAME_CAP = 10;
export const SECTOR_CAP = 25;

export const SINGLE_NAME = [
  { sym: "CLS", pct: 147 },
  { sym: "MU", pct: 67 },
  { sym: "AMD", pct: 35 },
  { sym: "MRVL", pct: 21 },
];

export const SECTOR_EXPOSURE = [
  { name: "Semiconductors", pct: 312 },
  { name: "Energy", pct: 7 },
  { name: "Consumer Staples", pct: 8 },
];

export const ASK_CHIPS = [
  "How did MU do this month?",
  "Where am I over-concentrated?",
  "Realized P&L this week",
  "What if CLS gets assigned?",
];
