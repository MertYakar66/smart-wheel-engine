"use client";

// Live data layer for the performance viewer. Fetches the read-only
// /api/portfolio/* engine endpoints (via the Next proxy) and exposes the
// SAME shapes the components already consume, with mock.ts as a typed
// per-slice fallback so the page renders even when the engine is down.
// Read-only + observational — no mutations, no EV authority.

import { useEffect, useState } from "react";

import {
  ACCOUNT,
  CONCENTRATION,
  CURRENCY,
  EQUITY,
  EQUITY_STATS,
  GATES,
  HOLDINGS,
  INCOME,
  IV_ASSUMPTION,
  RETURNS,
  SECTOR_CAP,
  SECTOR_EXPOSURE,
  SECTORS,
  SINGLE_NAME,
  SINGLE_NAME_CAP,
  type ConcentrationRow,
  type EquityPoint,
  type EquityStats,
  type Gates,
  type Holding,
  type IncomeView,
} from "./mock";
import { type SliceSource } from "./parts";

export interface Margin {
  availableFunds: number;
  excessLiquidity: number;
  maintMargin: number;
  cushionPct: number;
  stressed: boolean;
}

export interface PortfolioData {
  account: typeof ACCOUNT;
  returns: typeof RETURNS;
  equity: EquityPoint[];
  /** Sharpe/Sortino/MaxDD served on /history — null until that slice lands */
  stats: EquityStats | null;
  holdings: Holding[];
  income: IncomeView;
  sectors: { name: string; val: number; color?: string }[];
  currency: { name: string; val: number }[];
  singleName: { sym: string; pct: number }[];
  /** all-exposure per-underlying rows — null on an older engine payload */
  concentration: ConcentrationRow[] | null;
  sectorExposure: { name: string; pct: number }[];
  caps: { singleName: number; sector: number };
  margin: Margin;
  /** live R7–R10 gate overlay — null on an older engine payload */
  gates: Gates | null;
  ivAssumption: number | null;
}

/** Typed fallback assembled from mock.ts — used until/unless live data lands. */
export const MOCK_DATA: PortfolioData = {
  account: ACCOUNT,
  returns: RETURNS,
  equity: EQUITY,
  stats: EQUITY_STATS,
  holdings: HOLDINGS,
  income: INCOME,
  sectors: SECTORS,
  currency: CURRENCY,
  singleName: SINGLE_NAME,
  concentration: CONCENTRATION,
  sectorExposure: SECTOR_EXPOSURE,
  caps: { singleName: SINGLE_NAME_CAP, sector: SECTOR_CAP },
  margin: {
    availableFunds: ACCOUNT.availableFunds,
    excessLiquidity: ACCOUNT.excessLiquidity,
    maintMargin: ACCOUNT.maintMargin,
    cushionPct: 0.12,
    stressed: ACCOUNT.availableFunds < 0,
  },
  gates: GATES,
  ivAssumption: IV_ASSUMPTION,
};

async function fetchView<T>(sub: string): Promise<T> {
  const res = await fetch(`/api/portfolio/${sub}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${sub}: engine returned ${res.status}`);
  return (await res.json()) as T;
}

export type SliceName = "summary" | "positions" | "returns" | "income" | "risk" | "history";

export interface PortfolioState {
  data: PortfolioData;
  /** true once the account summary came from a real (live) IBKR drop */
  live: boolean;
  loading: boolean;
  /** per-slice provenance — drives the honest header label + per-card badges */
  sources: Record<SliceName, SliceSource>;
}

const ALL_MOCK: Record<SliceName, SliceSource> = {
  summary: "mock",
  positions: "mock",
  returns: "mock",
  income: "mock",
  risk: "mock",
  history: "mock",
};

/** Shape of the /risk slice — gates/ivAssumption/concentration are optional
 * so the hook stays null-safe against an older engine payload. */
interface RiskSlice {
  singleName: PortfolioData["singleName"];
  concentration?: ConcentrationRow[];
  sectorExposure: PortfolioData["sectorExposure"];
  sectors: PortfolioData["sectors"];
  currency: PortfolioData["currency"];
  singleNameCap: number;
  sectorCap: number;
  margin: Margin;
  gates?: Gates;
  ivAssumption?: number;
}

export function usePortfolioData(): PortfolioState {
  const [state, setState] = useState<PortfolioState>({
    data: MOCK_DATA,
    live: false,
    loading: true,
    sources: ALL_MOCK,
  });

  useEffect(() => {
    let cancelled = false;

    (async () => {
      // Fetch independently so one failing slice doesn't blank the page —
      // each falls back to its mock counterpart.
      const [summary, positions, returns, income, risk, history] = await Promise.allSettled([
        fetchView<PortfolioData["account"]>("summary"),
        fetchView<{ holdings: Holding[]; legs?: Holding[] }>("positions"),
        fetchView<{ returns: PortfolioData["returns"] }>("returns"),
        fetchView<IncomeView>("income"),
        fetchView<RiskSlice>("risk"),
        fetchView<{ equity: EquityPoint[]; stats?: EquityStats }>("history"),
      ]);

      if (cancelled) return;

      const ok = (r: PromiseSettledResult<unknown>) => r.status === "fulfilled";
      const next: PortfolioData = {
        account: ok(summary)
          ? (summary as PromiseFulfilledResult<PortfolioData["account"]>).value
          : MOCK_DATA.account,
        // Prefer the flat per-leg view (every stock + option leg); fall back to
        // the wheel-aggregated holdings if an older engine omits `legs`.
        holdings: ok(positions)
          ? (() => {
              const v = (
                positions as PromiseFulfilledResult<{ holdings: Holding[]; legs?: Holding[] }>
              ).value;
              return v.legs ?? v.holdings;
            })()
          : MOCK_DATA.holdings,
        returns: ok(returns)
          ? (returns as PromiseFulfilledResult<{ returns: PortfolioData["returns"] }>).value.returns
          : MOCK_DATA.returns,
        income: ok(income)
          ? (income as PromiseFulfilledResult<IncomeView>).value
          : MOCK_DATA.income,
        equity: ok(history)
          ? (history as PromiseFulfilledResult<{ equity: EquityPoint[] }>).value.equity
          : MOCK_DATA.equity,
        // stats is null (not mock) when the live history omits it — the strip
        // hides rather than decorate live data with fabricated mock stats.
        stats: ok(history)
          ? ((history as PromiseFulfilledResult<{ stats?: EquityStats }>).value.stats ?? null)
          : MOCK_DATA.stats,
        sectors: MOCK_DATA.sectors,
        currency: MOCK_DATA.currency,
        singleName: MOCK_DATA.singleName,
        concentration: MOCK_DATA.concentration,
        sectorExposure: MOCK_DATA.sectorExposure,
        caps: MOCK_DATA.caps,
        margin: MOCK_DATA.margin,
        gates: MOCK_DATA.gates,
        ivAssumption: MOCK_DATA.ivAssumption,
      };

      if (ok(risk)) {
        const r = (risk as PromiseFulfilledResult<RiskSlice>).value;
        next.sectors = r.sectors ?? MOCK_DATA.sectors;
        next.currency = r.currency ?? MOCK_DATA.currency;
        next.singleName = r.singleName ?? MOCK_DATA.singleName;
        next.sectorExposure = r.sectorExposure ?? MOCK_DATA.sectorExposure;
        next.caps = {
          singleName: r.singleNameCap ?? MOCK_DATA.caps.singleName,
          sector: r.sectorCap ?? MOCK_DATA.caps.sector,
        };
        next.margin = r.margin ?? MOCK_DATA.margin;
        // Live payload wins even when a field is absent (older engine): null
        // hides the board rather than blending mock rows into live data.
        next.concentration = r.concentration ?? null;
        next.gates = r.gates ?? null;
        next.ivAssumption = r.ivAssumption ?? null;
      }

      // Per-slice provenance: a fetched slice reports source "live"/"demo"
      // (engine), a failed one is "mock" (typed fallback). The page derives an
      // honest header from these so a demo fixture never reads as "Live IBKR".
      const srcOf = (r: PromiseSettledResult<unknown>): SliceSource => {
        if (r.status !== "fulfilled") return "mock";
        const s = (r.value as { source?: string } | null)?.source;
        return s === "live" ? "live" : "demo";
      };
      const sources: Record<SliceName, SliceSource> = {
        summary: srcOf(summary),
        positions: srcOf(positions),
        returns: srcOf(returns),
        income: srcOf(income),
        risk: srcOf(risk),
        history: srcOf(history),
      };

      setState({ data: next, live: sources.summary === "live", loading: false, sources });
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
