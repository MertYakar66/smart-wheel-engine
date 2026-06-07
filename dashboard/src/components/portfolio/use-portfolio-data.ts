"use client";

// Live data layer for the performance viewer. Fetches the read-only
// /api/portfolio/* engine endpoints (via the Next proxy) and exposes the
// SAME shapes the components already consume, with mock.ts as a typed
// per-slice fallback so the page renders even when the engine is down.
// Read-only + observational — no mutations, no EV authority.

import { useEffect, useState } from "react";

import {
  ACCOUNT,
  CURRENCY,
  EQUITY,
  HOLDINGS,
  RETURNS,
  SECTOR_CAP,
  SECTOR_EXPOSURE,
  SECTORS,
  SINGLE_NAME,
  SINGLE_NAME_CAP,
  type EquityPoint,
  type Holding,
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
  holdings: Holding[];
  sectors: { name: string; val: number; color?: string }[];
  currency: { name: string; val: number }[];
  singleName: { sym: string; pct: number }[];
  sectorExposure: { name: string; pct: number }[];
  caps: { singleName: number; sector: number };
  margin: Margin;
}

/** Typed fallback assembled from mock.ts — used until/unless live data lands. */
export const MOCK_DATA: PortfolioData = {
  account: ACCOUNT,
  returns: RETURNS,
  equity: EQUITY,
  holdings: HOLDINGS,
  sectors: SECTORS,
  currency: CURRENCY,
  singleName: SINGLE_NAME,
  sectorExposure: SECTOR_EXPOSURE,
  caps: { singleName: SINGLE_NAME_CAP, sector: SECTOR_CAP },
  margin: {
    availableFunds: ACCOUNT.availableFunds,
    excessLiquidity: ACCOUNT.excessLiquidity,
    maintMargin: ACCOUNT.maintMargin,
    cushionPct: 0.12,
    stressed: ACCOUNT.availableFunds < 0,
  },
};

async function fetchView<T>(sub: string): Promise<T> {
  const res = await fetch(`/api/portfolio/${sub}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${sub}: engine returned ${res.status}`);
  return (await res.json()) as T;
}

export type SliceName = "summary" | "positions" | "returns" | "risk" | "history";

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
  risk: "mock",
  history: "mock",
};

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
      const [summary, positions, returns, risk, history] = await Promise.allSettled([
        fetchView<PortfolioData["account"]>("summary"),
        fetchView<{ holdings: Holding[] }>("positions"),
        fetchView<{ returns: PortfolioData["returns"] }>("returns"),
        fetchView<{
          singleName: PortfolioData["singleName"];
          sectorExposure: PortfolioData["sectorExposure"];
          sectors: PortfolioData["sectors"];
          currency: PortfolioData["currency"];
          singleNameCap: number;
          sectorCap: number;
          margin: Margin;
        }>("risk"),
        fetchView<{ equity: EquityPoint[] }>("history"),
      ]);

      if (cancelled) return;

      const ok = (r: PromiseSettledResult<unknown>) => r.status === "fulfilled";
      const next: PortfolioData = {
        account: ok(summary)
          ? (summary as PromiseFulfilledResult<PortfolioData["account"]>).value
          : MOCK_DATA.account,
        holdings: ok(positions)
          ? (positions as PromiseFulfilledResult<{ holdings: Holding[] }>).value.holdings
          : MOCK_DATA.holdings,
        returns: ok(returns)
          ? (returns as PromiseFulfilledResult<{ returns: PortfolioData["returns"] }>).value.returns
          : MOCK_DATA.returns,
        equity: ok(history)
          ? (history as PromiseFulfilledResult<{ equity: EquityPoint[] }>).value.equity
          : MOCK_DATA.equity,
        sectors: MOCK_DATA.sectors,
        currency: MOCK_DATA.currency,
        singleName: MOCK_DATA.singleName,
        sectorExposure: MOCK_DATA.sectorExposure,
        caps: MOCK_DATA.caps,
        margin: MOCK_DATA.margin,
      };

      if (ok(risk)) {
        const r = (
          risk as PromiseFulfilledResult<{
            singleName: PortfolioData["singleName"];
            sectorExposure: PortfolioData["sectorExposure"];
            sectors: PortfolioData["sectors"];
            currency: PortfolioData["currency"];
            singleNameCap: number;
            sectorCap: number;
            margin: Margin;
          }>
        ).value;
        next.sectors = r.sectors ?? MOCK_DATA.sectors;
        next.currency = r.currency ?? MOCK_DATA.currency;
        next.singleName = r.singleName ?? MOCK_DATA.singleName;
        next.sectorExposure = r.sectorExposure ?? MOCK_DATA.sectorExposure;
        next.caps = {
          singleName: r.singleNameCap ?? MOCK_DATA.caps.singleName,
          sector: r.sectorCap ?? MOCK_DATA.caps.sector,
        };
        next.margin = r.margin ?? MOCK_DATA.margin;
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
