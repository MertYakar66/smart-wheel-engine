"use client";

// Dedicated data layer for the Trades tab — kept separate from
// use-portfolio-data.ts so the trade history (a large, independent slice)
// never blocks or complicates the core viewer. Fetches the read-only
// /api/portfolio/trades engine endpoint (Flex-sourced, includes expiries &
// assignments). Read-only + observational — no mutations, no EV authority.

import { useEffect, useState } from "react";

import { type SliceSource } from "./parts";

export interface Trade {
  datetime: string | null;
  date: string | null;
  symbol: string;
  occ_symbol: string | null;
  sec_type: string; // STK | OPT | CASH
  right: string | null; // C | P
  strike: number | null;
  expiry: string | null;
  side: string | null; // BUY | SELL
  qty: number | null;
  price: number | null;
  proceeds: number | null;
  commission: number | null;
  realized_pnl: number;
  currency: string | null;
  open_close: string | null; // O | C
  code: string | null; // Ep (expiry) | A (assignment) | ...
  multiplier: number;
  id: string;
}

export interface TickerAgg {
  symbol: string;
  currency: string;
  mixed_currency: boolean;
  opt_pnl: number;
  stk_pnl: number;
  other_pnl: number;
  opt_pnl_usd: number;
  stk_pnl_usd: number;
  total_pnl_usd: number;
  premium_collected: number;
  opt_count: number;
  stk_count: number;
  other_count: number;
  first: string | null;
  last: string | null;
}

export interface TradesTotals {
  trade_count: number;
  ticker_count: number;
  realized_usd: number;
  opt_realized_usd: number;
  stk_realized_usd: number;
}

export interface TradesCoverage {
  from: string | null;
  to: string | null;
  query: string | null;
  count: number;
}

export interface TradesData {
  trades: Trade[];
  tickers: TickerAgg[];
  totals: TradesTotals;
  coverage: TradesCoverage | null;
  source: SliceSource;
  loading: boolean;
}

const EMPTY_TOTALS: TradesTotals = {
  trade_count: 0,
  ticker_count: 0,
  realized_usd: 0,
  opt_realized_usd: 0,
  stk_realized_usd: 0,
};

const EMPTY: TradesData = {
  trades: [],
  tickers: [],
  totals: EMPTY_TOTALS,
  coverage: null,
  source: "mock",
  loading: true,
};

export function useTradesData(): TradesData {
  const [state, setState] = useState<TradesData>(EMPTY);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/portfolio/trades", { cache: "no-store" });
        if (!res.ok) throw new Error(`trades: engine returned ${res.status}`);
        const d = (await res.json()) as Partial<TradesData> & { source?: string };
        if (cancelled) return;
        setState({
          trades: d.trades ?? [],
          tickers: d.tickers ?? [],
          totals: d.totals ?? EMPTY_TOTALS,
          coverage: d.coverage ?? null,
          // The endpoint reports "live" for a real Flex drop; a served fixture
          // reports "demo". A failed fetch (engine offline) → "mock".
          source: d.source === "live" ? "live" : "demo",
          loading: false,
        });
      } catch {
        if (!cancelled) setState({ ...EMPTY, loading: false, source: "mock" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
