"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  WheelTrade,
  MarketRegime,
  LiveBookSummary,
  LiveBookLeg,
} from "@/types";

// ─── Engine API Response Types ─────────────────────────────────────────

/** Raw candidates row — every field optional so a degraded payload parses. */
interface RawCandidate {
  ticker?: string;
  strategy?: string;
  strike?: number;
  expiration?: string;
  dte?: number;
  premium?: number;
  probProfit?: number;
  probProfitCiLow?: number;
  probProfitCiHigh?: number;
  nScenarios?: number;
  evDollars?: number;
  evPerDay?: number;
  maxLoss?: number;
  iv?: number;
  targetDelta?: number;
  recommendation?: string;
  distributionSource?: string;
}

interface CandidatesResponse {
  trades: RawCandidate[];
  count: number;
}

interface RegimeResponse {
  regime: string;
  vix: number;
  vixPercentile?: number;
  contango?: boolean;
  termStructure?: string;
  vix3m?: number;
  vix6m?: number;
}

interface StatusResponse {
  status: string;
  engine: string;
  universe_size: number;
  vix: number;
  provider?: string;
  data_frontier?: string;
  error?: string;
}

/** Finite-number-or-null: never coerce an absent engine field to 0. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ─── Hook Return Type ──────────────────────────────────────────────────

interface EngineData {
  trades: WheelTrade[];
  regime: MarketRegime;
  connected: boolean;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  /** Freshest OHLCV date served by the engine (action=status data_frontier). */
  dataFrontier: string | null;
  provider: string | null;
  refresh: () => Promise<void>;
}

const EMPTY_REGIME: MarketRegime = {
  regime: "---",
  vix: 0,
  vixPercentile: null,
  contango: null,
  termStructure: null,
  vix3m: null,
  vix6m: null,
};

// Candidates scans are EOD data — nothing changes intra-minute. Cap the
// universe (full 511-name scan is ~40s; 120 is ~10-15s) and poll slowly.
const CANDIDATES_UNIVERSE_LIMIT = "120";
const POLL_MS = 120_000;

// ─── Main Hook ─────────────────────────────────────────────────────────

export function useEngineData(): EngineData {
  const [trades, setTrades] = useState<WheelTrade[]>([]);
  const [regime, setRegime] = useState<MarketRegime>(EMPTY_REGIME);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [dataFrontier, setDataFrontier] = useState<string | null>(null);
  const [provider, setProvider] = useState<string | null>(null);

  // A candidates scan can outlast the poll interval; never overlap requests.
  const inFlight = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);

    try {
      // Status first: it carries the data frontier the candidates call pins
      // its as_of to (deterministic reads instead of "whatever is current").
      const statusRes = await fetch("/api/engine?action=status", {
        signal: ctrl.signal,
      });
      if (!statusRes.ok) {
        throw new Error("Engine API unavailable. Start with: python engine_api.py");
      }
      const status: StatusResponse = await statusRes.json();
      if (status.error) {
        throw new Error(status.error);
      }
      setConnected(status.status === "connected");
      const frontier =
        typeof status.data_frontier === "string" && status.data_frontier
          ? status.data_frontier
          : null;
      setDataFrontier(frontier);
      setProvider(typeof status.provider === "string" ? status.provider : null);

      const candidatesQs = new URLSearchParams({
        limit: "15",
        universe_limit: CANDIDATES_UNIVERSE_LIMIT,
      });
      if (frontier) candidatesQs.set("as_of", frontier);

      const [candidatesRes, regimeRes] = await Promise.all([
        fetch(`/api/engine?action=candidates&${candidatesQs.toString()}`, {
          signal: ctrl.signal,
        }),
        fetch("/api/engine?action=regime&ticker=SPY", { signal: ctrl.signal }),
      ]);

      // Parse candidates. Absent numerics stay null — the panels render "—"
      // for missing values instead of fabricating zeros.
      if (candidatesRes.ok) {
        const candidateData: CandidatesResponse = await candidatesRes.json();
        const rawTrades = Array.isArray(candidateData.trades)
          ? candidateData.trades
          : [];
        setTrades(
          rawTrades.map(
            (t): WheelTrade => ({
              ticker: typeof t.ticker === "string" ? t.ticker : "—",
              strategy: t.strategy === "covered_call" ? "covered_call" : "short_put",
              strike: num(t.strike),
              expiration:
                typeof t.expiration === "string" && t.expiration
                  ? t.expiration
                  : null,
              dte: num(t.dte),
              premium: num(t.premium),
              probProfit: num(t.probProfit),
              probProfitCiLow: num(t.probProfitCiLow),
              probProfitCiHigh: num(t.probProfitCiHigh),
              nScenarios: num(t.nScenarios),
              evDollars: num(t.evDollars),
              evPerDay: num(t.evPerDay),
              maxLoss: num(t.maxLoss),
              iv: num(t.iv),
              targetDelta: num(t.targetDelta),
              recommendation:
                typeof t.recommendation === "string" ? t.recommendation : null,
              distributionSource:
                typeof t.distributionSource === "string"
                  ? t.distributionSource
                  : null,
            })
          )
        );
      }

      // Parse regime (a VIX-band heuristic endpoint — no trend/confidence).
      if (regimeRes.ok) {
        const regimeData: RegimeResponse = await regimeRes.json();
        setRegime({
          regime: (regimeData.regime as MarketRegime["regime"]) || "NEUTRAL",
          vix: num(regimeData.vix) ?? 0,
          vixPercentile: num(regimeData.vixPercentile),
          contango:
            typeof regimeData.contango === "boolean" ? regimeData.contango : null,
          termStructure:
            typeof regimeData.termStructure === "string"
              ? regimeData.termStructure
              : null,
          vix3m: num(regimeData.vix3m),
          vix6m: num(regimeData.vix6m),
        });
      }

      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setConnected(false);
      setError(
        err instanceof Error ? err.message : "Failed to connect to engine"
      );
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, []);

  // Initial fetch + slow poll; abort any in-flight request on unmount.
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, POLL_MS);
    return () => {
      clearInterval(interval);
      abortRef.current?.abort();
    };
  }, [fetchData]);

  return {
    trades,
    regime,
    connected,
    loading,
    error,
    lastUpdated,
    dataFrontier,
    provider,
    refresh: fetchData,
  };
}

// ─── Live IBKR Book Hook (read-only /api/portfolio proxy) ─────────────

interface LiveBook {
  summary: LiveBookSummary | null;
  legs: LiveBookLeg[];
  /** Unique underlyings held (for per-name calendar/earnings lookups). */
  heldTickers: string[];
  loading: boolean;
  /** Non-null when the book endpoints are unreachable — render an honest
   *  unavailable state, never stale zeros. */
  error: string | null;
  lastUpdated: Date | null;
}

interface RawLeg {
  sym?: string;
  name?: string;
  state?: string;
  qty?: number;
  mark?: number;
  mktValue?: number;
  uPnl?: number;
  pctNavExact?: number;
  breach?: boolean;
  sector?: string;
}

export function useLiveBook(): LiveBook {
  const [summary, setSummary] = useState<LiveBookSummary | null>(null);
  const [legs, setLegs] = useState<LiveBookLeg[]>([]);
  const [heldTickers, setHeldTickers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const inFlight = useRef(false);

  const fetchBook = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const [sumRes, posRes] = await Promise.all([
        fetch("/api/portfolio/summary"),
        fetch("/api/portfolio/positions"),
      ]);
      if (!sumRes.ok || !posRes.ok) {
        throw new Error("Portfolio endpoints unavailable");
      }
      const s = await sumRes.json();
      const p = await posRes.json();

      setSummary({
        asOf: typeof s.asOf === "string" ? s.asOf : null,
        netLiq: num(s.netLiq),
        dayChangeUsd: num(s.dayChangeUsd),
        dayChangePct: num(s.dayChangePct),
        cash: num(s.cash),
        unrealizedPnl: num(s.unrealizedPnl),
        realizedYtd: num(s.realizedYtd),
        premium30d: num(s.premium30d),
        winRate: num(s.winRate),
        availableFunds: num(s.availableFunds),
        excessLiquidity: num(s.excessLiquidity),
        maintMargin: num(s.maintMargin),
        source: typeof s.source === "string" ? s.source : null,
      });

      const rawLegs: RawLeg[] = Array.isArray(p.legs) ? p.legs : [];
      const parsed = rawLegs
        .filter((l): l is RawLeg & { sym: string } => typeof l.sym === "string")
        .map(
          (l): LiveBookLeg => ({
            sym: l.sym,
            name: typeof l.name === "string" ? l.name : l.sym,
            state: typeof l.state === "string" ? l.state : "unknown",
            qty: num(l.qty),
            mark: num(l.mark),
            mktValue: num(l.mktValue),
            uPnl: num(l.uPnl),
            pctNavExact: num(l.pctNavExact),
            breach: l.breach === true,
            sector: typeof l.sector === "string" ? l.sector : null,
          })
        );
      setLegs(parsed);
      setHeldTickers([...new Set(parsed.map((l) => l.sym))]);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load live book"
      );
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBook();
    const interval = setInterval(fetchBook, POLL_MS);
    return () => clearInterval(interval);
  }, [fetchBook]);

  return { summary, legs, heldTickers, loading, error, lastUpdated };
}

// ─── Individual Ticker Analysis Hook ──────────────────────────────────

/**
 * /api/analyze payload — authority is "heuristic_diagnostic": these scores
 * are diagnostics, NOT the EV ranking. The tradeable path is /api/candidates.
 * Render with that label; never present wheelScore as a verdict.
 */
interface TickerAnalysis {
  ticker: string;
  authority?: string;
  spotPrice?: number;
  marketCap?: number;
  peRatio?: number;
  beta?: number;
  sector?: string;
  iv30d?: number;
  rv30d?: number;
  ivRank?: number;
  ivPercentile?: number;
  volRiskPremium?: number | null;
  strangleScore?: number;
  stranglePhase?: string;
  wheelScore?: number;
  wheelRecommendation?: string;
  daysToEarnings?: number | null;
  nextEarningsDate?: string | null;
  vixLevel?: number;
  riskFreeRate?: number;
}

export function useTickerAnalysis(ticker: string) {
  const [data, setData] = useState<TickerAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchAnalysis() {
      setLoading(true);
      setError(null);
      setData(null);
      try {
        const res = await window.fetch(
          `/api/engine?action=analyze&ticker=${encodeURIComponent(ticker)}`
        );
        if (cancelled) return;
        if (res.ok) {
          setData(await res.json());
        } else {
          setError(`No engine analysis for ${ticker}`);
        }
      } catch {
        if (!cancelled) setError("Engine unavailable");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchAnalysis();
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  return { data, loading, error };
}
