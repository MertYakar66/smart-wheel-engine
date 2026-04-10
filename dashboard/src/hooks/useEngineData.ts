"use client";

import { useState, useEffect, useCallback } from "react";
import type { WheelTrade, MarketRegime, OptionsPortfolio } from "@/types";

// ─── Engine API Response Types ─────────────────────────────────────────

interface CandidatesResponse {
  trades: WheelTrade[];
  count: number;
}

interface RegimeResponse {
  regime: string;
  vix: number;
  vixPercentile: number;
  contango: boolean;
  trendScore: number;
  confidence: number;
}

interface StatusResponse {
  status: string;
  engine: string;
  universe_size: number;
  vix: number;
  error?: string;
}

// ─── Hook Return Type ──────────────────────────────────────────────────

interface EngineData {
  trades: WheelTrade[];
  regime: MarketRegime;
  portfolio: OptionsPortfolio;
  connected: boolean;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

// ─── Main Hook ─────────────────────────────────────────────────────────

export function useEngineData(): EngineData {
  const [trades, setTrades] = useState<WheelTrade[]>([]);
  const [regime, setRegime] = useState<MarketRegime>({
    regime: "NEUTRAL",
    vix: 15,
    trendScore: 0,
    confidence: 0,
  });
  const [portfolio, setPortfolio] = useState<OptionsPortfolio>({
    openPositions: 0,
    totalPremiumCollected: 0,
    winRate: 0,
    avgDaysHeld: 0,
  });
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch status and candidates in parallel
      const [statusRes, candidatesRes, regimeRes] = await Promise.all([
        fetch("/api/engine?action=status"),
        fetch("/api/engine?action=candidates&limit=15&min_score=50"),
        fetch("/api/engine?action=regime&ticker=SPY"),
      ]);

      // Check engine availability
      if (!statusRes.ok) {
        throw new Error("Engine API unavailable. Start with: python engine_api.py");
      }

      const status: StatusResponse = await statusRes.json();

      if (status.error) {
        throw new Error(status.error);
      }

      setConnected(status.status === "connected");

      // Parse candidates
      if (candidatesRes.ok) {
        const candidateData: CandidatesResponse = await candidatesRes.json();
        setTrades(candidateData.trades || []);
      }

      // Parse regime
      if (regimeRes.ok) {
        const regimeData: RegimeResponse = await regimeRes.json();
        setRegime({
          regime: (regimeData.regime as MarketRegime["regime"]) || "NEUTRAL",
          vix: regimeData.vix || 0,
          trendScore: regimeData.trendScore || 0,
          confidence: regimeData.confidence || 0,
        });
      }

      setLastUpdated(new Date());
    } catch (err) {
      setConnected(false);
      setError(
        err instanceof Error ? err.message : "Failed to connect to engine"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return {
    trades,
    regime,
    portfolio,
    connected,
    loading,
    error,
    lastUpdated,
    refresh: fetchData,
  };
}

// ─── Individual Ticker Analysis Hook ──────────────────────────────────

interface TickerAnalysis {
  ticker: string;
  spotPrice: number;
  marketCap: number;
  peRatio: number;
  beta: number;
  sector: string;
  iv30d: number;
  rv30d: number;
  ivRank: number;
  volRiskPremium: number;
  strangleScore: number;
  stranglePhase: string;
  wheelScore: number;
  wheelRecommendation: string;
  daysToEarnings: number | null;
  nextEarningsDate: string | null;
  vixLevel: number;
  riskFreeRate: number;
}

export function useTickerAnalysis(ticker: string) {
  const [data, setData] = useState<TickerAnalysis | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAnalysis() {
      setLoading(true);
      try {
        const res = await window.fetch(
          `/api/engine?action=analyze&ticker=${ticker}`
        );
        if (res.ok) {
          setData(await res.json());
        }
      } catch {
        // silent
      } finally {
        setLoading(false);
      }
    }
    fetchAnalysis();
  }, [ticker]);

  return { data, loading };
}

// ─── Committee Review Hook ────────────────────────────────────────────

interface CommitteeResult {
  ticker: string;
  judgment: string;
  reasoning: string;
  confidence: string;
  approvals: number;
  rejections: number;
  neutrals: number;
  advisors: Array<{
    name: string;
    judgment: string;
    summary: string;
    keyReasons: string[];
    confidence: string;
  }>;
  risksUnresolved: string[];
  requiredActions: string[];
  report: string;
}

export function useCommitteeReview(ticker: string | null) {
  const [data, setData] = useState<CommitteeResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) return;

    async function fetchReview() {
      setLoading(true);
      try {
        const res = await window.fetch(
          `/api/engine?action=committee&ticker=${ticker}`
        );
        if (res.ok) {
          setData(await res.json());
        }
      } catch {
        // silent
      } finally {
        setLoading(false);
      }
    }
    fetchReview();
  }, [ticker]);

  return { data, loading };
}
