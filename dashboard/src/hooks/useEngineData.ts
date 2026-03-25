"use client";

import { useState, useEffect, useCallback } from "react";
import type { WheelTrade, MarketRegime, OptionsPortfolio } from "@/types";

// ─── Engine API Response Types ─────────────────────────────────────────

interface VolEdgeData {
  ticker: string;
  iv_rv_spread: number | null;
  iv_rv_ratio: number | null;
  edge_score: number | null;
  vrp_percentile: number | null;
  vol_regime: number | null;
  iv_rank: number | null;
  rv_21d: number | null;
  atm_iv: number | null;
  date: string | null;
  error?: string;
}

interface RegimeData {
  ticker: string;
  trend_regime: number | null;
  trend_strength: number | null;
  vol_regime: number | null;
  vol_regime_percentile: number | null;
  liquidity_regime: number | null;
  regime_score: number | null;
  position_scalar: number | null;
  target_delta_adj: number | null;
  target_dte_adj: number | null;
  rv_21d?: number | null;
  close?: number | null;
  date: string | null;
  error?: string;
}

interface CandidateData {
  ticker: string;
  edge_score: number;
  iv_rv_spread: number;
  iv_rank: number;
  vol_regime: number;
}

interface EngineStatus {
  storage: {
    total_files: number;
    total_size_mb: number;
  };
  categories: Record<string, number>;
  total_features: number;
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

// ─── Regime Mapping ────────────────────────────────────────────────────

function mapRegime(regimeData: RegimeData | null): MarketRegime {
  if (!regimeData || regimeData.error) {
    return {
      regime: "NEUTRAL",
      vix: 15,
      trendScore: 0,
      confidence: 0,
    };
  }

  // Map trend_regime (0=down, 1=neutral, 2=up) and vol_regime to our regime types
  const trendRegime = regimeData.trend_regime ?? 1;
  const volRegime = regimeData.vol_regime ?? 1;
  const volPercentile = regimeData.vol_regime_percentile ?? 50;

  let regime: MarketRegime["regime"] = "NEUTRAL";

  // High vol regime takes precedence
  if (volRegime >= 2 && volPercentile > 70) {
    regime = "HIGH_VOL";
  } else if (trendRegime === 2) {
    regime = "BULL";
  } else if (trendRegime === 0) {
    regime = "BEAR";
  }

  // Trend strength maps to trend score (-1 to 1)
  const trendStrength = regimeData.trend_strength ?? 0.5;
  const trendScore = trendRegime === 0
    ? -trendStrength
    : trendRegime === 2
      ? trendStrength
      : 0;

  // Confidence based on regime score and trend strength
  const regimeScore = regimeData.regime_score ?? 50;
  const confidence = Math.min(1, (regimeScore / 100) * 0.5 + trendStrength * 0.5);

  // VIX approximation from realized volatility (annualized)
  const rv21d = regimeData.rv_21d ?? 0.20;
  const vix = rv21d * 100; // Convert to VIX-like scale

  return {
    regime,
    vix,
    trendScore,
    confidence,
  };
}

// ─── Trade Candidate Mapping ───────────────────────────────────────────

function mapCandidatesToTrades(
  candidates: CandidateData[],
  regimeData: RegimeData | null
): WheelTrade[] {
  // Get target parameters from regime
  const targetDelta = regimeData?.target_delta_adj ?? 0.3;
  const targetDTE = regimeData?.target_dte_adj ?? 30;

  return candidates.slice(0, 8).map((c) => {
    // Edge score determines strategy: high edge = sell premium (short put)
    // Vol regime affects strike selection
    const isHighIV = c.iv_rank > 70;
    const strategy = isHighIV ? "short_put" : "covered_call";

    // Estimate strike based on typical wheel strategy
    // This would be replaced with real options chain data
    const basePrice = 100; // Placeholder - would need real price data
    const strikeMultiplier = strategy === "short_put" ? 0.95 : 1.05;
    const strike = Math.round(basePrice * strikeMultiplier);

    // Calculate expiration date
    const expDate = new Date();
    expDate.setDate(expDate.getDate() + targetDTE);
    const expiration = expDate.toISOString().split("T")[0];

    // Estimate premium based on IV spread (higher spread = more premium)
    const ivSpread = Math.abs(c.iv_rv_spread) * 100;
    const premium = Math.round(50 + ivSpread * 5); // $50 base + IV spread premium

    // Probability of profit: inversely related to delta
    const delta = strategy === "short_put" ? -targetDelta : targetDelta;
    const probability = Math.round((1 - Math.abs(targetDelta)) * 100);

    // Expected P&L based on edge score
    const expectedPnL = Math.round(premium * (c.edge_score / 100));

    // Max loss for short put
    const maxLoss = strategy === "short_put"
      ? Math.round((strike - premium / 100) * 100)
      : 0;

    // IV from iv_rank (normalized to decimal)
    const iv = c.iv_rank / 100 * 0.5; // Scale to typical IV range

    return {
      ticker: c.ticker,
      strategy,
      strike,
      expiration,
      premium,
      probability,
      expectedPnL,
      maxLoss,
      iv,
      delta,
      score: Math.round(c.edge_score),
    };
  });
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
      // Fetch all data in parallel
      const [statusRes, candidatesRes, regimeRes] = await Promise.all([
        fetch("/api/engine?action=status"),
        fetch("/api/engine?action=candidates&limit=10"),
        fetch("/api/engine?action=regime&ticker=SPY"), // Use SPY as market proxy
      ]);

      // Check if engine is available
      if (!statusRes.ok) {
        throw new Error("Engine unavailable");
      }

      const status: EngineStatus = await statusRes.json();

      // If we have features, engine is connected
      if (status.total_features > 0) {
        setConnected(true);

        // Parse candidates
        const candidates: CandidateData[] = candidatesRes.ok
          ? await candidatesRes.json()
          : [];

        // Parse regime
        const regimeData: RegimeData | null = regimeRes.ok
          ? await regimeRes.json()
          : null;

        // Map to UI types
        setRegime(mapRegime(regimeData));
        setTrades(mapCandidatesToTrades(candidates, regimeData));

        // Portfolio stats from feature store stats
        setPortfolio({
          openPositions: 0, // Would come from position tracking
          totalPremiumCollected: 0, // Would come from trade history
          winRate: 0, // Would come from backtest results
          avgDaysHeld: regimeData?.target_dte_adj ?? 30,
        });

        setLastUpdated(new Date());
      } else {
        setConnected(false);
        setError("No features computed yet");
      }
    } catch (err) {
      setConnected(false);
      setError(err instanceof Error ? err.message : "Failed to connect to engine");
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

// ─── Individual Ticker Hook ────────────────────────────────────────────

export function useTickerVolEdge(ticker: string) {
  const [data, setData] = useState<VolEdgeData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetch() {
      setLoading(true);
      try {
        const res = await window.fetch(`/api/engine?action=vol_edge&ticker=${ticker}`);
        if (res.ok) {
          setData(await res.json());
        }
      } catch {
        // silent
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, [ticker]);

  return { data, loading };
}
