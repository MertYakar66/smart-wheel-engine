"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import { useTickerAnalysis } from "@/hooks/useEngineData";

/**
 * Engine read for one underlying (action=analyze): spot, IV30/RV30, IV rank,
 * vol-risk premium, sector/beta, earnings proximity.
 *
 * HONESTY: the endpoint self-labels authority="heuristic_diagnostic". The
 * wheel/strangle scores here are diagnostics, NOT the EV ranking — the
 * tradeable path is the candidates scan (EVEngine.evaluate). The caption
 * says so; nothing here may be presented as a verdict.
 */

interface TickerAnalysisPanelProps {
  ticker: string;
  /** Freshest engine OHLCV date, for the "as of" provenance line. */
  dataFrontier: string | null;
}

function fmt(v: number | null | undefined, digits = 2, suffix = ""): string {
  return typeof v === "number" && Number.isFinite(v)
    ? `${v.toFixed(digits)}${suffix}`
    : "—";
}

export function TickerAnalysisPanel({
  ticker,
  dataFrontier,
}: TickerAnalysisPanelProps) {
  const { data, loading, error } = useTickerAnalysis(ticker);

  // IV30/RV30/VRP arrive in PERCENT units; ivRank/ivPercentile in 0-1.
  const vrp = data?.volRiskPremium;

  return (
    <TerminalPanel
      title={`${ticker} — Engine Read`}
      tag="ANALYZE"
      headerRight={
        dataFrontier ? (
          <span className="text-[10px] text-terminal-dim">
            EOD · as of {dataFrontier}
          </span>
        ) : undefined
      }
    >
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Analyzing {ticker}…
        </div>
      ) : error || !data ? (
        <div className="flex h-full flex-col items-center justify-center gap-1 text-terminal-dim">
          <span>{error || `No engine analysis for ${ticker}`}</span>
          <span className="text-[10px]">
            Likely outside the S&P 500 engine universe
          </span>
        </div>
      ) : (
        <>
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ SPOT / PROFILE
          </div>
          <div className="grid grid-cols-2 gap-x-4">
            <TerminalRow label="Spot (EOD)" value={fmt(data.spotPrice, 2)} />
            <TerminalRow label="Beta" value={fmt(data.beta, 2)} />
            <TerminalRow label="P/E" value={fmt(data.peRatio, 1)} />
            <TerminalRow
              label="Mkt cap"
              value={
                typeof data.marketCap === "number" && data.marketCap > 0
                  ? `$${(data.marketCap / 1e9).toFixed(0)}B`
                  : "—"
              }
            />
          </div>
          <TerminalRow label="Sector" value={data.sector || "—"} />

          <TerminalDivider />

          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ VOL SURFACE
          </div>
          <div className="grid grid-cols-2 gap-x-4">
            <TerminalRow label="IV 30d" value={fmt(data.iv30d, 1, "%")} />
            <TerminalRow label="RV 30d" value={fmt(data.rv30d, 1, "%")} />
            <TerminalRow
              label="IV rank"
              value={
                typeof data.ivRank === "number"
                  ? `${(data.ivRank * 100).toFixed(0)}%`
                  : "—"
              }
            />
            <TerminalRow
              label="VRP"
              value={fmt(vrp, 2, " pts")}
              valueColor={
                typeof vrp === "number"
                  ? vrp >= 0
                    ? "text-terminal-green"
                    : "text-terminal-red"
                  : "text-terminal-dim"
              }
            />
          </div>

          <TerminalDivider />

          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ EVENTS / RATES
          </div>
          <div className="grid grid-cols-2 gap-x-4">
            <TerminalRow
              label="Days to earnings"
              value={
                typeof data.daysToEarnings === "number"
                  ? `${data.daysToEarnings}d`
                  : "—"
              }
            />
            <TerminalRow
              label="Risk-free"
              value={
                typeof data.riskFreeRate === "number"
                  ? `${(data.riskFreeRate * 100).toFixed(2)}%`
                  : "—"
              }
            />
          </div>

          <TerminalDivider />

          {/* Heuristic diagnostics — explicitly NOT the EV authority. */}
          <div className="mb-1 flex items-center gap-2">
            <span className="text-[10px] font-bold text-terminal-blue">
              ─ HEURISTIC DIAGNOSTICS
            </span>
            <TerminalBadge variant="amber">NOT EV AUTHORITY</TerminalBadge>
          </div>
          <div className="grid grid-cols-2 gap-x-4">
            <TerminalRow
              label="Wheel score"
              value={fmt(data.wheelScore, 0)}
            />
            <TerminalRow
              label="Wheel rec"
              value={data.wheelRecommendation || "—"}
            />
            <TerminalRow
              label="Strangle score"
              value={fmt(data.strangleScore, 0)}
            />
            <TerminalRow
              label="Strangle phase"
              value={data.stranglePhase || "—"}
            />
          </div>
          <div className="mt-1 text-[10px] text-terminal-dim">
            Diagnostics only — the tradeable ranking is the EV candidates scan
            (Options Engine panel).
          </div>
        </>
      )}
    </TerminalPanel>
  );
}
