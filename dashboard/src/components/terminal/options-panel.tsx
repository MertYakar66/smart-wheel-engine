"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import type { WheelTrade, MarketRegime, OptionsPortfolio } from "@/types";

interface OptionsPanelProps {
  trades: WheelTrade[];
  regime: MarketRegime;
  portfolio: OptionsPortfolio;
  connected: boolean;
}

const REGIME_COLORS: Record<string, "green" | "red" | "amber" | "blue"> = {
  BULL: "green",
  BEAR: "red",
  NEUTRAL: "blue",
  HIGH_VOL: "amber",
};

const REGIME_ARROWS: Record<string, string> = {
  BULL: "↑",
  BEAR: "↓",
  NEUTRAL: "→",
  HIGH_VOL: "⚡",
};

export function OptionsPanel({
  trades,
  regime,
  portfolio,
  connected,
}: OptionsPanelProps) {
  return (
    <TerminalPanel
      title="Options Engine"
      tag="WHEEL"
      headerRight={
        connected ? (
          <TerminalBadge variant="green">CONNECTED</TerminalBadge>
        ) : (
          <TerminalBadge variant="amber">PLACEHOLDER</TerminalBadge>
        )
      }
    >
      {/* Market Regime */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ MARKET REGIME
      </div>
      <div className="flex items-center gap-2 mb-1">
        <TerminalBadge variant={REGIME_COLORS[regime.regime]}>
          {regime.regime} {REGIME_ARROWS[regime.regime]}
        </TerminalBadge>
        <span className="text-terminal-dim text-[10px]">
          Conf: {(regime.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <TerminalRow label="VIX" value={regime.vix.toFixed(2)} />
      <TerminalRow
        label="Trend Score"
        value={
          (regime.trendScore >= 0 ? "+" : "") + regime.trendScore.toFixed(2)
        }
        valueColor={
          regime.trendScore >= 0 ? "text-terminal-green" : "text-terminal-red"
        }
      />

      <TerminalDivider />

      {/* Portfolio Stats */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ PORTFOLIO
      </div>
      <TerminalRow label="Open Positions" value={portfolio.openPositions} />
      <TerminalRow
        label="Premium Collected"
        value={`$${portfolio.totalPremiumCollected.toLocaleString()}`}
        valueColor="text-terminal-green"
      />
      <TerminalRow
        label="Win Rate"
        value={`${portfolio.winRate.toFixed(1)}%`}
        valueColor={
          portfolio.winRate >= 70
            ? "text-terminal-green"
            : "text-terminal-amber"
        }
      />
      <TerminalRow label="Avg Days Held" value={portfolio.avgDaysHeld} />

      <TerminalDivider />

      {/* Wheel Scanner - Trade Candidates */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ WHEEL SCANNER
      </div>
      <div className="flex items-center justify-between py-[1px] text-[10px] text-terminal-dim">
        <span className="w-12">SYM</span>
        <span className="w-14">TYPE</span>
        <span className="w-12 text-right">STRIKE</span>
        <span className="w-12 text-right">PROB</span>
        <span className="w-14 text-right">EV</span>
        <span className="w-10 text-right">SCR</span>
      </div>
      <TerminalDivider />

      {trades.length === 0 ? (
        <div className="py-2 text-center text-terminal-dim">
          No trade candidates. Connect smart-wheel-engine to scan.
        </div>
      ) : (
        trades.map((trade, i) => (
          <div
            key={i}
            className="flex items-center justify-between py-[2px] hover:bg-terminal-border/30"
          >
            <span className="w-12 text-terminal-amber">{trade.ticker}</span>
            <span className="w-14">
              <TerminalBadge
                variant={
                  trade.strategy === "short_put" ? "blue" : "green"
                }
              >
                {trade.strategy === "short_put" ? "SP" : "CC"}
              </TerminalBadge>
            </span>
            <span className="w-12 text-right text-terminal-text">
              ${trade.strike}
            </span>
            <span
              className={`w-12 text-right ${
                trade.probability >= 75
                  ? "text-terminal-green"
                  : trade.probability >= 60
                  ? "text-terminal-amber"
                  : "text-terminal-red"
              }`}
            >
              {trade.probability}%
            </span>
            <span
              className={`w-14 text-right ${
                trade.expectedPnL >= 0
                  ? "text-terminal-green"
                  : "text-terminal-red"
              }`}
            >
              ${trade.expectedPnL}
            </span>
            <span
              className={`w-10 text-right font-bold ${
                trade.score >= 80
                  ? "text-terminal-green"
                  : trade.score >= 60
                  ? "text-terminal-amber"
                  : "text-terminal-text"
              }`}
            >
              {trade.score}
            </span>
          </div>
        ))
      )}

      <TerminalDivider />

      {/* Greeks Summary */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ GREEKS (TOP PICK)
      </div>
      {trades.length > 0 ? (
        <div className="grid grid-cols-2 gap-x-4">
          <TerminalRow label="Delta" value={trades[0].delta.toFixed(3)} />
          <TerminalRow label="IV" value={`${(trades[0].iv * 100).toFixed(1)}%`} />
          <TerminalRow
            label="Premium"
            value={`$${trades[0].premium}`}
            valueColor="text-terminal-green"
          />
          <TerminalRow
            label="Max Loss"
            value={`$${trades[0].maxLoss}`}
            valueColor="text-terminal-red"
          />
        </div>
      ) : (
        <span className="text-terminal-dim">No data</span>
      )}
    </TerminalPanel>
  );
}
