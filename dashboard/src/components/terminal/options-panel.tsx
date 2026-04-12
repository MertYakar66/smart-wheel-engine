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
  ELEVATED: "amber",
  LOW_VOL: "green",
  "---": "blue",
};

const REGIME_ARROWS: Record<string, string> = {
  BULL: "↑",
  BEAR: "↓",
  NEUTRAL: "→",
  HIGH_VOL: "⚡",
  ELEVATED: "⚡",
  LOW_VOL: "─",
  "---": "...",
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
          <TerminalBadge variant="amber">OFFLINE</TerminalBadge>
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

      {/* Portfolio Stats — collapse when empty to stop wasting real estate */}
      {portfolio.openPositions > 0 ? (
        <>
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
        </>
      ) : (
        <>
          <div className="py-1 text-[10px] text-terminal-dim italic">
            No active positions — scan below to find entries
          </div>
          <TerminalDivider />
        </>
      )}

      {/* Wheel Scanner - Trade Candidates */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ WHEEL SCANNER
      </div>
      <div
        className="grid gap-x-1 py-[1px] text-[10px] text-terminal-dim"
        style={{
          gridTemplateColumns:
            "minmax(42px,1fr) 28px minmax(52px,1fr) minmax(42px,1fr) minmax(56px,1fr) minmax(36px,1fr)",
        }}
      >
        <span>SYM</span>
        <span>TYPE</span>
        <span className="text-right">STRIKE</span>
        <span className="text-right">PROB</span>
        <span className="text-right">EV</span>
        <span className="text-right">SCR</span>
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
            className="grid gap-x-1 py-[2px] items-center hover:bg-terminal-border/30"
            style={{
              gridTemplateColumns:
                "minmax(42px,1fr) 28px minmax(52px,1fr) minmax(42px,1fr) minmax(56px,1fr) minmax(36px,1fr)",
            }}
          >
            <span className="text-terminal-amber truncate">{trade.ticker}</span>
            <span>
              <TerminalBadge
                variant={trade.strategy === "short_put" ? "blue" : "green"}
              >
                {trade.strategy === "short_put" ? "SP" : "CC"}
              </TerminalBadge>
            </span>
            <span className="text-right text-terminal-text tabular-nums">
              ${trade.strike}
            </span>
            <span
              className={`text-right tabular-nums ${
                trade.probability >= 75
                  ? "text-terminal-green"
                  : trade.probability >= 60
                    ? "text-terminal-amber"
                    : "text-terminal-red"
              }`}
            >
              {trade.probability.toFixed(0)}%
            </span>
            <span
              className={`text-right tabular-nums ${
                trade.expectedPnL >= 0
                  ? "text-terminal-green"
                  : "text-terminal-red"
              }`}
            >
              ${trade.expectedPnL.toFixed(2)}
            </span>
            <span
              className={`text-right font-bold tabular-nums ${
                trade.score >= 80
                  ? "text-terminal-green"
                  : trade.score >= 60
                    ? "text-terminal-amber"
                    : "text-terminal-text"
              }`}
            >
              {trade.score.toFixed(1)}
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
          <TerminalRow label="IV" value={`${trades[0].iv.toFixed(1)}%`} />
          <TerminalRow
            label="Premium"
            value={`$${trades[0].premium.toFixed(2)}`}
            valueColor="text-terminal-green"
          />
          <TerminalRow
            label="Max Loss"
            value={`$${trades[0].maxLoss.toLocaleString()}`}
            valueColor="text-terminal-red"
          />
        </div>
      ) : (
        <span className="text-terminal-dim">No data</span>
      )}
    </TerminalPanel>
  );
}
