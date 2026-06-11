"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import { confidenceTrust, calibrationNote } from "@/lib/cockpit-trust";
import type { WheelTrade, MarketRegime } from "@/types";

interface OptionsPanelProps {
  trades: WheelTrade[];
  regime: MarketRegime;
  connected: boolean;
  /** One-shot highlight when the command line targets this panel. */
  flash?: boolean;
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

const SCANNER_COLS =
  "minmax(42px,1fr) 28px minmax(52px,1fr) minmax(42px,1fr) minmax(52px,1fr) minmax(48px,1fr)";

function fmtUsd0(v: number | null): string {
  if (v === null) return "—";
  return `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

/**
 * Calibration-aware prob_profit color, mirroring the cockpit's trust
 * treatment: the >0.90 top bin is over-confident (crisis-realized ~0.57), so
 * it renders amber (red when VIX > 25, the R11 regime) — never confident green.
 */
function probColor(probProfit: number, vix: number): string {
  const trust = confidenceTrust(probProfit, vix);
  if (trust === "hard-caution") return "text-terminal-red";
  if (trust === "soft-caution") return "text-terminal-amber";
  if (probProfit >= 0.6) return "text-terminal-green";
  return "text-terminal-red";
}

export function OptionsPanel({
  trades,
  regime,
  connected,
  flash,
}: OptionsPanelProps) {
  const top = trades.length > 0 ? trades[0] : null;
  const topTrust = top?.probProfit
    ? confidenceTrust(top.probProfit, regime.vix)
    : "trust";

  return (
    <TerminalPanel
      title="Options Engine"
      tag="WHEEL"
      flash={flash}
      headerRight={
        connected ? (
          <TerminalBadge variant="green">CONNECTED</TerminalBadge>
        ) : (
          <TerminalBadge variant="amber">OFFLINE</TerminalBadge>
        )
      }
    >
      {/* Market regime — honest label: the endpoint is a VIX-band heuristic.
          trendScore/confidence were fabricated server constants (removed). */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ REGIME (VIX BAND)
      </div>
      <div className="mb-1 flex items-center gap-2">
        <TerminalBadge variant={REGIME_COLORS[regime.regime] ?? "blue"}>
          {regime.regime}
        </TerminalBadge>
        {regime.termStructure && (
          <span
            className={`text-[10px] ${
              regime.contango === false
                ? "text-terminal-red"
                : "text-terminal-dim"
            }`}
          >
            {regime.termStructure}
          </span>
        )}
      </div>
      <TerminalRow
        label="VIX"
        value={regime.vix > 0 ? regime.vix.toFixed(2) : "—"}
      />
      <TerminalRow
        label="VIX 1y pctile"
        value={
          regime.vixPercentile !== null
            ? `${(regime.vixPercentile * 100).toFixed(0)}%`
            : "—"
        }
      />

      <TerminalDivider />

      {/* Wheel Scanner — EV-ranked candidates. EV and EV/D are per-contract
          dollars; PROB is the engine prob_profit with calibration coloring. */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ WHEEL SCANNER (EV-RANKED)
      </div>
      <div
        className="grid gap-x-1 py-[1px] text-[10px] text-terminal-dim"
        style={{ gridTemplateColumns: SCANNER_COLS }}
      >
        <span>SYM</span>
        <span>TYPE</span>
        <span className="text-right">STRIKE</span>
        <span className="text-right">PROB</span>
        <span className="text-right">EV</span>
        <span className="text-right">EV/D</span>
      </div>
      <TerminalDivider />

      {trades.length === 0 ? (
        <div className="py-2 text-center text-terminal-dim">
          No trade candidates. Connect smart-wheel-engine to scan.
        </div>
      ) : (
        trades.map((trade, i) => (
          <div
            key={`${trade.ticker}-${i}`}
            className="grid items-center gap-x-1 py-[2px] hover:bg-terminal-border/30"
            style={{ gridTemplateColumns: SCANNER_COLS }}
            title={
              trade.probProfit !== null
                ? (calibrationNote(
                    trade.probProfit,
                    confidenceTrust(trade.probProfit, regime.vix)
                  ) ?? undefined)
                : undefined
            }
          >
            <span className="truncate text-terminal-amber">{trade.ticker}</span>
            <span>
              <TerminalBadge
                variant={trade.strategy === "short_put" ? "blue" : "green"}
              >
                {trade.strategy === "short_put" ? "SP" : "CC"}
              </TerminalBadge>
            </span>
            <span className="text-right tabular-nums text-terminal-text">
              {trade.strike !== null ? `$${trade.strike}` : "—"}
            </span>
            <span
              className={`text-right tabular-nums ${
                trade.probProfit !== null
                  ? probColor(trade.probProfit, regime.vix)
                  : "text-terminal-dim"
              }`}
            >
              {trade.probProfit !== null
                ? `${(trade.probProfit * 100).toFixed(0)}%`
                : "—"}
            </span>
            <span
              className={`text-right tabular-nums ${
                trade.evDollars === null
                  ? "text-terminal-dim"
                  : trade.evDollars >= 0
                    ? "text-terminal-green"
                    : "text-terminal-red"
              }`}
            >
              {trade.evDollars !== null ? `$${trade.evDollars.toFixed(0)}` : "—"}
            </span>
            <span
              className={`text-right tabular-nums ${
                trade.evPerDay === null
                  ? "text-terminal-dim"
                  : trade.evPerDay >= 0
                    ? "text-terminal-green"
                    : "text-terminal-red"
              }`}
            >
              {trade.evPerDay !== null ? `$${trade.evPerDay.toFixed(1)}` : "—"}
            </span>
          </div>
        ))
      )}

      <TerminalDivider />

      {/* Top pick detail. Honest labels: targetDelta is the strike-SELECTION
          target (an input, not a measured Greek); expiration is modeled
          (as_of + dte); premium is per-share, max loss per-contract. */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ TOP PICK
      </div>
      {top ? (
        <>
          <div className="grid grid-cols-2 gap-x-4">
            <TerminalRow
              label="Target Δ"
              value={
                top.targetDelta !== null ? top.targetDelta.toFixed(2) : "—"
              }
            />
            <TerminalRow
              label="IV"
              value={top.iv !== null ? `${(top.iv * 100).toFixed(1)}%` : "—"}
            />
            <TerminalRow
              label="Premium /sh"
              value={top.premium !== null ? `$${top.premium.toFixed(2)}` : "—"}
              valueColor="text-terminal-green"
            />
            <TerminalRow
              label="Max loss /ct"
              value={fmtUsd0(top.maxLoss)}
              valueColor="text-terminal-red"
            />
            <TerminalRow
              label="~Exp (modeled)"
              value={top.expiration ? `~${top.expiration}` : "—"}
            />
            <TerminalRow
              label="DTE"
              value={top.dte !== null ? `${top.dte}d` : "—"}
            />
          </div>
          {/* Wilson sampling CI + N — the honesty fields the payload already
              carries; a bare "97%" without them overstates precision. */}
          {top.probProfit !== null &&
            top.probProfitCiLow !== null &&
            top.probProfitCiHigh !== null && (
              <div className="mt-0.5 text-[10px] text-terminal-dim">
                prob {(top.probProfit * 100).toFixed(0)}% (CI{" "}
                {(top.probProfitCiLow * 100).toFixed(0)}–
                {(top.probProfitCiHigh * 100).toFixed(0)}
                {top.nScenarios !== null ? `, N=${top.nScenarios}` : ""})
              </div>
            )}
          {top.probProfit !== null && topTrust !== "trust" && (
            <div
              className={`mt-0.5 text-[10px] ${
                topTrust === "hard-caution"
                  ? "text-terminal-red"
                  : "text-terminal-amber"
              }`}
            >
              {calibrationNote(top.probProfit, topTrust)}
            </div>
          )}
        </>
      ) : (
        <span className="text-terminal-dim">No data</span>
      )}
    </TerminalPanel>
  );
}
