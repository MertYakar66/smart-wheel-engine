"use client";

// The candidate cockpit table — each row is a decision unit, read left to
// right: verdict → P&L distribution (the headline) → calibration-aware
// confidence → the mechanical terms. ev_dollars appears only as a small,
// explicitly-labelled RANKING score, never as a dollar forecast.

import { TerminalBadge } from "@/components/terminal/panel";
import type { EngineCandidate } from "@/types/cockpit";
import { fmtPct, fmtUsd, verdictVariant } from "@/lib/cockpit-trust";
import { CalibratedProb } from "./calibrated-prob";
import { DistributionBar } from "./distribution-bar";

interface CockpitTableProps {
  candidates: EngineCandidate[];
  vix: number | null;
  selectedTicker?: string | null;
  onSelect?: (c: EngineCandidate) => void;
}

export function CockpitTable({
  candidates,
  vix,
  selectedTicker,
  onSelect,
}: CockpitTableProps) {
  if (!candidates.length) {
    return (
      <div className="py-8 text-center text-[12px] text-terminal-dim">
        No candidates ranked. Check the engine connection and as_of date.
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full border-collapse text-[11px]">
        <thead>
          <tr className="border-b border-terminal-border text-[9px] uppercase tracking-wider text-terminal-dim">
            <th className="px-2 py-1 text-left">Verdict</th>
            <th className="px-2 py-1 text-left">Sym</th>
            <th className="px-2 py-1 text-left" style={{ minWidth: 180 }}>
              P&L distribution{" "}
              <span className="normal-case text-terminal-dim/70">(tail · body · breakeven)</span>
            </th>
            <th className="px-2 py-1 text-left" style={{ minWidth: 96 }}>
              Confidence
            </th>
            <th className="px-2 py-1 text-right">Strike</th>
            <th className="px-2 py-1 text-right">Prem</th>
            <th className="px-2 py-1 text-right">DTE</th>
            <th className="px-2 py-1 text-right">IV</th>
            <th className="px-2 py-1 text-right" title="Cash-secured collateral = strike × 100">
              Collat
            </th>
            <th
              className="px-2 py-1 text-right"
              title="Return on collateral = premium / strike, annualized to 365/DTE"
            >
              ROC a.
            </th>
            <th
              className="px-2 py-1 text-right"
              title="5% conditional tail loss (CVaR) — the modeled crash-scenario loss"
            >
              CVaR5
            </th>
            <th
              className="px-2 py-1 text-right text-terminal-dim"
              title="EV is a RANKING score only — ~0 correlation with realized dollars. Not a forecast."
            >
              EV·rank
            </th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => {
            const collateral = c.strike * 100;
            const rocAnn =
              c.strike > 0 && c.dte > 0
                ? (c.premium / c.strike) * (365 / c.dte)
                : null;
            const selected = selectedTicker === c.ticker;
            const earningsSoon =
              typeof c.daysToEarnings === "number" &&
              c.daysToEarnings >= 0 &&
              c.daysToEarnings <= c.dte;
            return (
              <tr
                key={c.ticker}
                onClick={() => onSelect?.(c)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect?.(c);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-label={`Open dossier for ${c.ticker} — ${c.recommendation}`}
                className={`cursor-pointer border-b border-terminal-border/40 align-middle hover:bg-terminal-border/30 focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-terminal-amber ${
                  selected ? "bg-terminal-border/40" : ""
                }`}
              >
                <td className="px-2 py-1.5">
                  <TerminalBadge variant={verdictVariant(c.recommendation)}>
                    {c.recommendation}
                  </TerminalBadge>
                </td>
                <td className="px-2 py-1.5">
                  <div className="flex items-center gap-1">
                    <span className="font-bold text-terminal-amber">{c.ticker}</span>
                    {earningsSoon && (
                      <span
                        className="text-[9px] text-terminal-red"
                        title={`earnings in ${c.daysToEarnings}d — inside the ${c.dte}d hold`}
                      >
                        E
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-2 py-1.5">
                  <DistributionBar
                    cvar5={c.cvar5}
                    p25={c.pnlP25}
                    p50={c.pnlP50}
                    p75={c.pnlP75}
                    maxProfit={c.premium * 100}
                  />
                </td>
                <td className="px-2 py-1.5">
                  <CalibratedProb
                    probProfit={c.probProfit}
                    vix={vix}
                    ciLow={c.probProfitCiLow}
                    ciHigh={c.probProfitCiHigh}
                    nScenarios={c.nScenarios}
                    distributionSource={c.distributionSource}
                  />
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                  ${c.strike.toFixed(2)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-green">
                  ${c.premium.toFixed(2)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim">
                  {c.dte}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                  {fmtPct(c.iv, 1)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim">
                  {fmtUsd(collateral)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                  {fmtPct(rocAnn, 1)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-red">
                  {fmtUsd(c.cvar5)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim/70">
                  {fmtUsd(c.evDollars)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
