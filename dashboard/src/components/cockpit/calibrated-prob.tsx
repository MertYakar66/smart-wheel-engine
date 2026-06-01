// Calibration-aware prob_profit indicator.
//
// A 0-1 track with the engine's prob_profit as a dot. The colour and an
// optional "ghost" marker encode trust:
//   * mid-range (0.60-0.90) -> green dot, trusted.
//   * top bin (> 0.90), calm vol -> amber, "optimistic".
//   * top bin (> 0.90), elevated vol (VIX > 25, the R11 regime) -> red dot
//     PLUS a ghost marker at the crisis-realized ~0.57, with the gap drawn.
//
// Never renders a confident green "96%". The whole point is to make the
// untrustworthy top bin look cautious.

import {
  TOP_BIN_REALIZED,
  calibrationNote,
  confidenceTrust,
  type ConfidenceTrust,
} from "@/lib/cockpit-trust";

interface CalibratedProbProps {
  probProfit: number;
  vix: number | null;
}

const DOT_COLOR: Record<ConfidenceTrust, string> = {
  trust: "bg-terminal-green border-terminal-green",
  "soft-caution": "bg-terminal-amber border-terminal-amber",
  "hard-caution": "bg-terminal-red border-terminal-red",
};
const TEXT_COLOR: Record<ConfidenceTrust, string> = {
  trust: "text-terminal-green",
  "soft-caution": "text-terminal-amber",
  "hard-caution": "text-terminal-red",
};

export function CalibratedProb({ probProfit, vix }: CalibratedProbProps) {
  const trust = confidenceTrust(probProfit, vix);
  const note = calibrationNote(probProfit, trust);
  const pct = Math.round(probProfit * 100);
  const dotPct = Math.max(0, Math.min(100, probProfit * 100));
  const showGhost = trust === "hard-caution";
  const ghostPct = TOP_BIN_REALIZED * 100;

  const title =
    trust === "trust"
      ? `prob_profit ${probProfit.toFixed(3)} — mid-range, well-calibrated (trust).`
      : `${note}`;

  return (
    <div className="flex flex-col gap-0.5" title={title}>
      <div className="flex items-center justify-between gap-1">
        <span className={`text-[11px] font-bold tabular-nums ${TEXT_COLOR[trust]}`}>
          {pct}%
        </span>
        {trust !== "trust" && (
          <span className="text-[9px] uppercase tracking-wide text-terminal-dim">
            {trust === "hard-caution" ? "⚠ top-bin" : "top-bin"}
          </span>
        )}
      </div>
      <div className="relative h-1.5 w-full rounded-full bg-terminal-border">
        {/* trusted mid-range band 0.60-0.90 */}
        <div
          className="absolute inset-y-0 rounded-full bg-terminal-green/15"
          style={{ left: "60%", width: "30%" }}
        />
        {/* ghost: crisis-realized ~0.57 for the top bin */}
        {showGhost && (
          <div
            className="absolute -top-0.5 h-2.5 w-[2px] bg-terminal-dim"
            style={{ left: `${ghostPct}%` }}
            title={`crisis-realized ~${TOP_BIN_REALIZED.toFixed(2)}`}
          />
        )}
        {/* the engine's prob_profit dot */}
        <div
          className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border ${DOT_COLOR[trust]}`}
          style={{ left: `${dotPct}%` }}
        />
      </div>
      {showGhost && (
        <span className="text-[9px] leading-tight text-terminal-red/80">
          realized ~{Math.round(TOP_BIN_REALIZED * 100)}% in crisis
        </span>
      )}
    </div>
  );
}
