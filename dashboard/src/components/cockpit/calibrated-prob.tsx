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
  fmtN,
  fmtProbCi,
  num,
  samplingCiHonest,
  type ConfidenceTrust,
} from "@/lib/cockpit-trust";

interface CalibratedProbProps {
  probProfit: number;
  vix: number | null;
  /** Wilson 95% sampling-CI bounds + scenario count (0-1). Optional: absent on
   *  older engine payloads, in which case only the bare dot is drawn. */
  ciLow?: number | null;
  ciHigh?: number | null;
  nScenarios?: number | null;
  /** forward_distribution source label. The CI is only an honest sampling
   *  spread on the IID non-overlapping tier — see samplingCiHonest(). */
  distributionSource?: string | null;
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

export function CalibratedProb({
  probProfit,
  vix,
  ciLow,
  ciHigh,
  nScenarios,
  distributionSource,
}: CalibratedProbProps) {
  const trust = confidenceTrust(probProfit, vix);
  const note = calibrationNote(probProfit, trust);
  const pct = Math.round(probProfit * 100);
  const dotPct = Math.max(0, Math.min(100, probProfit * 100));
  const showGhost = trust === "hard-caution";
  const ghostPct = TOP_BIN_REALIZED * 100;

  // Wilson 95% SAMPLING CI — the sampling uncertainty of the k/N forward-
  // scenario frequency, orthogonal to the calibration trust above. The band on
  // the track makes "few scenarios -> wide, low-confidence estimate" visible.
  // Shown ONLY on the IID non-overlapping tier (samplingCiHonest): on
  // bootstrap/synthetic tiers the same field is a 5000-draw count whose Wilson
  // interval is false precision, so we suppress it and degrade to the bare dot
  // — same path as an absent CI on older payloads.
  const ciOk = samplingCiHonest(distributionSource);
  const lo = ciOk ? num(ciLow) : null;
  const hi = ciOk ? num(ciHigh) : null;
  const ci = lo !== null && hi !== null ? fmtProbCi(lo, hi) : "";
  const showBand = lo !== null && hi !== null;
  // Band geometry uses the exact (ordered) bounds; the caption text widens them.
  const bandLeft =
    showBand ? Math.max(0, Math.min(100, Math.min(lo, hi) * 100)) : 0;
  const bandRight =
    showBand ? Math.max(0, Math.min(100, Math.max(lo, hi) * 100)) : 0;
  const bandWidth = Math.max(0, bandRight - bandLeft);
  const nStr = ciOk ? fmtN(nScenarios) : null;

  const ciTitle = showBand
    ? ` · Wilson 95% sampling CI [${ci}]${nStr ? ` from ${nStr} windows` : ""}`
    : "";
  // With the VIX unknown, the R11 (elevated-vol) leg of the trust read cannot
  // resolve — say so instead of implying calm vol was confirmed.
  const vixUnknown = typeof vix !== "number";
  const title =
    (trust === "trust"
      ? `prob_profit ${probProfit.toFixed(3)} — mid-range, well-calibrated (trust).`
      : `${note}`) +
    (vixUnknown && trust !== "trust"
      ? " · VIX unavailable — R11 state unknown, treat as at-least-cautious"
      : "") +
    ciTitle;

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
        {/* Wilson 95% sampling-CI band — width encodes how few windows back the
            dot (regime-independent; complements the calibration ghost). Must
            stay BEFORE the ghost/dot below so the opaque dot paints on top and
            is never occluded when the point estimate sits on the band edge. */}
        {showBand && (
          <div
            className="absolute -inset-y-0.5 rounded-full border border-terminal-text/25 bg-terminal-text/10"
            style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}
            title={`Wilson 95% sampling CI [${ci}]`}
          />
        )}
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
      {ci && (
        <span
          className="text-[9px] leading-tight text-terminal-dim tabular-nums"
          title="Wilson 95% sampling CI for prob_profit (independent forward windows)"
        >
          sampling 95% CI [{ci}]{nStr ? ` · ${nStr}` : ""}
        </span>
      )}
      {showGhost && (
        <span className="text-[9px] leading-tight text-terminal-red/80">
          realized ~{Math.round(TOP_BIN_REALIZED * 100)}% in crisis
        </span>
      )}
    </div>
  );
}
