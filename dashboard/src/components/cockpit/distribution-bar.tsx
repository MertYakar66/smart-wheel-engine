// P&L distribution bar — the cockpit's headline per-row visual.
//
// Renders the engine's modeled outcome SHAPE, not a point estimate:
//   cvar5 ......... p25 [ p50 ] p75        with the breakeven (zero) line.
//
// For a short put the upper quartiles usually pin at the "keep full premium"
// ceiling (p25 == p50 == p75), so the interquartile box collapses to a sliver
// and the LONG red whisker out to cvar5 dominates — which is the point: the
// risk is in the tail, not in the body. ev_dollars (the point estimate) is
// deliberately NOT drawn here.
//
// cvar5 is the MEAN of the worst-5% scenarios and CAN be positive (observed
// live: ADI +922) — then the whisker is a PROFITABLE tail, drawn neutral
// (never loss-red) with the wording adjusted. Sign honesty over decoration.

interface DistributionBarProps {
  cvar5: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  /** Optional max-profit reference (premium × 100) for the right anchor. */
  maxProfit?: number | null;
  height?: number;
}

export function DistributionBar({
  cvar5,
  p25,
  p50,
  p75,
  maxProfit,
  height = 22,
}: DistributionBarProps) {
  const vals = [cvar5, p25, p50, p75, maxProfit, 0].filter(
    (v): v is number => typeof v === "number" && isFinite(v)
  );
  if (vals.length < 2) {
    return (
      <div className="text-[10px] text-terminal-dim italic">no distribution</div>
    );
  }

  const rawLo = Math.min(...vals);
  const rawHi = Math.max(...vals);
  const span = rawHi - rawLo || 1;
  const pad = span * 0.06;
  const lo = rawLo - pad;
  const hi = rawHi + pad;
  const denom = hi - lo || 1;
  const pos = (x: number) => ((x - lo) / denom) * 100;
  const clamp = (p: number) => Math.max(0, Math.min(100, p));

  const zeroPct = clamp(pos(0));
  const hasBox =
    typeof p25 === "number" && typeof p75 === "number" && isFinite(p25) && isFinite(p75);
  const boxLeft = hasBox ? clamp(pos(Math.min(p25!, p75!))) : 0;
  const boxRight = hasBox ? clamp(pos(Math.max(p25!, p75!))) : 0;
  const boxWidth = Math.max(hasBox ? 1.5 : 0, boxRight - boxLeft);
  const medPct =
    typeof p50 === "number" && isFinite(p50) ? clamp(pos(p50)) : null;
  const cvarPct =
    typeof cvar5 === "number" && isFinite(cvar5) ? clamp(pos(cvar5)) : null;
  // The whisker runs from the tail (cvar5) to the left edge of the box.
  const whiskerRight = hasBox ? boxLeft : zeroPct;
  // Positive cvar5 = the worst-5% mean still profits: neutral whisker, no
  // "crash tail" wording. Loss-red is reserved for an actual modeled loss.
  const tailProfit = typeof cvar5 === "number" && isFinite(cvar5) && cvar5 >= 0;

  const title =
    `5% tail (CVaR, worst-5% mean): ${fmtSigned(cvar5)}  ·  ` +
    `P25 ${fmt(p25)} · P50 ${fmt(p50)} · P75 ${fmt(p75)}  ·  ` +
    `breakeven at 0. ` +
    (tailProfit
      ? "Worst-5% mean is still a PROFIT — no modeled loss tail at this horizon."
      : "Body = likely outcome; red whisker = modeled crash tail.");

  return (
    <div
      className="relative w-full rounded-sm"
      style={{ height }}
      title={title}
      role="img"
      aria-label={title}
    >
      {/* loss/profit zone tints split at breakeven */}
      <div
        className="absolute inset-y-0 left-0 bg-red-500/5"
        style={{ width: `${zeroPct}%` }}
      />
      <div
        className="absolute inset-y-0 bg-green-500/5"
        style={{ left: `${zeroPct}%`, right: 0 }}
      />

      {/* tail whisker: cvar5 -> box left edge (neutral when the tail profits) */}
      {cvarPct !== null && whiskerRight > cvarPct && (
        <div
          className={`absolute top-1/2 h-[2px] -translate-y-1/2 ${
            tailProfit ? "bg-terminal-text/30" : "bg-terminal-red/70"
          }`}
          style={{ left: `${cvarPct}%`, width: `${whiskerRight - cvarPct}%` }}
        />
      )}
      {/* cvar5 endpoint cap */}
      {cvarPct !== null && (
        <div
          className={`absolute top-1/2 h-2.5 w-[2px] -translate-y-1/2 ${
            tailProfit ? "bg-terminal-text/60" : "bg-terminal-red"
          }`}
          style={{ left: `${cvarPct}%` }}
        />
      )}

      {/* interquartile box (p25..p75) */}
      {hasBox && (
        <div
          className="absolute top-1/2 -translate-y-1/2 rounded-[1px] border border-terminal-green/70 bg-terminal-green/30"
          style={{ left: `${boxLeft}%`, width: `${boxWidth}%`, height: height - 8 }}
        />
      )}
      {/* median tick */}
      {medPct !== null && (
        <div
          className="absolute top-1/2 w-[2px] -translate-y-1/2 bg-terminal-green"
          style={{ left: `${medPct}%`, height: height - 6 }}
        />
      )}

      {/* breakeven (zero) line — brightest reference */}
      <div
        className="absolute inset-y-0 w-[1px] bg-terminal-text/80"
        style={{ left: `${zeroPct}%` }}
      />
    </div>
  );
}

function fmt(v: number | null): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  const sign = v < 0 ? "-" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

/** Explicit "+" for the tail value: a positive cvar5 must read unmistakably
 *  as a profit in the tooltip, not as an unsigned "loss" amount. */
function fmtSigned(v: number | null): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  return `${v < 0 ? "-" : "+"}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}
