// ─── Cockpit trust calibration ───────────────────────────────────────────
//
// The single place that encodes WHAT TO TRUST AND WHAT TO DISTRUST in the
// engine's output. The verification campaign (docs/PROB_PROFIT_CALIBRATION_
// 2026-05-28.md, F4_TAIL_RISK_DIAGNOSTIC.md, heavy-verify I1/I11) established:
//
//   * prob_profit is well-calibrated in the MID range (~0.60-0.85). Trust it.
//   * The TOP BIN (> 0.90) is materially OVER-confident, especially in
//     elevated vol: crisis-realized ~0.57 vs ~0.96 forecast. Distrust it; it
//     is exactly the regime R11 (VIX > 25) sizes down.
//   * ev_dollars is a RANKING SCORE with ~0 correlation to realized dollars —
//     never present it as "you will make $X". Show the DISTRIBUTION instead.
//   * The strategy is a defensive premium sleeve, not a market-beater.
//
// These thresholds mirror engine/candidate_dossier.py R11 so the UI caution
// boundary matches the engine's own size-down boundary.

export const R11_VIX_THRESHOLD = 25.0;
export const R11_TOP_BIN_PROB = 0.9;
/** Mirrors engine/candidate_dossier.py MIN_PROCEED_EV_DOLLARS (the R5 floor). */
export const MIN_PROCEED_EV_FALLBACK = 10.0;
/** Empirically realized hit-rate of the crisis top bin (heavy-verify I1). */
export const TOP_BIN_REALIZED = 0.57;

export type ConfidenceTrust = "trust" | "soft-caution" | "hard-caution";

/**
 * Calibration trust for a prob_profit reading, given the market VIX level.
 *  - "trust"        — mid-range (0.60-0.90), the engine's honest zone.
 *  - "soft-caution" — top bin (> 0.90) in calm vol: over-confident but the
 *                     regime that breaks calibration is not present.
 *  - "hard-caution" — top bin (> 0.90) AND elevated vol (VIX > 25): the exact
 *                     R11 regime; draw it loud, surface the realized gap.
 */
export function confidenceTrust(
  probProfit: number | null | undefined,
  vix: number | null | undefined
): ConfidenceTrust {
  const pp = typeof probProfit === "number" ? probProfit : 0;
  const v = typeof vix === "number" ? vix : 0;
  if (pp > R11_TOP_BIN_PROB) {
    return v > R11_VIX_THRESHOLD ? "hard-caution" : "soft-caution";
  }
  return "trust";
}

/** Is R11 currently active given the market VIX level? */
export function r11Active(vix: number | null | undefined): boolean {
  return typeof vix === "number" && vix > R11_VIX_THRESHOLD;
}

/** A short caption surfacing the realized-vs-forecast gap for the top bin. */
export function calibrationNote(
  probProfit: number,
  trust: ConfidenceTrust
): string | null {
  if (trust === "trust") return null;
  const pct = Math.round(probProfit * 100);
  const realized = Math.round(TOP_BIN_REALIZED * 100);
  if (trust === "hard-caution") {
    return `engine ${probProfit.toFixed(2)} / crisis-realized ~${TOP_BIN_REALIZED.toFixed(2)} — top bin over-confident in elevated vol`;
  }
  return `engine ${pct}% is top-bin; crisis-realized ~${realized}% — treat as optimistic`;
}

// ─── Verdict / recommendation styling ─────────────────────────────────────

export type BadgeVariant = "green" | "amber" | "red" | "blue" | "default";

export function verdictVariant(
  v: string | null | undefined
): BadgeVariant {
  switch ((v || "").toLowerCase()) {
    case "proceed":
      return "green";
    case "review":
      return "amber";
    case "skip":
      return "red";
    case "blocked":
      return "red";
    default:
      return "default";
  }
}

// ─── VIX regime labelling ─────────────────────────────────────────────────

export function vixRegimeLabel(vix: number | null | undefined): {
  label: string;
  variant: BadgeVariant;
} {
  if (typeof vix !== "number" || !isFinite(vix))
    return { label: "UNKNOWN", variant: "default" };
  if (vix > 35) return { label: "CRISIS VOL", variant: "red" };
  if (vix > R11_VIX_THRESHOLD) return { label: "ELEVATED VOL", variant: "amber" };
  if (vix > 17) return { label: "NORMAL VOL", variant: "blue" };
  return { label: "CALM VOL", variant: "green" };
}

// ─── Formatters ───────────────────────────────────────────────────────────

export function fmtUsd(
  v: number | null | undefined,
  opts: { signed?: boolean; decimals?: number } = {}
): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  const { signed = false, decimals = 0 } = opts;
  const sign = signed && v >= 0 ? "+" : "";
  return `${sign}$${v.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

export function fmtPct(
  v: number | null | undefined,
  decimals = 0,
  scale = 100
): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  return `${(v * scale).toFixed(decimals)}%`;
}

export function num(v: unknown): number | null {
  const n = typeof v === "number" ? v : Number(v);
  return isFinite(n) ? n : null;
}
