// Shared frontier-staleness chip — used by regime-banner.tsx and
// cockpit/page.tsx header. Single source of truth so threshold/copy changes
// propagate to both render sites without two-site synchronization.
//
// Severity tiers (behind > 0 direction):
//   0          — "engine frontier" (dim)
//   1-30d      — amber
//   > 30d      — red
// beyond frontier (behind < 0) — amber + explicit label

interface FrontierChipProps {
  frontier: string | null;
  asOf: string;
  /** Days the loaded as_of lags the frontier (negative = beyond it). */
  behindFrontier: number | null;
  /** vs-today age — fallback when the frontier is unknown. */
  staleDays?: number | null;
  /** Size variant: "sm" (header) | "base" (banner). */
  size?: "sm" | "base";
}

export function FrontierChip({
  frontier,
  asOf,
  behindFrontier,
  staleDays,
  size = "base",
}: FrontierChipProps) {
  const cls = size === "sm" ? "text-xs" : "text-[10px]";

  if (frontier && asOf && behindFrontier !== null) {
    if (behindFrontier === 0) {
      return (
        <span
          className={`${cls} text-terminal-dim`}
          title={`as_of equals the engine's data frontier (${frontier}) — the freshest bar in its data files, not necessarily today.`}
        >
          (engine frontier)
        </span>
      );
    }
    if (behindFrontier > 0) {
      const color = behindFrontier > 30 ? "text-terminal-red" : "text-pf-caution";
      return (
        <span
          className={`${cls} tabular-nums ${color}`}
          title={`The engine has data through ${frontier}; this view ranks point-in-time as of ${asOf}.`}
        >
          ({behindFrontier}d behind frontier)
        </span>
      );
    }
    // behindFrontier < 0 — beyond frontier
    return (
      <span
        className={`${cls} text-pf-caution`}
        title={`as_of is past the engine's data frontier (${frontier}) — no data exists beyond it.`}
      >
        (beyond frontier — no data past {frontier})
      </span>
    );
  }

  // Frontier unknown — fall back to vs-today age
  if (!frontier && typeof staleDays === "number" && staleDays > 0) {
    return (
      <span
        className={`${cls} tabular-nums text-terminal-dim`}
        title="Engine frontier unknown (status unavailable); age shown vs today instead."
      >
        ({staleDays}d old vs today)
      </span>
    );
  }

  return null;
}
