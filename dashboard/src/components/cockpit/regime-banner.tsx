// Regime banner — today's "weather" at the top of the cockpit.
// One glance frames every verdict below: VIX level + term structure +
// percentile + regime label, whether R11 (elevated-vol top-bin haircut) is
// active, and the as_of date measured against the ENGINE'S DATA FRONTIER
// (not the wall clock). This is a premium sleeve, not a market-beater.
//
// Honesty rules:
//   * A FAILED VIX fetch renders an explicit "VIX unavailable — R11 state
//     unknown" amber state — never the calm/dormant default.
//   * The as_of chip compares to the frontier the engine actually reported
//     (/api/status data_frontier); only when the frontier is unknown does it
//     fall back to a vs-today age, labelled as such.

import { TerminalBadge } from "@/components/terminal/panel";
import { r11Active, vixRegimeLabel } from "@/lib/cockpit-trust";
import type { VixRegime } from "@/types/cockpit";

interface RegimeBannerProps {
  vixData: VixRegime | null;
  vixState: "loading" | "ok" | "failed";
  asOf: string;
  /** Engine data frontier (/api/status data_frontier), null when status failed. */
  frontier: string | null;
  /** Days the loaded as_of lags the frontier (negative = beyond it). */
  behindFrontier: number | null;
  /** vs-today age — fallback label when the frontier is unknown. */
  staleDays?: number | null;
  universeScanned?: number;
  universeTotal?: number;
  candidateCount?: number;
}

export function RegimeBanner({
  vixData,
  vixState,
  asOf,
  frontier,
  behindFrontier,
  staleDays,
  universeScanned,
  universeTotal,
  candidateCount,
}: RegimeBannerProps) {
  const vix = typeof vixData?.vix === "number" ? vixData.vix : null;
  const vixFailed = vixState === "failed";
  const { label, variant } = vixFailed
    ? { label: "VIX UNAVAILABLE", variant: "amber" as const }
    : vixState === "loading"
      ? { label: "…", variant: "default" as const }
      : vixRegimeLabel(vix);
  const r11 = r11Active(vix);
  const backwardation = vixData?.term_structure === "backwardation";
  const pctile =
    typeof vixData?.vix_percentile === "number" && isFinite(vixData.vix_percentile)
      ? Math.round(vixData.vix_percentile * 100)
      : null;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-pf-panel px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
            Regime
          </span>
          <TerminalBadge variant={variant}>{label}</TerminalBadge>
        </div>

        <div className="flex items-baseline gap-1">
          <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
            VIX
          </span>
          <span className="text-[15px] font-bold tabular-nums text-terminal-text">
            {vix !== null ? vix.toFixed(2) : "—"}
          </span>
        </div>

        {/* term structure + percentile — core premium-seller context */}
        {vix !== null && (
          <div
            className="flex items-baseline gap-1 text-[11px] tabular-nums"
            title="VIX term structure: spot → 3m → 6m. Contango (upward) is the normal premium-seller regime; backwardation signals near-term stress."
          >
            <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
              term
            </span>
            <span className="text-terminal-text">
              {vix.toFixed(1)}
              {typeof vixData?.vix_3m === "number" && (
                <> → {vixData.vix_3m.toFixed(1)}<span className="text-terminal-dim"> (3m)</span></>
              )}
              {typeof vixData?.vix_6m === "number" && (
                <> → {vixData.vix_6m.toFixed(1)}<span className="text-terminal-dim"> (6m)</span></>
              )}
            </span>
            {vixData?.term_structure && (
              <span
                className={backwardation ? "text-terminal-amber" : "text-terminal-dim"}
              >
                · {vixData.term_structure}
              </span>
            )}
            {pctile !== null && (
              <span className="text-terminal-dim">· {pctile}th %ile</span>
            )}
          </div>
        )}

        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
            R11
          </span>
          {vixFailed ? (
            <TerminalBadge variant="amber">UNKNOWN — VIX UNAVAILABLE</TerminalBadge>
          ) : vixState === "loading" ? (
            <TerminalBadge variant="default">…</TerminalBadge>
          ) : r11 ? (
            <TerminalBadge variant="amber">ACTIVE — SIZE DOWN TOP BIN</TerminalBadge>
          ) : (
            <TerminalBadge variant="default">dormant</TerminalBadge>
          )}
        </div>

        <div className="flex items-baseline gap-1">
          <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
            As of
          </span>
          <span className="text-[12px] font-semibold tabular-nums text-terminal-text">
            {asOf || "latest"}
          </span>
          {frontier && asOf && behindFrontier !== null ? (
            behindFrontier === 0 ? (
              <span
                className="text-[10px] text-terminal-dim"
                title={`as_of equals the engine's data frontier (${frontier}) — the freshest bar in its data files, not necessarily today.`}
              >
                (engine frontier)
              </span>
            ) : behindFrontier > 0 ? (
              <span
                className={`text-[10px] tabular-nums ${
                  behindFrontier > 30 ? "text-terminal-red" : "text-terminal-amber"
                }`}
                title={`The engine has data through ${frontier}; this view ranks point-in-time as of ${asOf}.`}
              >
                ({behindFrontier}d behind frontier)
              </span>
            ) : (
              <span
                className="text-[10px] text-terminal-amber"
                title={`as_of is past the engine's data frontier (${frontier}) — no data exists beyond it.`}
              >
                (beyond frontier — no data past {frontier})
              </span>
            )
          ) : (
            !frontier &&
            typeof staleDays === "number" &&
            staleDays > 0 && (
              <span
                className="text-[10px] tabular-nums text-terminal-dim"
                title="Engine frontier unknown (status unavailable); age shown vs today instead."
              >
                ({staleDays}d old vs today)
              </span>
            )
          )}
        </div>

        {typeof universeTotal === "number" && (
          <div className="flex items-baseline gap-1">
            <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
              Universe
            </span>
            <span className="text-[12px] tabular-nums text-terminal-text">
              {universeScanned ?? "—"}/{universeTotal} scanned
              {typeof candidateCount === "number" ? ` → ${candidateCount} ranked` : ""}
            </span>
          </div>
        )}
      </div>

      <div className="mt-1.5 text-[10px] leading-tight text-terminal-dim">
        {vixFailed
          ? "VIX unavailable — R11 (elevated-vol top-bin size-down) state UNKNOWN. Top-bin confidence cannot be regime-checked; treat high prob_profit rows with caution until the VIX feed returns."
          : r11
            ? "Elevated vol — the engine's top-bin confidence is over-stated in this regime (crisis-realized ~0.57 vs ~0.96 forecast). The cockpit haircuts high-confidence picks; size down."
            : "Defensive premium sleeve. Read the P&L distribution (not the EV point estimate); trust mid-range prob_profit (0.60-0.85), distrust the top bin."}
      </div>
    </div>
  );
}
