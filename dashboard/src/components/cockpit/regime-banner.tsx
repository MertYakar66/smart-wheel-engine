// Regime banner — today's "weather" at the top of the cockpit.
// One glance frames every verdict below: VIX level + regime label, whether
// R11 (elevated-vol top-bin haircut) is active, the as_of date, and a
// defensive-posture note. This is a premium sleeve, not a market-beater.

import { TerminalBadge } from "@/components/terminal/panel";
import { r11Active, vixRegimeLabel } from "@/lib/cockpit-trust";

interface RegimeBannerProps {
  vix: number | null;
  asOf: string;
  /** How many days old the loaded as_of is vs today (computed client-side). */
  staleDays?: number | null;
  universeScanned?: number;
  universeTotal?: number;
  candidateCount?: number;
}

export function RegimeBanner({
  vix,
  asOf,
  staleDays,
  universeScanned,
  universeTotal,
  candidateCount,
}: RegimeBannerProps) {
  const { label, variant } = vixRegimeLabel(vix);
  const r11 = r11Active(vix);

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
            {typeof vix === "number" ? vix.toFixed(2) : "—"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
            R11
          </span>
          {r11 ? (
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
          {typeof staleDays === "number" && staleDays >= 0 && (
            <span
              className={`text-[10px] tabular-nums ${
                staleDays > 30
                  ? "text-terminal-red"
                  : staleDays > 7
                    ? "text-terminal-amber"
                    : "text-terminal-dim"
              }`}
              title="Age of the loaded point-in-time data vs today. This is the freshest data available, not necessarily today."
            >
              ({staleDays}d old)
            </span>
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
        {r11
          ? "Elevated vol — the engine's top-bin confidence is over-stated in this regime (crisis-realized ~0.57 vs ~0.96 forecast). The cockpit haircuts high-confidence picks; size down."
          : "Defensive premium sleeve. Read the P&L distribution (not the EV point estimate); trust mid-range prob_profit (0.60-0.85), distrust the top bin."}
      </div>
    </div>
  );
}
