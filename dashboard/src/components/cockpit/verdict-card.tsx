// Glanceable verdict card — the quant-card FORMAT (not the Seeking-Alpha
// product): verdict + calibration-flagged confidence + a distribution
// sparkline + the top 1-2 reasons. Used inside the dossier drawer header.

import { TerminalBadge } from "@/components/terminal/panel";
import type { EngineCandidate, Verdict } from "@/types/cockpit";
import {
  calibrationNote,
  confidenceTrust,
  fmtN,
  fmtPct,
  fmtProbCi,
  fmtUsd,
  samplingCiHonest,
  verdictVariant,
} from "@/lib/cockpit-trust";
import { DistributionBar } from "./distribution-bar";

interface VerdictCardProps {
  candidate: EngineCandidate;
  vix: number | null;
  verdict?: Verdict | null;
  verdictReason?: string | null;
  topReasons?: string[];
}

export function VerdictCard({
  candidate: c,
  vix,
  verdict,
  verdictReason,
  topReasons = [],
}: VerdictCardProps) {
  const v = verdict || c.recommendation;
  const trust = confidenceTrust(c.probProfit, vix);
  const note = calibrationNote(c.probProfit, trust);
  // Wilson 95% sampling CI (orthogonal to the calibration trust): widen the
  // headline so it is never read as a precise 2-dp figure. Shown only on the
  // IID non-overlapping tier (samplingCiHonest) — elsewhere the CI is false
  // precision; degrade to the bare %. Mirrors the cockpit-table render.
  const ciOk = samplingCiHonest(c.distributionSource);
  const ppCi = ciOk ? fmtProbCi(c.probProfitCiLow, c.probProfitCiHigh) : "";
  const ppN = ciOk ? fmtN(c.nScenarios) : null;
  const rocAnn =
    c.strike > 0 && c.dte > 0 ? (c.premium / c.strike) * (365 / c.dte) : null;

  return (
    <div className="border border-terminal-border bg-terminal-panel p-3">
      <div className="flex items-center justify-between">
        <span className="text-[16px] font-bold text-terminal-amber">{c.ticker}</span>
        <TerminalBadge variant={verdictVariant(v)}>{v}</TerminalBadge>
      </div>
      <div className="mt-0.5 text-[10px] text-terminal-dim">
        short put · ${c.strike.toFixed(2)} · {c.dte}DTE · {fmtPct(c.iv, 1)} IV · spot $
        {c.spot.toFixed(2)}
      </div>

      <div className="mt-2">
        <div className="mb-0.5 text-[9px] uppercase tracking-wider text-terminal-dim">
          Modeled P&amp;L distribution
        </div>
        <DistributionBar
          cvar5={c.cvar5}
          p25={c.pnlP25}
          p50={c.pnlP50}
          p75={c.pnlP75}
          maxProfit={c.premium * 100}
          height={26}
        />
        <div className="mt-0.5 flex justify-between text-[9px] tabular-nums text-terminal-dim">
          <span className="text-terminal-red">tail {fmtUsd(c.cvar5)}</span>
          <span>breakeven 0</span>
          <span className="text-terminal-green">keep {fmtUsd(c.premium * 100)}</span>
        </div>
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
        <Metric label="Premium" value={fmtUsd(c.premium * 100)} color="text-terminal-green" />
        <Metric label="ROC ann." value={fmtPct(rocAnn, 1)} />
        <Metric
          label={ppN ? `prob_profit · ${ppN}` : "prob_profit"}
          value={`${Math.round(c.probProfit * 100)}%${ppCi ? ` [${ppCi}]` : ""}`}
          color={
            trust === "trust"
              ? "text-terminal-green"
              : trust === "soft-caution"
                ? "text-terminal-amber"
                : "text-terminal-red"
          }
        />
      </div>

      {note && (
        <div className="mt-2 border-l-2 border-terminal-red/60 pl-2 text-[10px] leading-tight text-terminal-red/90">
          {note}
        </div>
      )}

      {(verdictReason || topReasons.length > 0) && (
        <div className="mt-2 text-[10px] leading-snug text-terminal-text">
          <span className="text-[9px] uppercase tracking-wider text-terminal-dim">
            Why this verdict
          </span>
          {verdictReason && (
            <div className="mt-0.5 text-terminal-amber">{prettyReason(verdictReason)}</div>
          )}
          {topReasons.slice(0, 2).map((r, i) => (
            <div key={i} className="mt-0.5 text-terminal-dim">
              • {r}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  color = "text-terminal-text",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-terminal-dim">{label}</span>
      <span className={`tabular-nums font-semibold ${color}`}>{value}</span>
    </div>
  );
}

export function prettyReason(reason: string): string {
  const map: Record<string, string> = {
    elevated_vol_top_bin:
      "R11 — elevated-vol top bin: VIX > 25 and prob_profit > 0.90. Over-confident regime; size down.",
    sector_cap_breach: "R9 — would breach the 25% sector concentration cap.",
    single_name_breach: "R10 — would breach the 10% single-name concentration cap.",
    chart_missing: "R2 — no chart available to corroborate; review.",
    spot_mismatch: "R3 — chart spot disagrees with engine spot > 2%; skip.",
    phase_contradiction: "R4 — chart phase contradicts the trade; skip.",
    negative_ev: "R1 — negative EV; blocked (hard stop).",
    ev_below_threshold: "R5 — EV below the proceed threshold; review.",
    proceed: "Above EV threshold, no downgrade fired.",
  };
  return map[reason] || reason.replace(/_/g, " ");
}
