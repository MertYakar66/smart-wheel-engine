"use client";

// Expandable dossier panel — "why this verdict" one click away.
//
// Shows the VerdictCard plus a plain-language reviewer-chain trace (R1-R11):
// which downgrade conditions are met for THIS candidate given the EV row +
// the market VIX. The trace is derived from the same thresholds the engine's
// EnginePhaseReviewer uses (imported in cockpit-trust), so the boundary
// matches the engine's. Rules that need a chart (R2/R3/R4) or a supplied book
// (R7/R8/R9/R10) are shown as "needs chart" / "needs book" rather than
// guessed — the engine, not the cockpit, is the authority for those.

import type { EngineCandidate } from "@/types/cockpit";
import {
  MIN_PROCEED_EV_FALLBACK,
  R11_TOP_BIN_PROB,
  R11_VIX_THRESHOLD,
  fmtUsd,
} from "@/lib/cockpit-trust";
import { VerdictCard } from "./verdict-card";

interface DossierDrawerProps {
  candidate: EngineCandidate | null;
  vix: number | null;
  onClose: () => void;
}

type RuleState = "fires" | "clear" | "needs-chart" | "needs-book";

interface RuleRow {
  id: string;
  name: string;
  state: RuleState;
  detail: string;
}

function buildRuleTrace(c: EngineCandidate, vix: number | null): RuleRow[] {
  const ev = c.evDollars;
  const pp = c.probProfit;
  const topBin = pp > R11_TOP_BIN_PROB;
  const elevated = typeof vix === "number" && vix > R11_VIX_THRESHOLD;
  const earningsIn =
    typeof c.daysToEarnings === "number" &&
    c.daysToEarnings >= 0 &&
    c.daysToEarnings <= c.dte;

  return [
    {
      id: "R1",
      name: "Negative EV → blocked",
      state: ev < 0 ? "fires" : "clear",
      detail:
        ev < 0
          ? `EV ${fmtUsd(ev)} < 0 — hard stop.`
          : `EV ${fmtUsd(ev)} ≥ 0.`,
    },
    {
      id: "R2",
      name: "Chart missing → review",
      state: "needs-chart",
      detail: "Engine downgrades to review if no corroborating chart. Sandbox here has none.",
    },
    {
      id: "R5",
      name: "EV below proceed floor → review",
      state: ev > 0 && ev <= MIN_PROCEED_EV_FALLBACK ? "fires" : "clear",
      detail:
        ev > 0 && ev <= MIN_PROCEED_EV_FALLBACK
          ? `EV ${fmtUsd(ev)} ≤ $${MIN_PROCEED_EV_FALLBACK} proceed floor.`
          : `EV ${fmtUsd(ev)} above the $${MIN_PROCEED_EV_FALLBACK} proceed floor.`,
    },
    {
      id: "R6",
      name: "Dealer regime near gamma-flip / put wall",
      state: c.dealerRegime ? "clear" : "needs-book",
      detail: c.dealerRegime
        ? `Dealer regime: ${c.dealerRegime}.`
        : "No dealer positioning attached to this row.",
    },
    {
      id: "R9/R10",
      name: "Sector 25% / single-name 10% cap",
      state: "needs-book",
      detail: "Concentration caps fire only against a supplied book (nav + holdings).",
    },
    {
      id: "R11",
      name: "Elevated-vol top bin → review",
      state: topBin && elevated ? "fires" : "clear",
      detail:
        topBin && elevated
          ? `prob_profit ${pp.toFixed(2)} > ${R11_TOP_BIN_PROB} AND VIX ${vix!.toFixed(1)} > ${R11_VIX_THRESHOLD} — size down (crisis top bin realizes ~0.57).`
          : topBin
            ? `Top bin (pp ${pp.toFixed(2)}) but VIX ${typeof vix === "number" ? vix.toFixed(1) : "—"} ≤ ${R11_VIX_THRESHOLD} — R11 dormant.`
            : `prob_profit ${pp.toFixed(2)} ≤ ${R11_TOP_BIN_PROB} — not top bin.`,
    },
    {
      id: "EVT",
      name: "Earnings inside the hold",
      state: earningsIn ? "fires" : "clear",
      detail: earningsIn
        ? `Earnings in ${c.daysToEarnings}d, inside the ${c.dte}d hold — event risk.`
        : "No earnings inside the hold window.",
    },
  ];
}

const STATE_STYLE: Record<RuleState, { dot: string; label: string; text: string }> = {
  fires: { dot: "bg-terminal-red", label: "FIRES", text: "text-terminal-red" },
  clear: { dot: "bg-terminal-green/60", label: "clear", text: "text-terminal-dim" },
  "needs-chart": { dot: "bg-terminal-amber/70", label: "needs chart", text: "text-terminal-amber" },
  "needs-book": { dot: "bg-terminal-blue/70", label: "needs book", text: "text-terminal-blue" },
};

export function DossierDrawer({ candidate: c, vix, onClose }: DossierDrawerProps) {
  if (!c) return null;
  const trace = buildRuleTrace(c, vix);
  const fired = trace.filter((r) => r.state === "fires");
  const topReasons = fired.map((r) => `${r.id}: ${r.detail}`);

  return (
    <div className="flex h-full flex-col overflow-hidden border-l border-terminal-border bg-terminal-bg">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-terminal-border bg-terminal-header px-2">
        <span className="text-[11px] font-bold uppercase tracking-wider text-terminal-amber">
          Dossier · {c.ticker}
        </span>
        <button
          onClick={onClose}
          className="text-[12px] text-terminal-dim hover:text-terminal-text"
          aria-label="Close dossier"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 font-mono">
        <VerdictCard
          candidate={c}
          vix={vix}
          verdict={c.recommendation}
          topReasons={topReasons}
        />

        <div className="mt-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
            Reviewer chain (R1–R11)
          </div>
          <div className="flex flex-col gap-1">
            {trace.map((r) => {
              const st = STATE_STYLE[r.state];
              return (
                <div
                  key={r.id}
                  className="flex items-start gap-2 border-b border-terminal-border/30 pb-1"
                >
                  <span
                    className={`mt-1 h-2 w-2 shrink-0 rounded-full ${st.dot}`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] font-semibold text-terminal-text">
                        <span className="text-terminal-dim">{r.id}</span> {r.name}
                      </span>
                      <span className={`text-[9px] uppercase ${st.text}`}>{st.label}</span>
                    </div>
                    <div className="text-[10px] leading-tight text-terminal-dim">
                      {r.detail}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 text-[9px] leading-tight text-terminal-dim/70">
            Trace derived from this candidate&apos;s EV row + market VIX using the
            engine&apos;s own R11 thresholds. R2/R3/R4 (chart) and R7–R10 (book)
            resolve only inside the full <code>/api/tv/dossier</code> when a chart
            and portfolio context are supplied — the engine, not the cockpit, is
            the authority for those.
          </div>
        </div>

        <div className="mt-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
            EV diagnostics (rank inputs)
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
            <Kv k="ev_dollars (rank)" v={fmtUsd(c.evDollars)} dim />
            <Kv k="ev_per_day" v={fmtUsd(c.evPerDay)} dim />
            <Kv k="prob_assignment" v={`${Math.round(c.probAssignment * 100)}%`} />
            <Kv k="cvar_5" v={fmtUsd(c.cvar5)} danger />
            <Kv k="distribution" v={c.distributionSource || "—"} />
            <Kv k="heavy_tail" v={c.heavyTail ? "yes" : "no"} danger={c.heavyTail} />
            <Kv k="edge_vs_fair" v={c.edgeVsFair != null ? fmtUsd(c.edgeVsFair) : "—"} />
            <Kv
              k="breakeven_move"
              v={c.breakevenMovePct != null ? `${(c.breakevenMovePct * 100).toFixed(1)}%` : "—"}
            />
          </div>
          <div className="mt-1 text-[9px] leading-tight text-terminal-dim/70">
            ev_dollars ranks candidates; it does not forecast your dollars (~0
            correlation with realized P&amp;L). Read the distribution and cvar_5.
          </div>
        </div>
      </div>
    </div>
  );
}

function Kv({
  k,
  v,
  dim,
  danger,
}: {
  k: string;
  v: string;
  dim?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-terminal-dim">{k}</span>
      <span
        className={`tabular-nums ${
          danger ? "text-terminal-red" : dim ? "text-terminal-dim/80" : "text-terminal-text"
        }`}
      >
        {v}
      </span>
    </div>
  );
}
