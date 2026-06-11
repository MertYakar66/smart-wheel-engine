"use client";

// Expandable dossier panel — "why this verdict" one click away.
//
// PRIMARY TRUTH: the engine dossier (/api/tv/dossier — EnginePhaseReviewer
// R1-R11 verdict + verdict_reason + review_notes, fetched lazily by the page
// and joined by ticker). When the live book is attached (nav/holdings/
// puts_held from /api/portfolio) the engine resolves R7-R10 against the REAL
// portfolio. The client-side rule trace below is explicitly demoted to a
// "client estimate (offline fallback)" — it mirrors the engine thresholds
// but is NOT the authority and renders only as scaffolding while the engine
// dossier loads or when the fetch fails.

import { useEffect, useRef, useState } from "react";

import type { Dossier, EngineCandidate } from "@/types/cockpit";
import {
  MIN_PROCEED_EV_FALLBACK,
  R11_TOP_BIN_PROB,
  R11_VIX_THRESHOLD,
  fmtN,
  fmtProbCi,
  fmtUsd,
  fmtUsdSigned,
  samplingCiHonest,
} from "@/lib/cockpit-trust";
import { VerdictCard, prettyReason } from "./verdict-card";

export type DossierFetchState = "idle" | "loading" | "ready" | "error";

interface DossierDrawerProps {
  candidate: EngineCandidate | null;
  vix: number | null;
  /** Engine dossier for this ticker (joined by the page), null when absent. */
  dossier: Dossier | null;
  dossierState: DossierFetchState;
  /** Was a live nav/holdings book sent with the dossier fetch? */
  bookAttached: boolean;
  /** e.g. "NAV $152,381 · 3 holdings · 2 short puts (live)". */
  bookLabel?: string | null;
  /** as_of the shown rank was loaded with (for the copy-out ticket). */
  asOf?: string;
  engineVersion?: string | null;
  onRetryDossier?: () => void;
  onClose: () => void;
}

type RuleState = "fires" | "clear" | "needs-chart" | "needs-chain" | "needs-book";

interface RuleRow {
  id: string;
  name: string;
  state: RuleState;
  detail: string;
}

function buildRuleTrace(
  c: EngineCandidate,
  vix: number | null,
  bookAttached: boolean
): RuleRow[] {
  const ev = c.evDollars;
  const pp = c.probProfit;
  const topBin = pp > R11_TOP_BIN_PROB;
  const elevated = typeof vix === "number" && vix > R11_VIX_THRESHOLD;
  const earningsIn =
    typeof c.daysToEarnings === "number" &&
    c.daysToEarnings >= 0 &&
    c.daysToEarnings <= c.dte;
  // R6 mirror — the actual engine condition (candidate_dossier.py): fires on
  // short-gamma regime + strike at/above the put wall, OR on near_flip. The
  // old trace marked ANY non-null regime "clear" — a latent inversion.
  const r6WallBreach =
    c.dealerRegime === "short_gamma_amplifying" &&
    typeof c.nearestPutWallStrike === "number" &&
    isFinite(c.nearestPutWallStrike) &&
    c.strike >= c.nearestPutWallStrike;
  const r6NearFlip = c.dealerRegime === "near_flip";
  // R5 mirror — engine: ev >= $10 → proceed. The old trace used `<=` and
  // inverted the exact-$10 boundary.
  const r5Fires = ev >= 0 && ev < MIN_PROCEED_EV_FALLBACK;

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
      state: r5Fires ? "fires" : "clear",
      detail: r5Fires
        ? `EV ${fmtUsd(ev)} < $${MIN_PROCEED_EV_FALLBACK} proceed floor.`
        : ev < 0
          ? "n/a — R1 already blocks before R5 is reached."
          : `EV ${fmtUsd(ev)} ≥ the $${MIN_PROCEED_EV_FALLBACK} proceed floor.`,
    },
    {
      id: "REC",
      name: "prob_profit ≥ 0.65 (API-label leg, not an R-rule)",
      state: pp >= 0.65 ? "clear" : "fires",
      detail:
        pp >= 0.65
          ? `prob_profit ${pp.toFixed(2)} ≥ 0.65 — the label's second leg holds.`
          : `prob_profit ${pp.toFixed(2)} < 0.65 — the API "proceed" label needs BOTH EV ≥ $${MIN_PROCEED_EV_FALLBACK} AND pp ≥ 0.65, so the row labels review.`,
    },
    {
      id: "R6",
      name: "Dealer regime near gamma-flip / put wall",
      state: c.dealerRegime
        ? r6WallBreach || r6NearFlip
          ? "fires"
          : "clear"
        : "needs-chain",
      detail: c.dealerRegime
        ? r6WallBreach
          ? `Short-gamma regime + strike $${c.strike.toFixed(2)} at/above put wall $${c.nearestPutWallStrike!.toFixed(2)} — downgrade to review.`
          : r6NearFlip
            ? "Dealer regime near gamma flip — downgrade to review."
            : `Dealer regime: ${c.dealerRegime} — neither trigger condition met.`
        : "No dealer positioning on this row (needs an option chain, not a book).",
    },
    {
      id: "R9/R10",
      name: "Sector 25% / single-name 10% cap",
      state: "needs-book",
      detail: bookAttached
        ? "Live book attached — resolved by the engine dossier above (the client cannot evaluate sector maps)."
        : "Concentration caps fire only against a supplied book (nav + holdings).",
    },
    {
      id: "R11",
      name: "Elevated-vol top bin → review",
      state: topBin && elevated ? "fires" : "clear",
      detail:
        topBin && elevated
          ? `prob_profit ${pp.toFixed(2)} > ${R11_TOP_BIN_PROB} AND VIX ${vix!.toFixed(1)} > ${R11_VIX_THRESHOLD} — size down (crisis top bin realizes ~0.57).`
          : topBin
            ? `Top bin (pp ${pp.toFixed(2)}) but VIX ${typeof vix === "number" ? vix.toFixed(1) : "unavailable — R11 state unknown"} ${typeof vix === "number" ? `≤ ${R11_VIX_THRESHOLD} — R11 dormant.` : ""}`
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
  "needs-chain": { dot: "bg-terminal-amber/50", label: "needs chain", text: "text-terminal-amber/80" },
  "needs-book": { dot: "bg-terminal-blue/70", label: "needs book", text: "text-terminal-blue" },
};

/** Plain-text journal ticket — analysis artifact only, NO order routing. */
function buildTradePlan(
  c: EngineCandidate,
  dossier: Dossier | null,
  asOf: string | undefined,
  engineVersion: string | null | undefined,
  vix: number | null
): string {
  const ciOk = samplingCiHonest(c.distributionSource);
  const ci = ciOk ? fmtProbCi(c.probProfitCiLow, c.probProfitCiHigh) : "";
  const n = ciOk ? fmtN(c.nScenarios) : null;
  const verdictLine = dossier
    ? `${dossier.verdict} — ${dossier.verdict_reason} (engine reviewer R1-R11)`
    : `${c.recommendation} (API label: EV ≥ $10 + pp ≥ 0.65 — not the reviewer verdict)`;
  const lines = [
    "SHORT PUT — ANALYSIS TICKET (no order)",
    `ticker       ${c.ticker}`,
    `strike       $${c.strike.toFixed(2)}`,
    `~expiry      ${c.expiration || "—"} (modeled: as_of + ${c.dte} DTE — not a listed expiration)`,
    `premium      $${c.premium.toFixed(2)}/sh (${fmtUsd(c.premium * 100)}/contract)`,
    `spot         $${c.spot.toFixed(2)}${asOf ? ` (as_of ${asOf})` : ""}`,
    `EV (rank)    ${fmtUsdSigned(c.evDollars)} — ranking score, not a P&L forecast`,
    `prob_profit  ${c.probProfit.toFixed(2)}${ci ? ` [Wilson 95% CI ${ci}${n ? `, ${n}` : ""}]` : ""}`,
    `CVaR5        ${fmtUsdSigned(c.cvar5)} (mean of the worst-5% scenarios)`,
    `verdict      ${verdictLine}`,
    `VIX          ${typeof vix === "number" ? vix.toFixed(2) : "unavailable — R11 state unknown"}`,
    `as_of        ${asOf || "engine latest"} · engine ${engineVersion || "unknown"}`,
    `generated    ${new Date().toISOString()}`,
    "-- analysis only — no order. EVEngine.evaluate is the authority. --",
  ];
  return lines.join("\n");
}

export function DossierDrawer({
  candidate: c,
  vix,
  dossier,
  dossierState,
  bookAttached,
  bookLabel,
  asOf,
  engineVersion,
  onRetryDossier,
  onClose,
}: DossierDrawerProps) {
  const [copied, setCopied] = useState<"idle" | "ok" | "err">("idle");
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (copyTimer.current) clearTimeout(copyTimer.current);
    },
    []
  );

  if (!c) return null;
  const trace = buildRuleTrace(c, vix, bookAttached && dossierState === "ready");
  const fired = trace.filter((r) => r.state === "fires");
  const clientReasons = fired.map((r) => `${r.id}: ${r.detail}`);
  const engineLoaded = dossierState === "ready" && dossier != null;

  const copyPlan = async () => {
    try {
      await navigator.clipboard.writeText(
        buildTradePlan(c, engineLoaded ? dossier : null, asOf, engineVersion, vix)
      );
      setCopied("ok");
    } catch {
      setCopied("err");
    }
    if (copyTimer.current) clearTimeout(copyTimer.current);
    copyTimer.current = setTimeout(() => setCopied("idle"), 1800);
  };

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
          verdict={engineLoaded ? dossier!.verdict : c.recommendation}
          verdictReason={engineLoaded ? dossier!.verdict_reason : undefined}
          verdictSource={engineLoaded ? "engine" : "api_label"}
          topReasons={engineLoaded ? dossier!.review_notes : clientReasons}
        />

        <div className="mt-2 flex items-center gap-2">
          <button
            onClick={copyPlan}
            className="rounded-md border border-pf-accent/50 bg-pf-accent/15 px-2 py-1 text-[10px] font-semibold uppercase text-pf-accent hover:bg-pf-accent/25"
            title="Copy a plain-text journal ticket for this candidate. Analysis only — no order is created anywhere."
          >
            {copied === "ok" ? "copied ✓" : copied === "err" ? "copy failed" : "copy trade plan"}
          </button>
          <span className="text-[9px] text-terminal-dim/70">
            analysis ticket only — no order
          </span>
        </div>

        {/* Engine reviewer — the authority */}
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-amber">
              Engine reviewer (authoritative)
            </span>
            {engineLoaded && (
              <span
                className={`text-[9px] ${bookAttached ? "text-terminal-green" : "text-terminal-dim"}`}
                title={
                  bookAttached
                    ? "nav + holdings + short puts from /api/portfolio were attached — R7-R10 evaluated against the real book."
                    : "No portfolio context attached — R7-R10 skip silently (missing-evidence semantics)."
                }
              >
                {bookAttached ? "live book attached" : "no book attached"}
              </span>
            )}
          </div>

          {dossierState === "loading" && (
            <div className="animate-pulse border border-terminal-border/60 bg-terminal-panel px-2 py-2 text-[10px] text-terminal-dim">
              fetching engine dossier (runs the ranker — ~10s at this scan size)…
            </div>
          )}

          {dossierState === "error" && (
            <div className="border border-terminal-amber/40 bg-terminal-amber/10 px-2 py-2 text-[10px] text-terminal-amber">
              engine dossier unavailable — the client estimate below is the only
              trace shown (it is NOT the engine verdict).
              {onRetryDossier && (
                <button
                  onClick={onRetryDossier}
                  className="ml-2 underline underline-offset-2 hover:text-terminal-text"
                >
                  retry
                </button>
              )}
            </div>
          )}

          {dossierState === "ready" && !dossier && (
            <div className="border border-terminal-border/60 bg-terminal-panel px-2 py-2 text-[10px] text-terminal-dim">
              no engine dossier for {c.ticker} in the current batch — the
              dossier ranker returned a different top-N set for these params.
            </div>
          )}

          {engineLoaded && (
            <div className="border border-terminal-border/60 bg-terminal-panel px-2 py-2">
              <div className="text-[10px] text-terminal-amber">
                {prettyReason(dossier!.verdict_reason)}
              </div>
              {bookAttached && bookLabel && (
                <div className="mt-1 text-[9px] text-terminal-dim">
                  book: {bookLabel}
                </div>
              )}
              {dossier!.review_notes.length > 0 && (
                <ul className="mt-1.5 space-y-0.5">
                  {dossier!.review_notes.map((note, i) => (
                    <li
                      key={i}
                      className="text-[10px] leading-tight text-terminal-dim"
                    >
                      • {note}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* Client estimate — demoted, explicitly NOT the authority */}
        <div className="mt-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
            Client estimate (offline fallback)
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
            Client-side mirror of the engine thresholds — an ESTIMATE, not the
            verdict. The &quot;Engine reviewer&quot; section above is the authority
            (full R1–R11 chain from <code>/api/tv/dossier</code>). Until it loads,
            the badge falls back to the API recommendation label (EV floor +
            pp ≥ 0.65), which applies none of the R-rule downgrades.
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
            <Kv k="cvar_5" v={fmtUsdSigned(c.cvar5)} danger={typeof c.cvar5 === "number" && c.cvar5 < 0} />
            <Kv k="distribution" v={c.distributionSource || "—"} />
            <Kv k="heavy_tail" v={c.heavyTail ? "yes" : "no"} danger={c.heavyTail} />
            <Kv k="edge_vs_fair" v={c.edgeVsFair != null ? fmtUsd(c.edgeVsFair) : "—"} />
            <Kv
              k="breakeven_move"
              v={c.breakevenMovePct != null ? `${(c.breakevenMovePct * 100).toFixed(1)}%` : "—"}
            />
            <Kv
              k="target_delta"
              v={typeof c.targetDelta === "number" ? c.targetDelta.toFixed(2) : "—"}
              dim
            />
            <Kv k="~expiry (modeled)" v={c.expiration ? `~${c.expiration}` : "—"} dim />
          </div>
          <div className="mt-1 text-[9px] leading-tight text-terminal-dim/70">
            ev_dollars ranks candidates; it does not forecast your dollars (~0
            correlation with realized P&amp;L). Read the distribution and cvar_5
            (worst-5% MEAN — positive = even the tail profits). target_delta is
            the strike-selection target, not a measured Greek; ~expiry is
            modeled (as_of + DTE), not a listed contract.
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
