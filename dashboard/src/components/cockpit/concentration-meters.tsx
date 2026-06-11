"use client";

// Concentration panel — two layers, clearly labelled:
//
//   1. CLIENT ESTIMATE (instant): single-name exposure vs the 10% R10 cap,
//      computed from the candidate strikes against the book NAV. R9 (sector)
//      is impossible client-side (needs the engine's sector map) — never
//      guessed.
//   2. ENGINE GATES (on demand): /api/engine?action=concentration — the
//      operator surface where the ARMED R9/R10 production caps actually fire
//      (engine PR #351). Sequential consume in EV-rank order with structured
//      refusal reasons — the same gates the tracker enforces at
//      open_short_put. It re-runs the ranker server-side (~10s), so it runs
//      only on the explicit "check caps (engine)" button, never on a poll.
//
// NAV truth: prefilled from the live /api/portfolio/summary netLiq (labelled
// with its provenance); the manual input stays as a labelled override. The
// old hardcoded $250k default mis-sized the caps ~1.6x against the real book.

import { useEffect, useMemo, useRef, useState } from "react";

import type {
  ConcentrationPreview,
  ConcentrationRefusal,
  EngineCandidate,
  PortfolioSummaryLite,
} from "@/types/cockpit";
import { fmtUsd } from "@/lib/cockpit-trust";

const R10_CAP = 0.1; // 10% single-name notional cap

export interface ConcentrationRankParams {
  asOf: string;
  dte: string;
  delta: string;
  universeLimit: string;
  limit: string;
}

interface ConcentrationMetersProps {
  candidates: EngineCandidate[];
  /** The params the shown rank was loaded with — the engine check re-ranks
   *  with the SAME scope so its batch matches the table. */
  rankParams: ConcentrationRankParams;
  /** Only "proceed"/"review" rows form a hypothetical 1-contract book. */
  defaultNav?: number;
}

type NavSource = "live" | "snapshot" | "manual" | "default";

function refusalText(r: ConcentrationRefusal): string {
  if (r.reason === "single_name_breach" && typeof r.postOpenNamePct === "number") {
    return `R10 single-name ${(r.postOpenNamePct * 100).toFixed(1)}% > ${((r.nameLimitPct ?? R10_CAP) * 100).toFixed(0)}% NAV`;
  }
  if (r.reason === "sector_cap_breach" && typeof r.postOpenSectorPct === "number") {
    return `R9 ${r.sector ?? "sector"} ${(r.postOpenSectorPct * 100).toFixed(1)}% > ${((r.sectorLimit ?? 0.25) * 100).toFixed(0)}% NAV`;
  }
  return r.reason.replace(/_/g, " ");
}

export function ConcentrationMeters({
  candidates,
  rankParams,
  defaultNav = 250000,
}: ConcentrationMetersProps) {
  const [nav, setNav] = useState(defaultNav);
  const [navSource, setNavSource] = useState<NavSource>("default");
  const [navProvenance, setNavProvenance] = useState<string | null>(null);
  const [contracts, setContracts] = useState(1);

  // Engine gate check — on demand only (re-ranks server-side).
  const [gate, setGate] = useState<ConcentrationPreview | null>(null);
  const [gateState, setGateState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [gateError, setGateError] = useState<string | null>(null);
  const gateAbort = useRef<AbortController | null>(null);
  const gateSeq = useRef(0);

  // Prefill NAV from the live book (labelled by provenance); keep the manual
  // input as an override. Failure falls back to the editable default.
  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const res = await fetch("/api/portfolio/summary", {
          cache: "no-store",
          signal: ctrl.signal,
        });
        if (!res.ok) return;
        const s: PortfolioSummaryLite = await res.json();
        if (typeof s.netLiq === "number" && isFinite(s.netLiq) && s.netLiq > 0) {
          setNav(Math.round(s.netLiq));
          setNavSource(s.source === "live" ? "live" : "snapshot");
          setNavProvenance(s.source ?? null);
        }
      } catch {
        // summary unavailable — keep the labelled default
      }
    })();
    return () => ctrl.abort();
  }, []);

  useEffect(() => () => gateAbort.current?.abort(), []);

  const rows = useMemo(() => {
    const acting = candidates.filter((c) => c.recommendation !== "skip");
    return acting
      .map((c) => {
        const notional = c.strike * 100 * contracts;
        return { ticker: c.ticker, notional, pct: nav > 0 ? notional / nav : 0 };
      })
      .sort((a, b) => b.pct - a.pct);
  }, [candidates, nav, contracts]);

  const overCap = rows.filter((r) => r.pct > R10_CAP);
  const maxPct = rows.length ? Math.max(rows[0].pct, R10_CAP * 1.4) : R10_CAP * 1.4;

  const runEngineCheck = async () => {
    gateAbort.current?.abort();
    const ctrl = new AbortController();
    gateAbort.current = ctrl;
    const seq = ++gateSeq.current;
    setGateState("loading");
    setGateError(null);
    try {
      const qs = new URLSearchParams({
        action: "concentration",
        dte: rankParams.dte,
        delta: rankParams.delta,
        min_ev: "0",
        initial_capital: String(Math.max(1, Math.round(nav))),
        top_n: rankParams.limit,
      });
      if (rankParams.asOf) qs.set("as_of", rankParams.asOf);
      if (rankParams.universeLimit) qs.set("universe_limit", rankParams.universeLimit);
      const res = await fetch(`/api/engine?${qs.toString()}`, {
        cache: "no-store",
        signal: ctrl.signal,
      });
      const json: ConcentrationPreview = await res.json();
      if (seq !== gateSeq.current) return;
      if (!res.ok || json.error) {
        throw new Error(json.detail || json.error || `HTTP ${res.status}`);
      }
      setGate(json);
      setGateState("ready");
    } catch (e) {
      if (ctrl.signal.aborted || seq !== gateSeq.current) return;
      setGateError(e instanceof Error ? e.message : "engine check failed");
      setGateState("error");
    }
  };

  const refusalByTicker = useMemo(() => {
    const m = new Map<string, ConcentrationRefusal>();
    for (const r of gate?.refusals ?? []) m.set(r.ticker, r);
    return m;
  }, [gate]);

  return (
    <div className="rounded-xl border border-white/[0.08] bg-pf-panel p-2">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
          Single-name concentration — client estimate (R10 · 10% cap)
        </span>
        <div className="flex items-center gap-2 text-[10px] text-terminal-dim">
          <label className="flex items-center gap-1">
            NAV $
            <input
              type="number"
              value={nav}
              onChange={(e) => {
                setNav(Math.max(1, Number(e.target.value) || 0));
                setNavSource("manual");
              }}
              className="w-[90px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-1 text-[10px] text-terminal-text"
            />
          </label>
          <span
            className={`rounded-sm px-1 py-0.5 text-[9px] uppercase ${
              navSource === "live"
                ? "bg-terminal-green/15 text-terminal-green"
                : navSource === "snapshot"
                  ? "bg-terminal-amber/15 text-terminal-amber"
                  : "bg-terminal-border/40 text-terminal-dim"
            }`}
            title={
              navSource === "live"
                ? "Prefilled from the live /api/portfolio/summary netLiq."
                : navSource === "snapshot"
                  ? `Prefilled from /api/portfolio/summary (source: ${navProvenance ?? "unknown"} — not live).`
                  : navSource === "manual"
                    ? "Manually overridden."
                    : "Default placeholder — live NAV unavailable."
            }
          >
            {navSource === "live"
              ? "live NAV"
              : navSource === "snapshot"
                ? `${navProvenance ?? "snapshot"} NAV`
                : navSource === "manual"
                  ? "manual"
                  : "default (no live NAV)"}
          </span>
          <label className="flex items-center gap-1">
            contracts
            <input
              type="number"
              value={contracts}
              min={1}
              onChange={(e) => setContracts(Math.max(1, Number(e.target.value) || 1))}
              className="w-[48px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-1 text-[10px] text-terminal-text"
            />
          </label>
        </div>
      </div>

      <div className="mb-1 text-[10px] text-terminal-dim">
        Hypothetical: sell {contracts} put{contracts > 1 ? "s" : ""} per actionable
        candidate. {overCap.length > 0 ? (
          <span className="text-terminal-red">
            {overCap.length} name{overCap.length > 1 ? "s" : ""} would breach the 10% cap.
          </span>
        ) : (
          <span className="text-terminal-green">No name breaches the 10% cap.</span>
        )}
      </div>

      <div className="flex flex-col gap-0.5">
        {rows.slice(0, 12).map((r) => {
          const over = r.pct > R10_CAP;
          const widthPct = Math.min(100, (r.pct / maxPct) * 100);
          const capPct = (R10_CAP / maxPct) * 100;
          return (
            <div key={r.ticker} className="flex items-center gap-2" title={`${fmtUsd(r.notional)} collateral`}>
              <span className="w-12 shrink-0 text-[10px] text-terminal-amber">{r.ticker}</span>
              <div className="relative h-3.5 flex-1 rounded-sm bg-terminal-border/40">
                <div
                  className={`absolute inset-y-0 left-0 rounded-sm ${over ? "bg-terminal-red/60" : "bg-terminal-green/40"}`}
                  style={{ width: `${widthPct}%` }}
                />
                {/* 10% cap line */}
                <div
                  className="absolute inset-y-0 w-[2px] bg-terminal-text/70"
                  style={{ left: `${capPct}%` }}
                  title="10% R10 cap"
                />
                <span className="absolute inset-y-0 right-1 flex items-center text-[9px] tabular-nums text-terminal-text">
                  {(r.pct * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
        {rows.length === 0 && (
          <div className="py-2 text-center text-[10px] text-terminal-dim">
            No actionable candidates to size.
          </div>
        )}
      </div>

      {/* Engine gates — the armed R9/R10 authority, on demand */}
      <div className="mt-2 border-t border-terminal-border/40 pt-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-amber">
            Engine gates (armed R9 sector 25% / R10 single-name 10%)
          </span>
          <button
            onClick={runEngineCheck}
            disabled={gateState === "loading"}
            className="rounded-md border border-pf-accent/50 bg-pf-accent/15 px-2 py-1 text-[10px] font-semibold uppercase text-pf-accent hover:bg-pf-accent/25 disabled:opacity-50"
            title="Runs /api/concentration_preview: re-ranks server-side with the shown params and consumes the top-N in EV-rank order through the ARMED production caps (~10s). On demand only."
          >
            {gateState === "loading" ? "checking caps… (~10s)" : "check caps (engine)"}
          </button>
        </div>

        {gateState === "idle" && (
          <div className="mt-1 text-[9px] leading-tight text-terminal-dim/70">
            The bars above are a client estimate (R10 only — R9 needs the
            engine&apos;s sector map). The engine check runs the SAME armed gates
            the tracker enforces at <code>open_short_put</code>: sequential
            consume in EV-rank order, each admit changing the book the next
            candidate is checked against.
          </div>
        )}

        {gateState === "error" && (
          <div className="mt-1 text-[10px] text-pf-loss">
            engine check failed: {gateError}
          </div>
        )}

        {gateState === "ready" && gate && (
          <div className="mt-1.5">
            <div className="text-[10px] tabular-nums text-terminal-dim">
              consumed {gate.consumed ?? "—"} · opened{" "}
              <span className="text-terminal-green">{gate.opened ?? "—"}</span> · refused{" "}
              <span className={gate.refused ? "text-terminal-red" : "text-terminal-dim"}>
                {gate.refused ?? "—"}
              </span>
              {typeof gate.initial_capital === "number" && (
                <> · capital {fmtUsd(gate.initial_capital)}</>
              )}
              {gate.entry_date && <> · entry {gate.entry_date}</>}
            </div>
            <div className="mt-1 flex flex-col gap-0.5">
              {(gate.outcomes ?? []).map((o) => {
                const refusal = refusalByTicker.get(o.ticker);
                return (
                  <div
                    key={o.ticker}
                    className="flex items-center justify-between gap-2 border-b border-terminal-border/30 py-0.5 text-[10px]"
                  >
                    <span className="flex items-center gap-2">
                      <span
                        className={`w-14 shrink-0 rounded-sm px-1 text-center text-[9px] font-semibold uppercase ${
                          o.opened
                            ? "bg-terminal-green/15 text-terminal-green"
                            : "bg-terminal-red/15 text-terminal-red"
                        }`}
                      >
                        {o.opened ? "admit" : "refused"}
                      </span>
                      <span className="font-bold text-terminal-amber">{o.ticker}</span>
                    </span>
                    <span className="min-w-0 truncate text-right text-terminal-dim">
                      {o.opened
                        ? `EV ${fmtUsd(o.evDollars)}`
                        : refusal
                          ? refusalText(refusal)
                          : (o.refusalReason ?? "refused").replace(/_/g, " ")}
                    </span>
                  </div>
                );
              })}
              {(gate.outcomes ?? []).length === 0 && (
                <div className="py-1 text-center text-[10px] text-terminal-dim">
                  engine consumed no candidates for these params.
                </div>
              )}
            </div>
            <div className="mt-1 text-[9px] leading-tight text-terminal-dim/70">
              Authoritative: same refusal the tracker issues at open. Refuse-only
              gates — they never rescue a candidate (§2). Snapshot of the moment
              you clicked; re-check after changing params or NAV.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
