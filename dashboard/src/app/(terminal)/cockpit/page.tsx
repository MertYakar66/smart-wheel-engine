"use client";

// Decision cockpit — read top-to-bottom and act. NOT a browse portal.
// Regime banner → selection funnel → candidate cockpit table (P&L
// distribution + calibration-aware confidence) → one-click dossier drawer
// backed by the AUTHORITATIVE engine reviewer (/api/tv/dossier, R1-R11,
// live book attached). Every number comes from the engine via /api/engine
// (no engine logic here; display-only, downgrade-only context).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CockpitTable } from "@/components/cockpit/cockpit-table";
import { ConcentrationMeters } from "@/components/cockpit/concentration-meters";
import { DossierDrawer, type DossierFetchState } from "@/components/cockpit/dossier-drawer";
import { FrontierChip } from "@/components/cockpit/frontier-chip";
import { Funnel } from "@/components/cockpit/funnel";
import { RegimeBanner } from "@/components/cockpit/regime-banner";
import { CrossPageNav, WheelhouseHeader } from "@/components/shell/wheelhouse-header";
import { fmtUsd } from "@/lib/cockpit-trust";
import type {
  CandidatesResponse,
  Dossier,
  DossierResponse,
  EngineCandidate,
  EngineStatus,
  PortfolioPositionsLite,
  PortfolioSummaryLite,
  VixRegime,
} from "@/types/cockpit";

const DEFAULTS = {
  // Seeded from the engine's data_frontier (/api/status) on mount; "" falls
  // back to engine-latest. Never hardcode a date here — a stale literal once
  // booted every session 82 days behind the frontier (cockpit F1).
  asOf: "",
  dte: "35",
  delta: "0.25",
  universeLimit: "120", // full 503-name scan is slow; cap for interactivity
  limit: "15",
};

type RankParams = typeof DEFAULTS;

const paramsKey = (p: RankParams) =>
  [p.asOf, p.dte, p.delta, p.universeLimit, p.limit].join("|");

/** Whole days from a to b (ISO dates, UTC-pinned so it is deterministic). */
function daysBetweenIso(a: string, b: string): number | null {
  const ta = new Date(`${a}T00:00:00Z`).getTime();
  const tb = new Date(`${b}T00:00:00Z`).getTime();
  if (!isFinite(ta) || !isFinite(tb)) return null;
  return Math.round((tb - ta) / 86_400_000);
}

const MONTHS: Record<string, string> = {
  JAN: "01", FEB: "02", MAR: "03", APR: "04", MAY: "05", JUN: "06",
  JUL: "07", AUG: "08", SEP: "09", OCT: "10", NOV: "11", DEC: "12",
};

/** "MRVL 10JUL26 297.5 P" → { strike: 297.5, expiry: "2026-07-10" }. IBKR
 *  local-symbol format from /api/portfolio/positions leg names; null when it
 *  doesn't parse — the leg is then omitted from puts_held, never guessed. */
function parseOptionLegName(name: string): { strike: number; expiry: string } | null {
  const m = /\b(\d{1,2})([A-Z]{3})(\d{2})\s+(\d+(?:\.\d+)?)\s+[PC]\b/.exec(
    name.toUpperCase()
  );
  if (!m) return null;
  const [, dd, mon, yy, strikeStr] = m;
  const mm = MONTHS[mon];
  if (!mm) return null;
  const strike = parseFloat(strikeStr);
  if (!isFinite(strike) || strike <= 0) return null;
  return { strike, expiry: `20${yy}-${mm}-${dd.padStart(2, "0")}` };
}

export default function CockpitPage() {
  const [params, setParams] = useState(DEFAULTS);
  const [loadedParams, setLoadedParams] = useState<RankParams | null>(null);
  const [frontier, setFrontier] = useState<string | null>(null);
  const [data, setData] = useState<CandidatesResponse | null>(null);
  const [vixData, setVixData] = useState<VixRegime | null>(null);
  const [vixState, setVixState] = useState<"loading" | "ok" | "failed">("loading");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<EngineCandidate | null>(null);
  // Computed client-side (in an effect) so it never causes an SSR/client
  // hydration mismatch — vs-today fallback age when the frontier is unknown.
  const [staleDays, setStaleDays] = useState<number | null>(null);

  // Engine dossier batch — lazy (first drawer open), cached per rank
  // param-set: one /api/tv/dossier call costs roughly a candidates call.
  const [dossierMap, setDossierMap] = useState<Record<string, Dossier>>({});
  const [dossierKey, setDossierKey] = useState<string | null>(null);
  const [dossierState, setDossierState] = useState<DossierFetchState>("idle");
  const [bookAttached, setBookAttached] = useState(false);
  // positionsUnavailable: summary succeeded but positions endpoint failed.
  // Surfaces explicit "live book unavailable — R7-R10 not engaged" in the drawer.
  const [positionsUnavailable, setPositionsUnavailable] = useState(false);
  const [bookLabel, setBookLabel] = useState<string | null>(null);
  // Pre-fetched summary nav/source — passed down to ConcentrationMeters so the
  // cockpit page does not issue two separate /api/portfolio/summary requests
  // on the same load (one from fetchDossiers, one from the component mount).
  const [summaryNav, setSummaryNav] = useState<number | null>(null);
  const [summaryNavSource, setSummaryNavSource] = useState<string | null>(null);

  const rankAbort = useRef<AbortController | null>(null);
  const rankSeq = useRef(0);
  const dossierAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!params.asOf) {
      setStaleDays(null);
      return;
    }
    const asOfMs = new Date(`${params.asOf}T00:00:00`).getTime();
    const days = Math.floor((Date.now() - asOfMs) / 86_400_000);
    setStaleDays(Number.isFinite(days) ? days : null);
  }, [params.asOf]);

  // Days the picked as_of lags the engine frontier (negative = beyond it).
  const behindFrontier = useMemo(() => {
    if (!frontier || !params.asOf) return null;
    return daysBetweenIso(params.asOf, frontier);
  }, [frontier, params.asOf]);

  const load = useCallback(async (p: RankParams) => {
    // Latest-request-wins (quality F7): abort the in-flight rank + vix pair
    // so a slow stale-params response can never overwrite a newer rank.
    rankAbort.current?.abort();
    const ctrl = new AbortController();
    rankAbort.current = ctrl;
    const seq = ++rankSeq.current;
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({
        action: "candidates",
        dte: p.dte,
        delta: p.delta,
        universe_limit: p.universeLimit,
        limit: p.limit,
        min_ev: "0",
      });
      if (p.asOf) qs.set("as_of", p.asOf);
      const [cRes, vRes] = await Promise.allSettled([
        fetch(`/api/engine?${qs.toString()}`, { cache: "no-store", signal: ctrl.signal }),
        fetch(`/api/engine?action=vix`, { cache: "no-store", signal: ctrl.signal }),
      ]);
      if (seq !== rankSeq.current) return; // superseded by a newer rank

      if (cRes.status !== "fulfilled") {
        throw cRes.reason instanceof Error
          ? cRes.reason
          : new Error("Failed to load candidates");
      }
      const cJson: CandidatesResponse = await cRes.value.json();
      if (!cRes.value.ok || cJson.error) {
        throw new Error(cJson.detail || cJson.error || `HTTP ${cRes.value.status}`);
      }
      if (seq !== rankSeq.current) return;
      setData(cJson);
      setLoadedParams(p);
      // keep selection if still present
      setSelected((prev) =>
        prev ? cJson.trades.find((t) => t.ticker === prev.ticker) || null : null
      );

      // VIX — a failed fetch renders an explicit UNKNOWN state downstream,
      // never the calm/dormant default (cockpit F11).
      if (vRes.status === "fulfilled" && vRes.value.ok) {
        try {
          const vJson: VixRegime = await vRes.value.json();
          if (seq !== rankSeq.current) return;
          if (typeof vJson.vix === "number" && isFinite(vJson.vix)) {
            setVixData(vJson);
            setVixState("ok");
          } else {
            setVixData(null);
            setVixState("failed");
          }
        } catch {
          setVixData(null);
          setVixState("failed");
        }
      } else {
        setVixData(null);
        setVixState("failed");
      }
    } catch (e) {
      if (ctrl.signal.aborted || seq !== rankSeq.current) return;
      setError(e instanceof Error ? e.message : "Failed to load candidates");
      setData(null);
    } finally {
      if (seq === rankSeq.current) setLoading(false);
    }
  }, []);

  // Boot: resolve the engine data frontier first, seed as_of from it, then
  // rank. If status fails, as_of stays "" (engine-latest) — honest fallback.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let asOf = "";
      try {
        const res = await fetch("/api/engine?action=status", { cache: "no-store" });
        const j: EngineStatus = await res.json();
        if (
          res.ok &&
          typeof j.data_frontier === "string" &&
          /^\d{4}-\d{2}-\d{2}$/.test(j.data_frontier)
        ) {
          asOf = j.data_frontier;
          if (!cancelled) setFrontier(j.data_frontier);
        }
      } catch {
        // status unavailable — engine-latest fallback
      }
      if (cancelled) return;
      const p = { ...DEFAULTS, asOf };
      setParams(p);
      load(p);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchDossiers = useCallback(async (p: RankParams, k: string) => {
    dossierAbort.current?.abort();
    const ctrl = new AbortController();
    dossierAbort.current = ctrl;
    setDossierKey(k);
    setDossierState("loading");
    setDossierMap({});
    setBookAttached(false);
    setPositionsUnavailable(false);
    setBookLabel(null);
    try {
      // Live book (best-effort): nav + share holdings + short puts. This is
      // what lets the engine resolve R7-R10 against the REAL portfolio.
      // Failures degrade to a bookless dossier — R7-R10 then skip silently
      // (Q3 missing-evidence semantics), and the drawer says so.
      let nav: number | null = null;
      let holdings = "";
      let putsHeld = "";
      let label: string | null = null;
      // positionsOk tracks whether the positions endpoint succeeded; bookAttached
      // requires BOTH summary AND positions to avoid evaluating R9/R10 against
      // a nav-only zero-position book (half-book verdicts are misleading — idx 3/7).
      let positionsOk = false;
      try {
        const [sRes, pRes] = await Promise.all([
          fetch("/api/portfolio/summary", { cache: "no-store", signal: ctrl.signal }),
          fetch("/api/portfolio/positions", { cache: "no-store", signal: ctrl.signal }),
        ]);
        if (sRes.ok) {
          const s: PortfolioSummaryLite = await sRes.json();
          if (typeof s.netLiq === "number" && isFinite(s.netLiq) && s.netLiq > 0) {
            nav = s.netLiq;
            // Store summary nav for ConcentrationMeters prop-passing (avoids
            // a duplicate /api/portfolio/summary request on the same cockpit load).
            setSummaryNav(Math.round(s.netLiq));
            setSummaryNavSource(s.source ?? null);
            let nHoldings = 0;
            let nPuts = 0;
            if (pRes.ok) {
              positionsOk = true;
              const pos: PortfolioPositionsLite = await pRes.json();
              const hl: string[] = [];
              const ph: string[] = [];
              for (const leg of pos.legs ?? []) {
                if (!leg?.sym) continue;
                if (leg.state === "shares" && typeof leg.qty === "number" && leg.qty !== 0) {
                  hl.push(`${leg.sym}:${Math.trunc(leg.qty)}`);
                } else if (leg.state === "short_put") {
                  // Prefer server-computed strike/expiry fields (added in D26 build_positions_flat)
                  // over regex-parsed name strings — more robust for non-standard IBKR formats.
                  // Fall back to regex only when the structured fields are absent.
                  let strike: number | null = null;
                  let expiry: string | null = null;
                  if (typeof leg.strike === "number" && isFinite(leg.strike) && leg.strike > 0
                      && typeof leg.expiry === "string" && leg.expiry) {
                    strike = leg.strike;
                    expiry = leg.expiry;
                  } else {
                    // Labeled fallback: name-parse only for legs missing structured fields.
                    const parsed = parseOptionLegName(leg.name ?? "");
                    if (parsed) { strike = parsed.strike; expiry = parsed.expiry; }
                  }
                  if (strike !== null && expiry !== null) {
                    const contracts = Math.abs(Math.trunc(leg.qty ?? 1)) || 1;
                    ph.push(`${leg.sym}:${strike}:${contracts}:${expiry}`);
                  }
                }
              }
              holdings = hl.join(",");
              putsHeld = ph.join(",");
              nHoldings = hl.length;
              nPuts = ph.length;
            }
            label = positionsOk
              ? `NAV ${fmtUsd(nav)} · ${nHoldings} holding${nHoldings === 1 ? "" : "s"} · ${nPuts} short put${nPuts === 1 ? "" : "s"} (${s.source ?? "unknown source"})`
              : `NAV ${fmtUsd(nav)} (${s.source ?? "unknown source"}) — positions unavailable`;
          }
        }
      } catch {
        // book unavailable — bookless dossier
      }

      const qs = new URLSearchParams({
        action: "dossier",
        top_n: p.limit,
        dte: p.dte,
        delta: p.delta,
        min_ev: "0",
      });
      if (p.asOf) qs.set("as_of", p.asOf);
      if (p.universeLimit) qs.set("universe_limit", p.universeLimit);
      if (nav != null) qs.set("nav", String(Math.round(nav)));
      if (holdings) qs.set("holdings", holdings);
      if (putsHeld) qs.set("puts_held", putsHeld);
      const res = await fetch(`/api/engine?${qs.toString()}`, {
        cache: "no-store",
        signal: ctrl.signal,
      });
      const json: DossierResponse = await res.json();
      if (ctrl.signal.aborted) return;
      if (!res.ok || json.error) {
        throw new Error(json.detail || json.error || `HTTP ${res.status}`);
      }
      const map: Record<string, Dossier> = {};
      for (const d of json.dossiers ?? []) {
        if (d?.ticker) map[d.ticker] = d;
      }
      setDossierMap(map);
      // bookAttached = true ONLY when summary AND positions both succeeded;
      // nav-alone (positions failed) degrades to bookless — R7-R10 then skip
      // silently on absent evidence rather than evaluating against an empty book.
      setBookAttached(nav != null && positionsOk);
      // Surface an explicit warning in the drawer when summary succeeded but
      // positions specifically failed (distinguishable from "no portfolio at all").
      setPositionsUnavailable(nav != null && !positionsOk);
      setBookLabel(label);
      setDossierState("ready");
    } catch {
      if (ctrl.signal.aborted) return;
      setDossierState("error");
    }
  }, []);

  // Lazy trigger: first drawer open per rank fetches the dossier batch. An
  // errored batch stays errored (manual retry in the drawer) — no auto loop.
  useEffect(() => {
    if (!selected || !loadedParams) return;
    const k = paramsKey(loadedParams);
    if (dossierKey === k && dossierState !== "idle") return;
    fetchDossiers(loadedParams, k);
  }, [selected, loadedParams, dossierKey, dossierState, fetchDossiers]);

  const candidates = useMemo(() => data?.trades ?? [], [data]);
  const vix = vixData?.vix ?? null;
  const currentKey = loadedParams ? paramsKey(loadedParams) : null;
  const dossierFresh = dossierKey !== null && dossierKey === currentKey;
  const selectedDossier =
    selected && dossierFresh ? dossierMap[selected.ticker] ?? null : null;
  const effectiveDossierState: DossierFetchState = dossierFresh
    ? dossierState
    : selected
      ? "loading"
      : "idle";
  const retryDossier = useCallback(() => {
    if (loadedParams) fetchDossiers(loadedParams, paramsKey(loadedParams));
  }, [loadedParams, fetchDossiers]);

  const drawerProps = {
    candidate: selected,
    vix,
    dossier: selectedDossier,
    dossierState: effectiveDossierState,
    bookAttached: bookAttached && dossierFresh,
    positionsUnavailable: positionsUnavailable && dossierFresh,
    bookLabel: dossierFresh ? bookLabel : null,
    asOf: loadedParams?.asOf ?? params.asOf,
    engineVersion: data?.engine_version ?? null,
    onRetryDossier: retryDossier,
  };

  return (
    <div className="min-h-screen bg-pf-bg font-mono text-terminal-text">
      <WheelhouseHeader
        page="Cockpit"
        right={
          <>
            <span className="text-[11px] uppercase tracking-wider text-terminal-dim">as of</span>
            <span className="text-sm font-semibold tabular-nums text-terminal-text">
              {params.asOf || "latest"}
            </span>
            <FrontierChip
              frontier={frontier}
              asOf={params.asOf}
              behindFrontier={behindFrontier}
              staleDays={staleDays}
              size="sm"
            />
          </>
        }
        status={
          data?.count != null ? (
            <span className="tabular-nums">
              {data.count} returned · {candidates.length} shown
            </span>
          ) : undefined
        }
      >
        <CrossPageNav active="Cockpit" />
      </WheelhouseHeader>

      <main className="mx-auto max-w-[1400px] space-y-3 px-5 py-4">
        {/* Controls toolbar */}
        <div className="rounded-xl border border-white/[0.08] bg-pf-panel p-3">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div className="font-sans">
              <h1 className="text-sm font-semibold tracking-tight text-terminal-text">
                Decision Cockpit
              </h1>
              <p className="text-[11px] text-terminal-dim">
                short-put EV ranking · read the distribution, not the point estimate
              </p>
            </div>
            <div className="flex flex-wrap items-end gap-2">
              <Field label="as_of">
                <div className="flex items-center gap-1">
                  <input
                    type="date"
                    value={params.asOf}
                    onChange={(e) => setParams((p) => ({ ...p, asOf: e.target.value }))}
                    className="w-[120px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-1 text-[11px] text-terminal-text"
                  />
                  {frontier && params.asOf !== frontier && (
                    <button
                      onClick={() => setParams((p) => ({ ...p, asOf: frontier }))}
                      title={`Set as_of to the engine data frontier (${frontier}) — the freshest bar available.`}
                      className="rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-1 text-[10px] text-terminal-amber hover:bg-terminal-border/40"
                    >
                      frontier
                    </button>
                  )}
                </div>
              </Field>
              <Field label="DTE">
                <NumInput value={params.dte} onChange={(v) => setParams((p) => ({ ...p, dte: v }))} />
              </Field>
              <Field label="delta">
                <NumInput
                  value={params.delta}
                  step="0.05"
                  onChange={(v) => setParams((p) => ({ ...p, delta: v }))}
                />
              </Field>
              <Field label="scan">
                <NumInput
                  value={params.universeLimit}
                  onChange={(v) => setParams((p) => ({ ...p, universeLimit: v }))}
                />
              </Field>
              <Field label="top-N">
                <NumInput
                  value={params.limit}
                  onChange={(v) => setParams((p) => ({ ...p, limit: v }))}
                />
              </Field>
              <button
                onClick={() => load(params)}
                disabled={loading}
                className="rounded-md border border-pf-accent/50 bg-pf-accent/15 px-3 py-1 text-[11px] font-semibold uppercase text-pf-accent hover:bg-pf-accent/25 disabled:opacity-50"
              >
                {loading ? "ranking…" : "rank"}
              </button>
            </div>
          </div>
        </div>

        <RegimeBanner
          vixData={vixData}
          vixState={vixState}
          asOf={params.asOf}
          frontier={frontier}
          behindFrontier={behindFrontier}
          staleDays={staleDays}
          universeScanned={data?.universe_scanned}
          universeTotal={data?.universe_total}
          candidateCount={data?.count}
        />

        {error && (
          <div className="rounded-xl border border-pf-loss/40 bg-pf-loss/10 px-3 py-2 text-[11px] text-pf-loss">
            Engine error: {error}
            <div className="mt-0.5 text-[10px] text-terminal-dim">
              Start the API: <code>python engine_api.py</code> (port 8787). Full 503-name scans
              are slow — lower “scan”.
            </div>
          </div>
        )}

        <div className="flex gap-3">
          {/* Main column */}
          <div className="min-w-0 flex-1 space-y-3">
            <Funnel
              universeTotal={data?.universe_total}
              universeScanned={data?.universe_scanned}
              ranked={data?.count}
              shown={candidates.length}
              dropsSummary={data?.drops_summary}
            />
            <div className="overflow-hidden rounded-xl border border-white/[0.08] bg-pf-panel">
              {loading && !candidates.length ? (
                <div className="py-10 text-center text-[12px] text-terminal-dim">
                  Ranking {params.universeLimit} names as of {params.asOf || "latest"}…
                </div>
              ) : (
                <CockpitTable
                  candidates={candidates}
                  vix={vix}
                  selectedTicker={selected?.ticker}
                  onSelect={setSelected}
                />
              )}
            </div>
            {candidates.length > 0 && (
              <ConcentrationMeters
                candidates={candidates}
                rankParams={loadedParams ?? params}
                preloadedNav={summaryNav}
                preloadedNavSource={summaryNavSource}
              />
            )}
            <TrustLegend />
          </div>

          {/* Drawer column — desktop side panel (lg+) */}
          {selected && (
            <div className="hidden w-[360px] shrink-0 lg:block" style={{ minHeight: 480 }}>
              <DossierDrawer {...drawerProps} onClose={() => setSelected(null)} />
            </div>
          )}
        </div>
      </main>

      {/* Drawer — mobile/tablet bottom sheet (below lg, where the side column
          is hidden so a row click would otherwise do nothing visible). */}
      {selected && (
        <div
          className="fixed inset-0 z-50 flex flex-col justify-end bg-black/60 lg:hidden"
          role="dialog"
          aria-modal="true"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[85vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <DossierDrawer {...drawerProps} onClose={() => setSelected(null)} />
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[9px] uppercase tracking-wider text-terminal-dim">{label}</span>
      {children}
    </label>
  );
}

function NumInput({
  value,
  onChange,
  step,
}: {
  value: string;
  onChange: (v: string) => void;
  step?: string;
}) {
  return (
    <input
      type="number"
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-[64px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-1 text-[11px] text-terminal-text"
    />
  );
}

function TrustLegend() {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-pf-panel p-3 text-[10px] leading-snug text-terminal-dim">
      <span className="font-bold uppercase tracking-wider text-pf-accent">
        How to read this
      </span>
      <ul className="mt-1 space-y-0.5">
        <li>
          <span className="text-terminal-text">P&amp;L distribution</span> — green box is
          the likely outcome (keep premium); the whisker is CVaR5, the MEAN of the
          worst-5% scenarios: red when a modeled loss, neutral when even the tail
          profits (positive CVaR5 — it happens). The risk is in the tail, not the body.
        </li>
        <li>
          <span className="text-terminal-text">Confidence</span> — trust mid-range
          prob_profit (0.60–0.85, green). The top bin (&gt;0.90) is over-confident
          (amber); in elevated vol (red) the crisis-realized rate is ~0.57, not ~0.96.
          When the VIX is unavailable the R11 state is UNKNOWN, never assumed calm.
        </li>
        <li>
          <span className="text-terminal-text">EV·rank</span> — a ranking score only
          (~0 correlation with realized dollars). Never “you will make $X”.
        </li>
        <li>
          <span className="text-terminal-text">Verdicts</span> — the drawer&apos;s
          engine reviewer (R1–R11, live book attached) is the authority; the table
          badge is the API label (EV floor + pp ≥ 0.65) until the dossier loads.
        </li>
        <li>
          <span className="text-terminal-text">Columns &amp; sort</span> — sorting,
          filters and optional columns are display-only; engine rank is the
          authoritative order. ~Exp is modeled (as_of + DTE), not a listed
          expiration; Cushion = breakeven move % (negative = room below spot);
          Ω = gain/loss probability-mass ratio.
        </li>
      </ul>
    </div>
  );
}
