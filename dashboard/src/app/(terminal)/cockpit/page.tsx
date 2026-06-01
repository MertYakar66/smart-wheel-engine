"use client";

// Decision cockpit — read top-to-bottom and act. NOT a browse portal.
// Regime banner → selection funnel → candidate cockpit table (P&L
// distribution + calibration-aware confidence) → one-click dossier drawer.
// Every number comes from the engine via /api/engine (no engine logic here).

import { useCallback, useEffect, useMemo, useState } from "react";

import { CockpitTable } from "@/components/cockpit/cockpit-table";
import { ConcentrationMeters } from "@/components/cockpit/concentration-meters";
import { DossierDrawer } from "@/components/cockpit/dossier-drawer";
import { Funnel } from "@/components/cockpit/funnel";
import { RegimeBanner } from "@/components/cockpit/regime-banner";
import type {
  CandidatesResponse,
  EngineCandidate,
  VixRegime,
} from "@/types/cockpit";

const DEFAULTS = {
  asOf: "2026-03-20", // freshest PIT data
  dte: "35",
  delta: "0.25",
  universeLimit: "120", // full 503-name scan is slow; cap for interactivity
  limit: "15",
};

export default function CockpitPage() {
  const [params, setParams] = useState(DEFAULTS);
  const [data, setData] = useState<CandidatesResponse | null>(null);
  const [vix, setVix] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<EngineCandidate | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({
        action: "candidates",
        as_of: params.asOf,
        dte: params.dte,
        delta: params.delta,
        universe_limit: params.universeLimit,
        limit: params.limit,
        min_ev: "0",
      });
      const [cRes, vRes] = await Promise.all([
        fetch(`/api/engine?${qs.toString()}`, { cache: "no-store" }),
        fetch(`/api/engine?action=vix`, { cache: "no-store" }),
      ]);
      const cJson: CandidatesResponse = await cRes.json();
      if (!cRes.ok || cJson.error) {
        throw new Error(cJson.detail || cJson.error || `HTTP ${cRes.status}`);
      }
      setData(cJson);
      try {
        const vJson: VixRegime = await vRes.json();
        setVix(typeof vJson.vix === "number" ? vJson.vix : null);
      } catch {
        setVix(null);
      }
      // keep selection if still present
      setSelected((prev) =>
        prev ? cJson.trades.find((t) => t.ticker === prev.ticker) || null : null
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load candidates");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const candidates = useMemo(() => data?.trades ?? [], [data]);

  return (
    <div className="min-h-screen bg-terminal-bg p-3 font-mono text-terminal-text">
      {/* Title + controls */}
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-[15px] font-bold uppercase tracking-wider text-terminal-amber">
            Decision Cockpit
          </h1>
          <p className="text-[10px] text-terminal-dim">
            short-put EV ranking · read the distribution, not the point estimate
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <Field label="as_of">
            <input
              type="date"
              value={params.asOf}
              onChange={(e) => setParams((p) => ({ ...p, asOf: e.target.value }))}
              className="w-[120px] border border-terminal-border bg-terminal-panel px-1 py-0.5 text-[11px] text-terminal-text"
            />
          </Field>
          <Field label="DTE">
            <NumInput
              value={params.dte}
              onChange={(v) => setParams((p) => ({ ...p, dte: v }))}
            />
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
            onClick={load}
            disabled={loading}
            className="border border-terminal-amber/60 bg-terminal-amber/10 px-3 py-1 text-[11px] font-semibold uppercase text-terminal-amber hover:bg-terminal-amber/20 disabled:opacity-50"
          >
            {loading ? "ranking…" : "rank"}
          </button>
        </div>
      </div>

      <RegimeBanner
        vix={vix}
        asOf={params.asOf}
        universeScanned={data?.universe_scanned}
        universeTotal={data?.universe_total}
        candidateCount={data?.count}
      />

      {error && (
        <div className="mt-2 border border-terminal-red/60 bg-terminal-red/10 px-3 py-2 text-[11px] text-terminal-red">
          Engine error: {error}
          <div className="mt-0.5 text-[10px] text-terminal-dim">
            Start the API: <code>python engine_api.py</code> (port 8787). Full
            503-name scans are slow — lower “scan”.
          </div>
        </div>
      )}

      <div className="mt-2 flex gap-2">
        {/* Main column */}
        <div className="min-w-0 flex-1">
          <div className="mb-2">
            <Funnel
              universeTotal={data?.universe_total}
              universeScanned={data?.universe_scanned}
              ranked={data?.count}
              shown={candidates.length}
            />
          </div>
          <div className="border border-terminal-border bg-terminal-panel">
            {loading && !candidates.length ? (
              <div className="py-10 text-center text-[12px] text-terminal-dim">
                Ranking {params.universeLimit} names as of {params.asOf}…
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
            <div className="mt-2">
              <ConcentrationMeters candidates={candidates} />
            </div>
          )}
          <TrustLegend />
        </div>

        {/* Drawer column */}
        {selected && (
          <div className="hidden w-[360px] shrink-0 lg:block" style={{ minHeight: 480 }}>
            <DossierDrawer
              candidate={selected}
              vix={vix}
              onClose={() => setSelected(null)}
            />
          </div>
        )}
      </div>
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
      className="w-[64px] border border-terminal-border bg-terminal-panel px-1 py-0.5 text-[11px] text-terminal-text"
    />
  );
}

function TrustLegend() {
  return (
    <div className="mt-2 border border-terminal-border bg-terminal-panel p-2 text-[10px] leading-snug text-terminal-dim">
      <span className="font-bold uppercase tracking-wider text-terminal-blue">
        How to read this
      </span>
      <ul className="mt-1 space-y-0.5">
        <li>
          <span className="text-terminal-text">P&amp;L distribution</span> — green box is
          the likely outcome (keep premium); the long red whisker is the modeled crash
          tail (CVaR5). The risk is in the tail, not the body.
        </li>
        <li>
          <span className="text-terminal-text">Confidence</span> — trust mid-range
          prob_profit (0.60–0.85, green). The top bin (&gt;0.90) is over-confident
          (amber); in elevated vol (red) the crisis-realized rate is ~0.57, not ~0.96.
        </li>
        <li>
          <span className="text-terminal-text">EV·rank</span> — a ranking score only
          (~0 correlation with realized dollars). Never “you will make $X”.
        </li>
      </ul>
    </div>
  );
}
