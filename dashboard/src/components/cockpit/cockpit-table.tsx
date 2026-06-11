"use client";

// The candidate cockpit table — each row is a decision unit, read left to
// right: verdict → P&L distribution (the headline) → calibration-aware
// confidence → the mechanical terms. ev_dollars appears only as a small,
// explicitly-labelled RANKING score, never as a dollar forecast.
//
// Sorting, quick filters and the column chooser are DISPLAY-ONLY: they
// reorder or hide rows the engine already ranked — they never re-rank,
// re-score, or upgrade a verdict (§2). Default order is the engine rank;
// any deviation shows an explicit "engine rank" reset.

import { useEffect, useMemo, useRef, useState } from "react";

import { TerminalBadge } from "@/components/terminal/panel";
import type { EngineCandidate } from "@/types/cockpit";
import {
  fmtPct,
  fmtUsd,
  fmtUsdSigned,
  pnlToneClass,
  verdictVariant,
} from "@/lib/cockpit-trust";
import { CalibratedProb } from "./calibrated-prob";
import { DistributionBar } from "./distribution-bar";

const PREFS_KEY = "swe.cockpit.table.v1";

type SortKey =
  | "prob"
  | "strike"
  | "premium"
  | "dte"
  | "iv"
  | "collat"
  | "roc"
  | "cvar5"
  | "ev"
  | "cushion"
  | "omega"
  | "assign";

interface SortState {
  key: SortKey;
  dir: 1 | -1; // 1 = ascending, -1 = descending
}

interface Filters {
  /** Minimum prob_profit in percent ("60" = 0.60). Empty = off. */
  minPp: string;
  /** Minimum evDollars. Empty = off. */
  minEv: string;
  /** Keep only "proceed" rows. */
  proceedOnly: boolean;
}

interface OptCols {
  cushion: boolean;
  omega: boolean;
  assign: boolean;
  exp: boolean;
}

const DEFAULT_FILTERS: Filters = { minPp: "", minEv: "", proceedOnly: false };
const DEFAULT_COLS: OptCols = { cushion: false, omega: false, assign: false, exp: false };

const num = (v: unknown): number | null =>
  typeof v === "number" && isFinite(v) ? v : null;

const ACCESSORS: Record<SortKey, (c: EngineCandidate) => number | null> = {
  prob: (c) => num(c.probProfit),
  strike: (c) => num(c.strike),
  premium: (c) => num(c.premium),
  dte: (c) => num(c.dte),
  iv: (c) => num(c.iv),
  collat: (c) => num(c.strike * 100),
  roc: (c) =>
    c.strike > 0 && c.dte > 0 ? (c.premium / c.strike) * (365 / c.dte) : null,
  cvar5: (c) => num(c.cvar5),
  ev: (c) => num(c.evDollars),
  cushion: (c) => num(c.breakevenMovePct),
  omega: (c) => num(c.omegaRatio),
  assign: (c) => num(c.probAssignment),
};

interface CockpitTableProps {
  candidates: EngineCandidate[];
  vix: number | null;
  selectedTicker?: string | null;
  onSelect?: (c: EngineCandidate) => void;
}

export function CockpitTable({
  candidates,
  vix,
  selectedTicker,
  onSelect,
}: CockpitTableProps) {
  const [sort, setSort] = useState<SortState | null>(null);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [cols, setCols] = useState<OptCols>(DEFAULT_COLS);
  const [chooserOpen, setChooserOpen] = useState(false);
  const hydrated = useRef(false);

  // Load persisted prefs AFTER hydration (in a frame callback, not the
  // effect body) so SSR markup and the first client render agree — no
  // hydration mismatch — and no synchronous setState-in-effect cascade.
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      try {
        const raw = localStorage.getItem(PREFS_KEY);
        if (raw) {
          const p = JSON.parse(raw);
          if (p && typeof p === "object") {
            if (
              p.sort &&
              typeof p.sort.key === "string" &&
              p.sort.key in ACCESSORS &&
              (p.sort.dir === 1 || p.sort.dir === -1)
            ) {
              setSort({ key: p.sort.key as SortKey, dir: p.sort.dir });
            }
            if (p.filters && typeof p.filters === "object") {
              setFilters({ ...DEFAULT_FILTERS, ...p.filters });
            }
            if (p.cols && typeof p.cols === "object") {
              setCols({ ...DEFAULT_COLS, ...p.cols });
            }
          }
        }
      } catch {
        // corrupted prefs — fall back to defaults
      }
      hydrated.current = true;
    });
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    if (!hydrated.current) return;
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({ sort, filters, cols }));
    } catch {
      // storage unavailable (private mode) — prefs just don't persist
    }
  }, [sort, filters, cols]);

  const rows = useMemo(() => {
    let out = candidates;
    const minPp = parseFloat(filters.minPp);
    if (isFinite(minPp)) out = out.filter((c) => c.probProfit * 100 >= minPp);
    const minEv = parseFloat(filters.minEv);
    if (isFinite(minEv)) out = out.filter((c) => c.evDollars >= minEv);
    if (filters.proceedOnly) out = out.filter((c) => c.recommendation === "proceed");
    if (sort) {
      const acc = ACCESSORS[sort.key];
      out = [...out].sort((a, b) => {
        const va = acc(a);
        const vb = acc(b);
        if (va === null && vb === null) return 0;
        if (va === null) return 1; // nulls last, either direction
        if (vb === null) return -1;
        return (va - vb) * sort.dir;
      });
    }
    return out;
  }, [candidates, filters, sort]);

  const filtersActive =
    filters.minPp !== "" || filters.minEv !== "" || filters.proceedOnly;

  // Cycle: none → desc → asc → none (engine rank).
  const cycleSort = (key: SortKey) =>
    setSort((s) =>
      !s || s.key !== key
        ? { key, dir: -1 }
        : s.dir === -1
          ? { key, dir: 1 }
          : null
    );

  const ariaSort = (key: SortKey): "ascending" | "descending" | "none" =>
    sort?.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none";

  const sortGlyph = (key: SortKey) =>
    sort?.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : "";

  if (!candidates.length) {
    return (
      <div className="py-8 text-center text-[12px] text-terminal-dim">
        No candidates ranked. Check the engine connection and as_of date.
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* display-only toolbar: filters + column chooser */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-terminal-border/60 px-2 py-1.5">
        <div className="flex flex-wrap items-center gap-2 text-[10px] text-terminal-dim">
          <label className="flex items-center gap-1">
            min pp%
            <input
              type="number"
              value={filters.minPp}
              min={0}
              max={100}
              placeholder="—"
              onChange={(e) => setFilters((f) => ({ ...f, minPp: e.target.value }))}
              className="w-[52px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1 py-0.5 text-[10px] text-terminal-text"
            />
          </label>
          <label className="flex items-center gap-1">
            min EV $
            <input
              type="number"
              value={filters.minEv}
              placeholder="—"
              onChange={(e) => setFilters((f) => ({ ...f, minEv: e.target.value }))}
              className="w-[60px] rounded-md border border-white/[0.08] bg-pf-panel2 px-1 py-0.5 text-[10px] text-terminal-text"
            />
          </label>
          <label className="flex cursor-pointer items-center gap-1">
            <input
              type="checkbox"
              checked={filters.proceedOnly}
              onChange={(e) =>
                setFilters((f) => ({ ...f, proceedOnly: e.target.checked }))
              }
            />
            proceed only
          </label>
          {filtersActive && (
            <button
              onClick={() => setFilters(DEFAULT_FILTERS)}
              className="text-[10px] text-pf-caution underline-offset-2 hover:underline"
            >
              clear filters
            </button>
          )}
        </div>
        <div className="flex items-center gap-2 text-[10px]">
          <span className="tabular-nums text-terminal-dim">
            {rows.length}/{candidates.length} rows
            {filtersActive ? " · filtered" : ""}
            {sort ? " · sorted" : ""}
            {(filtersActive || sort) && (
              <span className="text-terminal-dim/70"> (display-only)</span>
            )}
          </span>
          {sort && (
            <button
              onClick={() => setSort(null)}
              title="Restore the engine's EV rank order — the authoritative ordering"
              className="rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-0.5 text-[10px] text-terminal-amber hover:bg-terminal-border/40"
            >
              engine rank ⟲
            </button>
          )}
          <div className="relative">
            <button
              onClick={() => setChooserOpen((o) => !o)}
              aria-expanded={chooserOpen}
              className="rounded-md border border-white/[0.08] bg-pf-panel2 px-1.5 py-0.5 text-[10px] text-terminal-dim hover:text-terminal-text"
            >
              columns ▾
            </button>
            {chooserOpen && (
              <div className="absolute right-0 z-10 mt-1 w-[220px] rounded-md border border-white/[0.08] bg-pf-panel2 p-2 text-[10px] text-terminal-dim shadow-lg">
                <ColToggle
                  label="Cushion (breakeven move %)"
                  checked={cols.cushion}
                  onChange={(v) => setCols((c) => ({ ...c, cushion: v }))}
                />
                <ColToggle
                  label="Omega ratio (gain/loss mass)"
                  checked={cols.omega}
                  onChange={(v) => setCols((c) => ({ ...c, omega: v }))}
                />
                <ColToggle
                  label="Assignment probability"
                  checked={cols.assign}
                  onChange={(v) => setCols((c) => ({ ...c, assign: v }))}
                />
                <ColToggle
                  label="~exp (modeled: as_of + DTE)"
                  checked={cols.exp}
                  onChange={(v) => setCols((c) => ({ ...c, exp: v }))}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="w-full overflow-x-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-terminal-border text-[9px] uppercase tracking-wider text-terminal-dim">
              <th className="px-2 py-1 text-left">Verdict</th>
              <th className="px-2 py-1 text-left">Sym</th>
              <th className="px-2 py-1 text-left" style={{ minWidth: 180 }}>
                P&L distribution{" "}
                <span className="normal-case text-terminal-dim/70">(tail · body · breakeven)</span>
              </th>
              <SortTh
                k="prob"
                align="left"
                minWidth={96}
                ariaSort={ariaSort("prob")}
                onClick={cycleSort}
                title="Sort by prob_profit (display-only)"
              >
                Confidence{sortGlyph("prob")}
              </SortTh>
              <SortTh k="strike" ariaSort={ariaSort("strike")} onClick={cycleSort}>
                Strike{sortGlyph("strike")}
              </SortTh>
              <SortTh k="premium" ariaSort={ariaSort("premium")} onClick={cycleSort}>
                Prem{sortGlyph("premium")}
              </SortTh>
              <SortTh k="dte" ariaSort={ariaSort("dte")} onClick={cycleSort}>
                DTE{sortGlyph("dte")}
              </SortTh>
              {cols.exp && (
                <th
                  className="px-2 py-1 text-right"
                  title="Modeled contract date: as_of + DTE — the synthetic contract the EV math priced. NOT a listed chain expiration."
                >
                  ~Exp
                </th>
              )}
              <SortTh k="iv" ariaSort={ariaSort("iv")} onClick={cycleSort}>
                IV{sortGlyph("iv")}
              </SortTh>
              {cols.cushion && (
                <SortTh
                  k="cushion"
                  ariaSort={ariaSort("cushion")}
                  onClick={cycleSort}
                  title="Breakeven move: how far spot can move before the position loses (negative = cushion below spot). Sort display-only."
                >
                  Cushion{sortGlyph("cushion")}
                </SortTh>
              )}
              {cols.omega && (
                <SortTh
                  k="omega"
                  ariaSort={ariaSort("omega")}
                  onClick={cycleSort}
                  title="Omega ratio: probability-weighted gain mass / loss mass over the modeled distribution. Sort display-only."
                >
                  Ω{sortGlyph("omega")}
                </SortTh>
              )}
              {cols.assign && (
                <SortTh
                  k="assign"
                  ariaSort={ariaSort("assign")}
                  onClick={cycleSort}
                  title="Modeled probability of assignment (finishing ITM). Sort display-only."
                >
                  Assign{sortGlyph("assign")}
                </SortTh>
              )}
              <SortTh
                k="collat"
                ariaSort={ariaSort("collat")}
                onClick={cycleSort}
                title="Cash-secured collateral = strike × 100"
              >
                Collat{sortGlyph("collat")}
              </SortTh>
              <SortTh
                k="roc"
                ariaSort={ariaSort("roc")}
                onClick={cycleSort}
                title="Return on collateral = premium / strike, annualized to 365/DTE"
              >
                ROC a.{sortGlyph("roc")}
              </SortTh>
              <SortTh
                k="cvar5"
                ariaSort={ariaSort("cvar5")}
                onClick={cycleSort}
                title="Mean P&L of the worst-5% scenarios (per contract). Negative = modeled tail loss; POSITIVE = even the tail profits."
              >
                CVaR5{sortGlyph("cvar5")}
              </SortTh>
              <SortTh
                k="ev"
                ariaSort={ariaSort("ev")}
                onClick={cycleSort}
                dim
                title="EV is a RANKING score only — ~0 correlation with realized dollars. Not a forecast."
              >
                EV·rank{sortGlyph("ev")}
              </SortTh>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const collateral = c.strike * 100;
              const rocAnn =
                c.strike > 0 && c.dte > 0
                  ? (c.premium / c.strike) * (365 / c.dte)
                  : null;
              const selected = selectedTicker === c.ticker;
              const earningsSoon =
                typeof c.daysToEarnings === "number" &&
                c.daysToEarnings >= 0 &&
                c.daysToEarnings <= c.dte;
              return (
                <tr
                  key={c.ticker}
                  onClick={() => onSelect?.(c)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect?.(c);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Open dossier for ${c.ticker} — ${c.recommendation}`}
                  className={`cursor-pointer border-b border-terminal-border/40 align-middle hover:bg-terminal-border/30 focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-terminal-amber ${
                    selected ? "bg-terminal-border/40" : ""
                  }`}
                >
                  <td className="px-2 py-1.5">
                    <TerminalBadge variant={verdictVariant(c.recommendation)}>
                      {c.recommendation}
                    </TerminalBadge>
                  </td>
                  <td className="px-2 py-1.5">
                    <div className="flex items-center gap-1">
                      <span className="font-bold text-terminal-amber">{c.ticker}</span>
                      {earningsSoon && (
                        <span
                          className="text-[9px] text-terminal-red"
                          title={`earnings in ${c.daysToEarnings}d — inside the ${c.dte}d hold`}
                        >
                          E
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-2 py-1.5">
                    <DistributionBar
                      cvar5={c.cvar5}
                      p25={c.pnlP25}
                      p50={c.pnlP50}
                      p75={c.pnlP75}
                      maxProfit={c.premium * 100}
                    />
                  </td>
                  <td className="px-2 py-1.5">
                    <CalibratedProb
                      probProfit={c.probProfit}
                      vix={vix}
                      ciLow={c.probProfitCiLow}
                      ciHigh={c.probProfitCiHigh}
                      nScenarios={c.nScenarios}
                      distributionSource={c.distributionSource}
                    />
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                    ${c.strike.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-green">
                    ${c.premium.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim">
                    {c.dte}
                  </td>
                  {cols.exp && (
                    <td
                      className="px-2 py-1.5 text-right tabular-nums text-terminal-dim"
                      title="Modeled: as_of + DTE. Not a listed chain expiration."
                    >
                      {c.expiration ? `~${c.expiration}` : "—"}
                    </td>
                  )}
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                    {fmtPct(c.iv, 1)}
                  </td>
                  {cols.cushion && (
                    <td
                      className={`px-2 py-1.5 text-right tabular-nums ${
                        typeof c.breakevenMovePct === "number" &&
                        isFinite(c.breakevenMovePct)
                          ? c.breakevenMovePct < 0
                            ? "text-terminal-green"
                            : "text-terminal-red"
                          : "text-terminal-dim"
                      }`}
                      title="How far spot can move before loss (negative = cushion below spot)"
                    >
                      {fmtPct(c.breakevenMovePct, 1)}
                    </td>
                  )}
                  {cols.omega && (
                    <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                      {typeof c.omegaRatio === "number" && isFinite(c.omegaRatio)
                        ? c.omegaRatio.toFixed(1)
                        : "—"}
                    </td>
                  )}
                  {cols.assign && (
                    <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                      {fmtPct(c.probAssignment, 0)}
                    </td>
                  )}
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim">
                    {fmtUsd(collateral)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text">
                    {fmtPct(rocAnn, 1)}
                  </td>
                  <td
                    className={`px-2 py-1.5 text-right tabular-nums ${pnlToneClass(c.cvar5)}`}
                  >
                    {fmtUsdSigned(c.cvar5)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-dim/70">
                    {fmtUsd(c.evDollars)}
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={20}
                  className="py-6 text-center text-[11px] text-terminal-dim"
                >
                  0 of {candidates.length} ranked rows match the display filters —{" "}
                  <button
                    onClick={() => setFilters(DEFAULT_FILTERS)}
                    className="text-pf-caution underline-offset-2 hover:underline"
                  >
                    clear filters
                  </button>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortTh({
  k,
  children,
  onClick,
  ariaSort,
  title,
  align = "right",
  minWidth,
  dim,
}: {
  k: SortKey;
  children: React.ReactNode;
  onClick: (k: SortKey) => void;
  ariaSort: "ascending" | "descending" | "none";
  title?: string;
  align?: "left" | "right";
  minWidth?: number;
  dim?: boolean;
}) {
  return (
    <th
      aria-sort={ariaSort}
      className={`px-2 py-1 ${align === "left" ? "text-left" : "text-right"} ${
        dim ? "text-terminal-dim" : ""
      }`}
      style={minWidth ? { minWidth } : undefined}
    >
      <button
        onClick={() => onClick(k)}
        title={title || "Sort (display-only — engine rank is the authority)"}
        className="uppercase tracking-wider hover:text-terminal-text"
      >
        {children}
      </button>
    </th>
  );
}

function ColToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 py-0.5">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
