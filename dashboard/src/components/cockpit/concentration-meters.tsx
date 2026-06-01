"use client";

// Concentration meters — single-name exposure vs the 10% R10 cap (now armed
// in production, D17). Answers "if I act on these candidates, do I
// over-concentrate one name?" using the REAL candidate strikes (collateral =
// strike × 100 per contract) against an adjustable book NAV.
//
// HONESTY NOTE on R9 (25% sector cap): per-sector aggregation needs the
// engine's DEFAULT_SECTOR_MAP, which `/api/candidates` does not emit. Rather
// than guess sectors client-side, the live R9 (and R10) VERDICT is available
// from `/api/engine?action=dossier&nav=...&holdings=...` (verdict_reason
// "sector_cap_breach" / "single_name_breach"). The continuous sector bars are
// a documented follow-up; we visualize the single-name cap, which the
// candidate data fully supports.

import { useMemo, useState } from "react";

import type { EngineCandidate } from "@/types/cockpit";
import { fmtUsd } from "@/lib/cockpit-trust";

const R10_CAP = 0.1; // 10% single-name notional cap

interface ConcentrationMetersProps {
  candidates: EngineCandidate[];
  /** Only "proceed"/"review" rows form a hypothetical 1-contract book. */
  defaultNav?: number;
}

export function ConcentrationMeters({
  candidates,
  defaultNav = 250000,
}: ConcentrationMetersProps) {
  const [nav, setNav] = useState(defaultNav);
  const [contracts, setContracts] = useState(1);

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

  return (
    <div className="border border-terminal-border bg-terminal-panel p-2">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
          Single-name concentration (R10 · 10% cap)
        </span>
        <div className="flex items-center gap-2 text-[10px] text-terminal-dim">
          <label className="flex items-center gap-1">
            NAV $
            <input
              type="number"
              value={nav}
              onChange={(e) => setNav(Math.max(1, Number(e.target.value) || 0))}
              className="w-[90px] border border-terminal-border bg-terminal-bg px-1 py-0.5 text-[10px] text-terminal-text"
            />
          </label>
          <label className="flex items-center gap-1">
            contracts
            <input
              type="number"
              value={contracts}
              min={1}
              onChange={(e) => setContracts(Math.max(1, Number(e.target.value) || 1))}
              className="w-[48px] border border-terminal-border bg-terminal-bg px-1 py-0.5 text-[10px] text-terminal-text"
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

      <div className="mt-1.5 text-[9px] leading-tight text-terminal-dim/70">
        R9 sector cap (25%) needs the engine&apos;s sector map (not on
        <code> /api/candidates</code>); the live R9/R10 verdict comes from
        <code> /api/engine?action=dossier&amp;nav=…&amp;holdings=…</code>. Per-sector
        bars are a follow-up — we visualize what the candidate data supports.
      </div>
    </div>
  );
}
