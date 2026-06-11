"use client";

// Realized-income analytics — the wheel feedback loop: which names actually
// earn the premium and which bleed. Fed by GET /api/portfolio/income
// (ledger-derived; wheel_tracker reuse engine-side). Realized history only —
// shown distinctly from any forward EV score (finding I1). Display-only.

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtUsd } from "@/lib/cockpit-trust";
import { type IncomeView } from "./mock";
import { PfCard, ProvenanceBadge, fmtSignedUsd, pnlColor, type SliceSource } from "./parts";

const GAIN = "#34d399";
const LOSS = "#f2495e";
const tipStyle = {
  background: "#0b0f14",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 11,
  color: "#e2e8f0",
};

function Chip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-white/[0.08] bg-pf-bg px-2 py-1">
      <span className="text-[10px] uppercase tracking-wide text-terminal-dim">{label}</span>
      <span className={`text-[11px] font-semibold tabular-nums ${color ?? "text-terminal-text"}`}>
        {value}
      </span>
    </span>
  );
}

const TOP_N = 8;

// Props are required — no mock defaults. The only caller (portfolio/page.tsx)
// always passes live-or-mock data; optional props with mock defaults are a
// fabrication hazard: an omitted prop renders MOCK data with no provenance
// badge (idx 20).
export function IncomePanel({
  income,
  source,
}: {
  income: IncomeView;
  source?: SliceSource;
}) {
  const [showAll, setShowAll] = useState(false);
  const names = showAll ? income.byName : income.byName.slice(0, TOP_N);
  const maxAbs = Math.max(1, ...income.byName.map((n) => Math.abs(n.pnl)));

  return (
    <PfCard
      pad={false}
      title="Income · realized"
      right={
        <span className="flex items-center gap-1.5 text-[10px] text-terminal-dim">
          <ProvenanceBadge source={source} />
          closed trades, from the ledger
        </span>
      }
    >
      <div className="px-4 pb-4 pt-2">
        <div className="flex flex-wrap gap-2">
          <Chip
            label="Total realized"
            value={fmtSignedUsd(income.totalRealized)}
            color={pnlColor(income.totalRealized)}
          />
          <Chip
            label="Realized YTD"
            value={fmtSignedUsd(income.realizedYtd)}
            color={pnlColor(income.realizedYtd)}
          />
          <Chip label="Premium 30d" value={fmtUsd(income.premium30d)} color="text-pf-accent" />
          <Chip label="Win rate" value={`${Math.round(income.winRate * 100)}%`} />
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Monthly realized P&L bars */}
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-terminal-dim">
              Realized P&L by month
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={income.byMonth} margin={{ top: 6, right: 8, left: 4, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis
                  dataKey="m"
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  minTickGap={12}
                />
                <YAxis
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={46}
                  tickFormatter={(v) => `$${Math.round(Number(v) / 1000)}k`}
                />
                <Tooltip
                  contentStyle={tipStyle}
                  formatter={(v) => [fmtSignedUsd(Number(v)), "Realized"]}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                />
                <Bar dataKey="pnl" radius={[3, 3, 0, 0]} maxBarSize={22}>
                  {income.byMonth.map((m) => (
                    <Cell key={m.m} fill={m.pnl >= 0 ? GAIN : LOSS} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-ticker league table */}
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wide text-terminal-dim">
                Realized P&L by name
              </span>
              {income.byName.length > TOP_N && (
                <button
                  onClick={() => setShowAll((s) => !s)}
                  className="text-[10px] text-terminal-dim transition-colors hover:text-terminal-text"
                >
                  {showAll ? "top 8" : `show all ${income.byName.length}`}
                </button>
              )}
            </div>
            <div className="space-y-1">
              {names.map((n) => (
                <div key={n.sym} className="flex items-center gap-2">
                  <span className="w-12 shrink-0 font-mono text-[11px] font-medium text-terminal-text">
                    {n.sym}
                  </span>
                  <div className="relative h-1.5 flex-1 rounded-full bg-white/[0.07]">
                    <div
                      className="absolute top-0 h-full rounded-full"
                      style={{
                        width: `${(Math.abs(n.pnl) / maxAbs) * 100}%`,
                        background: n.pnl >= 0 ? GAIN : LOSS,
                        opacity: 0.75,
                      }}
                    />
                  </div>
                  <span
                    className={`w-20 shrink-0 text-right font-mono text-[11px] tabular-nums ${pnlColor(n.pnl)}`}
                  >
                    {fmtSignedUsd(n.pnl)}
                  </span>
                </div>
              ))}
              {income.byName.length === 0 && (
                <p className="text-[11px] text-terminal-dim">No closed trades in the ledger.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </PfCard>
  );
}
