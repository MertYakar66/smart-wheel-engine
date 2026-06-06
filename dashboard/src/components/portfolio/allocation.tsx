"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import { CURRENCY, SECTORS } from "./mock";
import { PfCard } from "./parts";

const RESOLVED: Record<string, string> = {
  "var(--color-pf-accent)": "#2dd4bf",
  "var(--color-pf-csp)": "#56b6f5",
  "var(--color-pf-assigned)": "#b79cfb",
};

export function Allocation() {
  const top = SECTORS[0];
  return (
    <PfCard title="Allocation" className="h-full">
      <div className="flex items-center gap-4">
        <div className="relative shrink-0" style={{ width: 132, height: 132 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={SECTORS}
                dataKey="val"
                nameKey="name"
                innerRadius={44}
                outerRadius={62}
                paddingAngle={2}
                stroke="none"
                startAngle={90}
                endAngle={-270}
              >
                {SECTORS.map((s) => (
                  <Cell key={s.name} fill={RESOLVED[s.color] ?? s.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-lg font-semibold tabular-nums text-terminal-text">
              {top.val}%
            </span>
            <span className="text-[9px] uppercase tracking-wide text-terminal-dim">
              {top.name}
            </span>
          </div>
        </div>

        <div className="min-w-0 flex-1 space-y-2">
          {SECTORS.map((s) => (
            <div key={s.name} className="flex items-center gap-2 text-xs">
              <span
                className="h-2 w-2 shrink-0 rounded-sm"
                style={{ background: RESOLVED[s.color] ?? s.color }}
              />
              <span className="truncate text-terminal-text">{s.name}</span>
              <span className="ml-auto tabular-nums text-terminal-dim">{s.val}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 border-t border-white/[0.08] pt-3">
        <div className="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-wider text-terminal-dim">
          <span>Currency</span>
        </div>
        <div className="flex h-2 w-full overflow-hidden rounded-full bg-pf-bg">
          <div className="h-full bg-pf-accent" style={{ width: `${CURRENCY[0].val}%` }} />
          <div className="h-full bg-pf-assigned" style={{ width: `${CURRENCY[1].val}%` }} />
        </div>
        <div className="mt-1.5 flex justify-between text-[11px] tabular-nums text-terminal-dim">
          <span className="text-pf-accent">USD {CURRENCY[0].val}%</span>
          <span className="text-pf-assigned">CAD {CURRENCY[1].val}%</span>
        </div>
      </div>
    </PfCard>
  );
}
