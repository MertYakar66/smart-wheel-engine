"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtUsd } from "@/lib/cockpit-trust";
import { EQUITY, PERIOD_WINDOW } from "./mock";
import { PeriodToggle, PfCard, type Period } from "./parts";

const ACCENT = "#2dd4bf";
const BENCH = "#7c8696";
const tipStyle = {
  background: "#0b0f14",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 11,
  color: "#e2e8f0",
};

export function EquityCurve({
  period,
  onPeriod,
}: {
  period: Period;
  onPeriod: (p: Period) => void;
}) {
  const [tab, setTab] = useState<"equity" | "premium">("equity");
  const data = useMemo(
    () => EQUITY.slice(EQUITY.length - PERIOD_WINDOW[period]),
    [period]
  );

  return (
    <PfCard pad={false} className="h-full">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pt-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-terminal-text">
            {tab === "equity" ? "Portfolio Value" : "Premium Income"}
          </h3>
          <div className="inline-flex rounded-lg border border-white/[0.08] bg-pf-bg p-0.5 text-[11px]">
            {(["equity", "premium"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-2 py-1 font-medium transition-colors ${
                  tab === t
                    ? "bg-pf-accent/15 text-pf-accent"
                    : "text-terminal-dim hover:text-terminal-text"
                }`}
              >
                {t === "equity" ? "Equity" : "Premium"}
              </button>
            ))}
          </div>
          {tab === "equity" && (
            <div className="hidden items-center gap-3 text-[10px] text-terminal-dim sm:flex">
              <span className="flex items-center gap-1">
                <span className="h-0.5 w-3 rounded" style={{ background: ACCENT }} />
                Portfolio
              </span>
              <span className="flex items-center gap-1">
                <span className="h-0.5 w-3 rounded" style={{ background: BENCH }} />
                SPY
              </span>
            </div>
          )}
        </div>
        <PeriodToggle value={period} onChange={onPeriod} size="xs" />
      </div>

      <div className="px-2 pb-3 pt-4" style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          {tab === "equity" ? (
            <AreaChart data={data} margin={{ top: 6, right: 12, left: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="pfFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={ACCENT} stopOpacity={0.32} />
                  <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis
                dataKey="m"
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                minTickGap={16}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={46}
                domain={["auto", "auto"]}
                tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
              />
              <Tooltip
                contentStyle={tipStyle}
                formatter={(v, n) => [fmtUsd(Number(v)), n === "port" ? "Portfolio" : "SPY"]}
                cursor={{ stroke: "rgba(255,255,255,0.15)" }}
              />
              <Area
                type="monotone"
                dataKey="port"
                stroke={ACCENT}
                strokeWidth={2}
                fill="url(#pfFill)"
                dot={false}
                activeDot={{ r: 3, fill: ACCENT }}
              />
              <Line
                type="monotone"
                dataKey="spy"
                stroke={BENCH}
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
              />
            </AreaChart>
          ) : (
            <BarChart data={data} margin={{ top: 6, right: 12, left: 4, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis
                dataKey="m"
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                minTickGap={16}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={46}
                tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
              />
              <Tooltip
                contentStyle={tipStyle}
                formatter={(v) => [fmtUsd(Number(v)), "Premium"]}
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
              />
              <Bar dataKey="premium" fill={ACCENT} radius={[3, 3, 0, 0]} maxBarSize={26} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </PfCard>
  );
}
