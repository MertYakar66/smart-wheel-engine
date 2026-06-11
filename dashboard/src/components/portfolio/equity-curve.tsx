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
import { EQUITY as MOCK_EQUITY, type EquityPoint, type EquityStats, type Period } from "./mock";
import { PeriodToggle, PfCard, ProvenanceBadge, orDash, type SliceSource } from "./parts";

const ACCENT = "#2dd4bf";
const BENCH = "#7c8696";
const LOSS = "#f2495e";
const tipStyle = {
  background: "#0b0f14",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 11,
  color: "#e2e8f0",
};

const DAY_MS = 86_400_000;
// WINDOW_DAYS mirrors engine/ibkr_portfolio_adapter.py::_anchor_index timedelta
// constants (30/90/365) and the YTD date(year,1,1) anchor.  Both sides must
// stay in lockstep so the chart's first→last move matches the displayed return
// KPI — a one-sided tune would silently desync them.
// TODO: add a lockstep integration test that pins these constants against the
// Python adapter's _anchor_index to enforce parity at CI time.
const WINDOW_DAYS: Partial<Record<Period, number>> = { "1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365 };

type ChartPoint = EquityPoint & { spyPlot: number | null; dd: number };

/** Window the curve by calendar date, anchored at the last point STRICTLY
 * BEFORE the window start — the same anchor the period-return KPI uses, so
 * the chart's first→last move matches the displayed return. Falls back to
 * the full series (flagged) when points carry no dates or too few remain. */
function windowByDate(
  equity: EquityPoint[],
  period: Period
): { data: EquityPoint[]; fallback: string | null } {
  if (period === "All" || equity.length === 0) return { data: equity, fallback: null };
  if (equity.some((p) => !p.date)) {
    return { data: equity, fallback: "history points carry no dates" };
  }
  const last = new Date(equity[equity.length - 1].date as string);
  const start =
    period === "YTD"
      ? new Date(Date.UTC(last.getUTCFullYear(), 0, 1))
      : new Date(last.getTime() - (WINDOW_DAYS[period] ?? 0) * DAY_MS);
  let anchor = 0;
  equity.forEach((p, i) => {
    if (new Date(p.date as string) < start) anchor = i;
  });
  const data = equity.slice(anchor);
  if (data.length < 2) return { data: equity, fallback: `not enough points for ${period}` };
  return { data, fallback: null };
}

export function EquityCurve({
  period,
  onPeriod,
  equity = MOCK_EQUITY,
  stats = null,
  source,
}: {
  period: Period;
  onPeriod: (p: Period) => void;
  equity?: EquityPoint[];
  stats?: EquityStats | null;
  source?: SliceSource;
}) {
  const [tab, setTab] = useState<"equity" | "premium" | "drawdown">("equity");

  // The refresh script carries the previous SPY value forward onto daily
  // appends; a trailing run of to-the-dollar identical values is that stale
  // copy, not a flat market. Stop plotting the benchmark past its last real
  // move and say so ("SPY as of …") instead of drawing a fake flat line.
  const { lastSpyIdx, spyStale } = useMemo(() => {
    let idx = 0;
    for (let i = 1; i < equity.length; i++) if (equity[i].spy !== equity[i - 1].spy) idx = i;
    return { lastSpyIdx: idx, spyStale: equity.length > 1 && idx < equity.length - 1 };
  }, [equity]);

  const { data, fallback } = useMemo(() => {
    const w = windowByDate(equity, period);
    const data: ChartPoint[] = [];
    let peak = -Infinity;
    for (let i = 0; i < w.data.length; i++) {
      const p = w.data[i];
      peak = Math.max(peak, p.port);
      const globalIdx = equity.length - w.data.length + i;
      data.push({
        ...p,
        spyPlot: globalIdx <= lastSpyIdx ? p.spy : null,
        dd: peak > 0 ? p.port / peak - 1 : 0,
      });
    }
    return { data, fallback: w.fallback };
  }, [equity, period, lastSpyIdx]);

  const spyAsOf = equity[lastSpyIdx]?.m;
  // Premium bars: the engine emits the monthly aggregate only on the last
  // point of each month (null elsewhere) — drop the null slots entirely so
  // daily appends don't render empty x-axis stubs.
  const premiumData = useMemo(() => data.filter((p) => p.premium != null), [data]);

  return (
    <PfCard pad={false} className="h-full">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pt-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-terminal-text">
            {tab === "equity" ? "Portfolio Value" : tab === "premium" ? "Premium Income" : "Drawdown"}
          </h3>
          <ProvenanceBadge source={source} />
          <div className="inline-flex rounded-lg border border-white/[0.08] bg-pf-bg p-0.5 text-[11px]">
            {(["equity", "premium", "drawdown"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-2 py-1 font-medium transition-colors ${
                  tab === t
                    ? "bg-pf-accent/15 text-pf-accent"
                    : "text-terminal-dim hover:text-terminal-text"
                }`}
              >
                {t === "equity" ? "Equity" : t === "premium" ? "Premium" : "Drawdown"}
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
                {spyStale ? `SPY as of ${spyAsOf}` : "SPY"}
              </span>
            </div>
          )}
        </div>
        <PeriodToggle value={period} onChange={onPeriod} size="xs" />
      </div>

      {(stats || fallback) && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 pt-2 text-[10px] tabular-nums text-terminal-dim">
          {stats && (
            <span title="computed on the monthly-grain equity series (annualized at 12 periods/yr)">
              Sharpe {orDash(stats.sharpe, (v) => v.toFixed(2))} · Sortino{" "}
              {orDash(stats.sortino, (v) => v.toFixed(2))} · Max DD{" "}
              {orDash(stats.maxDrawdown, (v) => `−${(v * 100).toFixed(1)}%`)}
              {stats.maxDrawdownPeriods != null && ` (${stats.maxDrawdownPeriods} per.)`}
            </span>
          )}
          {fallback && <span className="text-pf-caution">showing full history — {fallback}</span>}
        </div>
      )}

      <div className="px-2 pb-3 pt-4">
        {/* Numeric height (not "100%") so recharts has a positive dimension on
            the first paint, before its ResizeObserver measures the parent. With
            height="100%" the container reports -1 on the first render and logs
            "The width(-1) and height(-1) of chart should be greater than 0". */}
        <ResponsiveContainer width="100%" height={300}>
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
                formatter={(v, n) =>
                  v == null ? ["—", "SPY"] : [fmtUsd(Number(v)), n === "port" ? "Portfolio" : "SPY"]
                }
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
                dataKey="spyPlot"
                stroke={BENCH}
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
              />
            </AreaChart>
          ) : tab === "premium" ? (
            <BarChart data={premiumData} margin={{ top: 6, right: 12, left: 4, bottom: 0 }}>
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
          ) : (
            <AreaChart data={data} margin={{ top: 6, right: 12, left: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="pfDdFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={LOSS} stopOpacity={0} />
                  <stop offset="100%" stopColor={LOSS} stopOpacity={0.32} />
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
                domain={["auto", 0]}
                tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={tipStyle}
                formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "Drawdown"]}
                cursor={{ stroke: "rgba(255,255,255,0.15)" }}
              />
              <Area
                type="monotone"
                dataKey="dd"
                stroke={LOSS}
                strokeWidth={1.5}
                fill="url(#pfDdFill)"
                dot={false}
                activeDot={{ r: 3, fill: LOSS }}
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
        {tab === "drawdown" && (
          <p className="px-2 pt-1 text-[10px] text-terminal-dim">
            Underwater vs the running peak within the visible window.
          </p>
        )}
      </div>
    </PfCard>
  );
}
