"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";

interface ChartPanelProps {
  ticker: string;
  onClose?: () => void;
}

type ChartType = "bollinger" | "rsi" | "atr" | "ohlcv" | "strangle";

interface ChartData {
  date: string;
  close?: number;
  open?: number;
  high?: number;
  low?: number;
  upper?: number | null;
  middle?: number | null;
  lower?: number | null;
  volume?: number;
  rsi_14?: number | null;
  rsi_2?: number | null;
  atr?: number | null;
  atr_pct?: number | null;
  sma20?: number | null;
  sma50?: number | null;
  score?: number;
  phase?: string;
  bb_score?: number;
  atr_score?: number;
  rsi_score?: number;
  trend_score?: number;
  range_score?: number;
}

interface AnalysisData {
  ticker: string;
  spotPrice: number;
  marketCap: number;
  peRatio: number;
  beta: number;
  sector: string;
  iv30d: number;
  rv30d: number;
  ivRank: number;
  volRiskPremium: number;
  strangleScore: number;
  stranglePhase: string;
  strangleRecommendation: string;
  wheelScore: number;
  wheelRecommendation: string;
  daysToEarnings: number | null;
  nextEarningsDate: string | null;
  vixLevel: number;
  riskFreeRate: number;
  creditRating: string;
}

const CHART_COLORS = {
  price: "#e2e8f0",
  upper: "#ef4444",
  middle: "#eab308",
  lower: "#22c55e",
  area: "#06b6d4",
  volume: "#334155",
  rsi: "#8b5cf6",
  rsi2: "#f97316",
  atr: "#06b6d4",
  sma20: "#eab308",
  sma50: "#8b5cf6",
  score: "#22c55e",
  grid: "#1e293b",
};

export function ChartPanel({ ticker, onClose }: ChartPanelProps) {
  const [chartType, setChartType] = useState<ChartType>("bollinger");
  const [days, setDays] = useState(120);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchChart = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `/api/engine?action=chart&chart_type=${chartType}&ticker=${ticker}&days=${days}`
      );
      if (res.ok) {
        const json = await res.json();
        setChartData(json.data || []);
      }
    } catch {
      // silent
    }
    setLoading(false);
  }, [ticker, chartType, days]);

  const fetchAnalysis = useCallback(async () => {
    try {
      const res = await fetch(`/api/engine?action=analyze&ticker=${ticker}`);
      if (res.ok) {
        setAnalysis(await res.json());
      }
    } catch {
      // silent
    }
  }, [ticker]);

  useEffect(() => {
    fetchChart();
    fetchAnalysis();
  }, [fetchChart, fetchAnalysis]);

  const chartButtons: { type: ChartType; label: string }[] = [
    { type: "bollinger", label: "BB" },
    { type: "ohlcv", label: "PRICE" },
    { type: "rsi", label: "RSI" },
    { type: "atr", label: "ATR" },
    { type: "strangle", label: "STRANGLE" },
  ];

  const dayButtons = [
    { d: 30, label: "1M" },
    { d: 60, label: "2M" },
    { d: 120, label: "6M" },
    { d: 252, label: "1Y" },
    { d: 504, label: "2Y" },
  ];

  return (
    <TerminalPanel
      title={`${ticker} CHART`}
      tag="ANALYSIS"
      headerRight={
        onClose ? (
          <button
            onClick={onClose}
            className="text-terminal-dim hover:text-terminal-text text-xs px-1"
          >
            ✕
          </button>
        ) : null
      }
    >
      {/* Analysis Summary */}
      {analysis && (
        <>
          <div className="flex items-center justify-between mb-1">
            <span className="text-terminal-amber font-bold">
              ${analysis.spotPrice.toFixed(2)}
            </span>
            <span className="text-terminal-dim text-[9px]">
              {analysis.sector}
            </span>
            <TerminalBadge
              variant={
                analysis.wheelScore >= 75
                  ? "green"
                  : analysis.wheelScore >= 55
                    ? "amber"
                    : "red"
              }
            >
              WHEEL {analysis.wheelScore.toFixed(0)}
            </TerminalBadge>
          </div>
          <div className="grid grid-cols-4 gap-1 mb-1 text-[9px]">
            <TerminalRow
              label="IV"
              value={`${analysis.iv30d.toFixed(1)}%`}
            />
            <TerminalRow
              label="RV"
              value={`${analysis.rv30d.toFixed(1)}%`}
            />
            <TerminalRow
              label="VRP"
              value={`${analysis.volRiskPremium > 0 ? "+" : ""}${analysis.volRiskPremium.toFixed(1)}%`}
              valueColor={
                analysis.volRiskPremium > 0
                  ? "text-terminal-green"
                  : "text-terminal-red"
              }
            />
            <TerminalRow
              label="Beta"
              value={analysis.beta.toFixed(2)}
            />
          </div>
          {analysis.strangleScore > 0 && (
            <div className="flex items-center gap-2 mb-1 text-[9px]">
              <span className="text-terminal-dim">Strangle:</span>
              <TerminalBadge
                variant={
                  analysis.strangleScore >= 70
                    ? "green"
                    : analysis.strangleScore >= 50
                      ? "amber"
                      : "red"
                }
              >
                {analysis.strangleScore.toFixed(0)}/100{" "}
                {analysis.stranglePhase}
              </TerminalBadge>
            </div>
          )}
          <TerminalDivider />
        </>
      )}

      {/* Chart Type Selector */}
      <div className="flex items-center gap-1 mb-1">
        {chartButtons.map((btn) => (
          <button
            key={btn.type}
            onClick={() => setChartType(btn.type)}
            className={`px-2 py-[2px] text-[9px] font-mono border ${
              chartType === btn.type
                ? "border-terminal-blue text-terminal-blue bg-terminal-blue/10"
                : "border-terminal-border text-terminal-dim hover:text-terminal-text"
            }`}
          >
            {btn.label}
          </button>
        ))}
        <span className="mx-1 text-terminal-border">│</span>
        {dayButtons.map((btn) => (
          <button
            key={btn.d}
            onClick={() => setDays(btn.d)}
            className={`px-1 py-[2px] text-[9px] font-mono ${
              days === btn.d
                ? "text-terminal-amber"
                : "text-terminal-dim hover:text-terminal-text"
            }`}
          >
            {btn.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="h-[280px] w-full">
        {loading ? (
          <div className="flex items-center justify-center h-full text-terminal-dim">
            Loading chart data...
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-dim">
            No data available
          </div>
        ) : chartType === "bollinger" ? (
          <BollingerChart data={chartData} />
        ) : chartType === "rsi" ? (
          <RSIChart data={chartData} />
        ) : chartType === "atr" ? (
          <ATRChart data={chartData} />
        ) : chartType === "strangle" ? (
          <StrangleChart data={chartData} />
        ) : (
          <OHLCVChart data={chartData} />
        )}
      </div>
    </TerminalPanel>
  );
}

// ─── Bollinger Bands Chart ─────────────────────────────────────────────

function BollingerChart({ data }: { data: ChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v) => v.slice(5)}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          tick={{ fontSize: 8, fill: "#64748b" }}
          domain={["auto", "auto"]}
          width={50}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
        />
        <Area
          dataKey="upper"
          stroke="none"
          fill={CHART_COLORS.area}
          fillOpacity={0.05}
        />
        <Area
          dataKey="lower"
          stroke="none"
          fill="#0f172a"
          fillOpacity={1}
        />
        <Line
          dataKey="upper"
          stroke={CHART_COLORS.upper}
          strokeWidth={1}
          dot={false}
          strokeDasharray="4 2"
        />
        <Line
          dataKey="middle"
          stroke={CHART_COLORS.middle}
          strokeWidth={1}
          dot={false}
        />
        <Line
          dataKey="lower"
          stroke={CHART_COLORS.lower}
          strokeWidth={1}
          dot={false}
          strokeDasharray="4 2"
        />
        <Line
          dataKey="close"
          stroke={CHART_COLORS.price}
          strokeWidth={1.5}
          dot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── RSI Chart ─────────────────────────────────────────────────────────

function RSIChart({ data }: { data: ChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v) => v.slice(5)}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          tick={{ fontSize: 8, fill: "#64748b" }}
          domain={[0, 100]}
          width={30}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
        />
        <ReferenceLine y={70} stroke={CHART_COLORS.upper} strokeDasharray="3 3" />
        <ReferenceLine y={30} stroke={CHART_COLORS.lower} strokeDasharray="3 3" />
        <ReferenceLine y={50} stroke="#475569" strokeDasharray="2 4" />
        <Line
          dataKey="rsi_14"
          stroke={CHART_COLORS.rsi}
          strokeWidth={1.5}
          dot={false}
          name="RSI(14)"
        />
        <Line
          dataKey="rsi_2"
          stroke={CHART_COLORS.rsi2}
          strokeWidth={1}
          dot={false}
          name="RSI(2)"
          opacity={0.6}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── ATR Chart ─────────────────────────────────────────────────────────

function ATRChart({ data }: { data: ChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v) => v.slice(5)}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          tick={{ fontSize: 8, fill: "#64748b" }}
          domain={["auto", "auto"]}
          width={40}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
        />
        <Area
          dataKey="atr"
          stroke={CHART_COLORS.atr}
          fill={CHART_COLORS.atr}
          fillOpacity={0.15}
          strokeWidth={1.5}
          name="ATR(14)"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── OHLCV Chart ───────────────────────────────────────────────────────

function OHLCVChart({ data }: { data: ChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v) => v.slice(5)}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          yAxisId="price"
          tick={{ fontSize: 8, fill: "#64748b" }}
          domain={["auto", "auto"]}
          width={50}
        />
        <YAxis
          yAxisId="vol"
          orientation="right"
          tick={{ fontSize: 8, fill: "#334155" }}
          width={40}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
        />
        <Bar
          yAxisId="vol"
          dataKey="volume"
          fill={CHART_COLORS.volume}
          opacity={0.4}
        />
        <Line
          yAxisId="price"
          dataKey="close"
          stroke={CHART_COLORS.price}
          strokeWidth={1.5}
          dot={false}
        />
        <Line
          yAxisId="price"
          dataKey="sma20"
          stroke={CHART_COLORS.sma20}
          strokeWidth={1}
          dot={false}
          strokeDasharray="3 2"
        />
        <Line
          yAxisId="price"
          dataKey="sma50"
          stroke={CHART_COLORS.sma50}
          strokeWidth={1}
          dot={false}
          strokeDasharray="3 2"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Strangle Timing Chart ─────────────────────────────────────────────

function StrangleChart({ data }: { data: ChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v) => v.slice(5)}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          tick={{ fontSize: 8, fill: "#64748b" }}
          domain={[0, 100]}
          width={30}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
        />
        <ReferenceLine y={80} stroke={CHART_COLORS.lower} strokeDasharray="3 3" label={{ value: "Strong Entry", fill: "#22c55e", fontSize: 8 }} />
        <ReferenceLine y={60} stroke={CHART_COLORS.middle} strokeDasharray="3 3" label={{ value: "Conditional", fill: "#eab308", fontSize: 8 }} />
        <Area
          dataKey="score"
          stroke={CHART_COLORS.score}
          fill={CHART_COLORS.score}
          fillOpacity={0.15}
          strokeWidth={2}
          name="Entry Score"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
