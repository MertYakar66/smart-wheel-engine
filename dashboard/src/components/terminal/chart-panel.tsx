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

type ChartType = "bollinger" | "rsi" | "atr" | "ohlcv" | "strangle" | "payoff" | "memo";

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
  const [memoText, setMemoText] = useState<string>("");

  // Payoff-specific state
  const [payoffMeta, setPayoffMeta] = useState<{
    strike: number;
    premium: number;
    breakeven: number;
    maxProfit: number;
    maxLoss: number;
    strategy: string;
  } | null>(null);

  // Strike recommendations
  const [strikes, setStrikes] = useState<Array<{
    strike: number;
    premium: number;
    delta: number;
    probabilityOtm: number;
    annualizedReturn: number;
    score: number;
    breakeven: number;
    distanceFromSpotPct: number;
  }>>([]);

  // Expected move
  const [expectedMove, setExpectedMove] = useState<{
    bands: Array<{ label: string; upper: number; lower: number; probability_within: number; move_pct: number }>;
    period_vol: number;
  } | null>(null);

  const fetchChart = useCallback(async () => {
    setLoading(true);
    try {
      if (chartType === "payoff") {
        // Fetch payoff diagram + strikes + expected move in parallel with timeout
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        try {
          const [payRes, strikesRes, emRes] = await Promise.all([
            fetch(`/api/engine?action=payoff&ticker=${ticker}&strategy=csp&dte=45`, { signal: controller.signal }),
            fetch(`/api/engine?action=strikes&ticker=${ticker}&strategy=csp&dte=45`, { signal: controller.signal }),
            fetch(`/api/engine?action=expected_move&ticker=${ticker}&dte=45`, { signal: controller.signal }),
          ]);
          if (payRes.ok) {
            const json = await payRes.json();
            setChartData(json.data || []);
            setPayoffMeta({
              strike: json.strike,
              premium: json.premium,
              breakeven: json.breakeven,
              maxProfit: json.maxProfit,
              maxLoss: json.maxLoss,
              strategy: json.strategy,
            });
          }
          if (strikesRes.ok) {
            const json = await strikesRes.json();
            setStrikes(json.recommendations || []);
          }
          if (emRes.ok) {
            const json = await emRes.json();
            setExpectedMove({ bands: json.bands || [], period_vol: json.period_vol || 0 });
          }
        } finally {
          clearTimeout(timeout);
        }
      } else if (chartType === "memo") {
        const res = await fetch(`/api/engine?action=memo&ticker=${ticker}`);
        if (res.ok) {
          const json = await res.json();
          setMemoText(json.memo || "No memo generated. Ensure Ollama is running.");
        }
      } else {
        const res = await fetch(
          `/api/engine?action=chart&chart_type=${chartType}&ticker=${ticker}&days=${days}`
        );
        if (res.ok) {
          const json = await res.json();
          setChartData(json.data || []);
        }
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
    { type: "strangle", label: "TIMING" },
    { type: "payoff", label: "PAYOFF" },
    { type: "memo", label: "AI MEMO" },
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
          <div className="grid grid-cols-4 gap-1 mb-1 text-[9px]">
            <TerminalRow
              label="IV Rank"
              value={`${(analysis.ivRank < 1 ? analysis.ivRank * 100 : analysis.ivRank).toFixed(1)}%`}
              valueColor={
                (analysis.ivRank < 1 ? analysis.ivRank : analysis.ivRank / 100) >= 0.5
                  ? "text-terminal-green"
                  : "text-terminal-dim"
              }
            />
            <TerminalRow
              label="VIX"
              value={analysis.vixLevel.toFixed(1)}
            />
            <TerminalRow
              label="Earn"
              value={analysis.daysToEarnings != null ? `${analysis.daysToEarnings}d` : "N/A"}
            />
            <TerminalRow
              label="Rating"
              value={analysis.creditRating || "N/A"}
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
      <div className="h-[280px] w-full" style={{ minWidth: 0, minHeight: 0 }}>
        {loading ? (
          <div className="flex items-center justify-center h-full text-terminal-dim">
            Loading chart data...
          </div>
        ) : chartType === "memo" ? (
          <div className="h-full overflow-y-auto px-2 py-1 text-[11px] text-terminal-text whitespace-pre-wrap font-mono leading-relaxed">
            {memoText || "Generating trade memo... (requires Ollama with qwen2.5:72b)"}
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-dim">
            No data available
          </div>
        ) : (
          <div key={chartType} className="w-full h-full" style={{ minWidth: 0, minHeight: 0 }}>
            {chartType === "bollinger" ? (
              <BollingerChart data={chartData} />
            ) : chartType === "rsi" ? (
              <RSIChart data={chartData} />
            ) : chartType === "atr" ? (
              <ATRChart data={chartData} />
            ) : chartType === "strangle" ? (
              <StrangleChart data={chartData} />
            ) : chartType === "payoff" ? (
              <PayoffChart data={chartData} meta={payoffMeta} />
            ) : (
              <OHLCVChart data={chartData} />
            )}
          </div>
        )}
      </div>

      {/* Expected Move Bands (shown for payoff view) */}
      {chartType === "payoff" && expectedMove && expectedMove.bands.length > 0 && (
        <>
          <TerminalDivider />
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ EXPECTED MOVE ({expectedMove.period_vol}% period vol)
          </div>
          {expectedMove.bands.map((band) => (
            <div
              key={band.label}
              className="flex items-center justify-between py-[1px] text-[10px]"
            >
              <span className="text-terminal-dim w-8">{band.label}</span>
              <span className="text-terminal-green">${band.lower}</span>
              <span className="text-terminal-dim">─</span>
              <span className="text-terminal-red">${band.upper}</span>
              <span className="text-terminal-dim">
                ±{band.move_pct}% ({band.probability_within}% prob)
              </span>
            </div>
          ))}
        </>
      )}

      {/* Strike Recommendations (shown for payoff view) */}
      {chartType === "payoff" && strikes.length > 0 && (
        <>
          <TerminalDivider />
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ CSP STRIKE RECOMMENDATIONS
          </div>
          <div className="flex items-center justify-between py-[1px] text-[9px] text-terminal-dim">
            <span className="w-14">STRIKE</span>
            <span className="w-12">PREM</span>
            <span className="w-10">DELTA</span>
            <span className="w-12">P(OTM)</span>
            <span className="w-14">ANN.RET</span>
            <span className="w-10">SCORE</span>
          </div>
          {strikes.map((s, i) => (
            <div
              key={i}
              className="flex items-center justify-between py-[2px] hover:bg-terminal-border/30 text-[10px]"
            >
              <span className="w-14 text-terminal-amber">${s.strike}</span>
              <span className="w-12 text-terminal-green">${s.premium}</span>
              <span className="w-10 text-terminal-text">
                {s.delta.toFixed(2)}
              </span>
              <span
                className={`w-12 ${
                  s.probabilityOtm >= 75
                    ? "text-terminal-green"
                    : "text-terminal-amber"
                }`}
              >
                {s.probabilityOtm}%
              </span>
              <span className="w-14 text-terminal-text">
                {s.annualizedReturn}%
              </span>
              <span
                className={`w-10 font-bold ${
                  s.score >= 70
                    ? "text-terminal-green"
                    : s.score >= 50
                      ? "text-terminal-amber"
                      : "text-terminal-text"
                }`}
              >
                {s.score}
              </span>
            </div>
          ))}
        </>
      )}
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
          tickFormatter={(v: string) => v.slice(5, 10)}
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
          tickFormatter={(v: string) => v.slice(5, 10)}
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
          tickFormatter={(v: string) => v.slice(5, 10)}
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
          tickFormatter={(v: string) => v.slice(5, 10)}
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
          tickFormatter={(v: string) => v.slice(5, 10)}
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

// ─── Payoff Chart ──────────────────────────────────────────────────────

function PayoffChart({
  data,
  meta,
}: {
  data: ChartData[];
  meta: { strike: number; premium: number; breakeven: number; maxProfit: number; strategy: string } | null;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
        <XAxis
          dataKey="price"
          tick={{ fontSize: 8, fill: "#64748b" }}
          tickFormatter={(v: number) => `$${v}`}
          interval={Math.floor(data.length / 8)}
        />
        <YAxis
          tick={{ fontSize: 8, fill: "#64748b" }}
          width={50}
          tickFormatter={(v: number) => `$${v}`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            fontSize: 10,
            color: "#e2e8f0",
          }}
          formatter={(value: number) => [`$${value.toFixed(0)}`, "P&L"]}
          labelFormatter={(label: number) => `Price: $${label}`}
        />
        <ReferenceLine y={0} stroke="#475569" strokeWidth={1} />
        {meta && (
          <>
            <ReferenceLine
              x={meta.strike}
              stroke={CHART_COLORS.upper}
              strokeDasharray="3 3"
              label={{ value: `Strike $${meta.strike}`, fill: "#ef4444", fontSize: 9 }}
            />
            <ReferenceLine
              x={meta.breakeven}
              stroke={CHART_COLORS.middle}
              strokeDasharray="3 3"
              label={{ value: `BE $${meta.breakeven}`, fill: "#eab308", fontSize: 9 }}
            />
          </>
        )}
        <Area
          dataKey="pnl"
          stroke="#22c55e"
          fill="#22c55e"
          fillOpacity={0.1}
          strokeWidth={2}
          name="P&L"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
