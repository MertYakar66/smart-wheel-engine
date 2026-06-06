"use client";

import { Coins, Percent, Sparkles, TrendingDown, Trophy, Wallet } from "lucide-react";
import { fmtUsd } from "@/lib/cockpit-trust";
import { ACCOUNT, RETURNS } from "./mock";
import {
  PfCard,
  PeriodToggle,
  fmtSignedPct,
  fmtSignedUsd,
  pnlColor,
  type Period,
} from "./parts";

function Kpi({
  label,
  value,
  sub,
  subColor,
  icon: Icon,
  accentTop,
  children,
}: {
  label: string;
  value: string;
  sub?: string;
  subColor?: string;
  icon: React.ElementType;
  accentTop?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <PfCard accentTop={accentTop} pad={false}>
      <div className="p-4">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wider text-terminal-dim">
            {label}
          </span>
          <Icon className="h-3.5 w-3.5 text-terminal-dim" />
        </div>
        <div className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-terminal-text">
          {value}
        </div>
        {sub && (
          <div className={`mt-1 text-xs tabular-nums ${subColor ?? "text-terminal-dim"}`}>
            {sub}
          </div>
        )}
        {children}
      </div>
    </PfCard>
  );
}

export function KpiCards({
  period,
  onPeriod,
}: {
  period: Period;
  onPeriod: (p: Period) => void;
}) {
  const r = RETURNS[period];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      <Kpi
        label="Net Liquidation"
        value={fmtUsd(ACCOUNT.netLiq)}
        sub={`${fmtSignedUsd(ACCOUNT.dayChangeUsd)} (${fmtSignedPct(ACCOUNT.dayChangePct)}) today`}
        subColor={pnlColor(ACCOUNT.dayChangeUsd)}
        icon={Wallet}
        accentTop
      />
      <Kpi
        label={`Total Return · ${period}`}
        value={fmtSignedPct(r.pct)}
        sub={`${fmtSignedUsd(r.usd)}`}
        subColor={pnlColor(r.usd)}
        icon={r.pct >= 0 ? Sparkles : TrendingDown}
        accentTop
      >
        <div className="mt-3">
          <PeriodToggle value={period} onChange={onPeriod} size="xs" />
        </div>
      </Kpi>
      <Kpi
        label="Unrealized P&L"
        value={fmtSignedUsd(ACCOUNT.unrealizedPnl)}
        sub="open positions, mark-to-market"
        subColor={pnlColor(ACCOUNT.unrealizedPnl)}
        icon={TrendingDown}
      />
      <Kpi
        label="Realized P&L · YTD"
        value={fmtSignedUsd(ACCOUNT.realizedYtd)}
        sub="closed trades, 2026"
        subColor={pnlColor(ACCOUNT.realizedYtd)}
        icon={Coins}
      />
      <Kpi
        label="Premium · 30d"
        value={fmtSignedUsd(ACCOUNT.premium30d)}
        sub="option income collected"
        subColor="text-pf-accent"
        icon={Percent}
      />
      <Kpi
        label="Win Rate"
        value={`${Math.round(ACCOUNT.winRate * 100)}%`}
        sub="closed wheel cycles"
        icon={Trophy}
      />
    </div>
  );
}
