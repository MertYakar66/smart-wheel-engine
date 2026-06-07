"use client";

import { Coins, Percent, Sparkles, TrendingDown, Trophy, Wallet } from "lucide-react";
import { fmtUsd } from "@/lib/cockpit-trust";
import { ACCOUNT as MOCK_ACCOUNT, RETURNS as MOCK_RETURNS } from "./mock";
import {
  PfCard,
  PeriodToggle,
  fmtSignedPct,
  fmtSignedUsd,
  orDash,
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
  account = MOCK_ACCOUNT,
  returns = MOCK_RETURNS,
}: {
  period: Period;
  onPeriod: (p: Period) => void;
  account?: typeof MOCK_ACCOUNT;
  returns?: typeof MOCK_RETURNS;
}) {
  const r = returns[period];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      <Kpi
        label="Net Liquidation"
        value={fmtUsd(account.netLiq)}
        sub={
          account.dayChangeUsd == null
            ? "today —"
            : `${fmtSignedUsd(account.dayChangeUsd)} (${fmtSignedPct(account.dayChangePct ?? 0)}) today`
        }
        subColor={pnlColor(account.dayChangeUsd ?? 0)}
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
        value={orDash(account.unrealizedPnl, (v) => fmtSignedUsd(v))}
        sub="open positions, mark-to-market"
        subColor={pnlColor(account.unrealizedPnl ?? 0)}
        icon={TrendingDown}
      />
      <Kpi
        label="Realized P&L · YTD"
        value={orDash(account.realizedYtd, (v) => fmtSignedUsd(v))}
        sub="closed trades, 2026"
        subColor={pnlColor(account.realizedYtd ?? 0)}
        icon={Coins}
      />
      <Kpi
        label="Premium · 30d"
        value={orDash(account.premium30d, (v) => fmtSignedUsd(v))}
        sub="option income collected"
        subColor="text-pf-accent"
        icon={Percent}
      />
      <Kpi
        label="Win Rate"
        value={orDash(account.winRate, (v) => `${Math.round(v * 100)}%`)}
        sub="closed wheel cycles"
        icon={Trophy}
      />
    </div>
  );
}
