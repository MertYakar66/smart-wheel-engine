"use client";

// Portfolio performance viewer — AESTHETICS round (Slate-teal direction +
// gradient hero chart). All data is mock (see components/portfolio/mock.ts);
// the /api/portfolio/* endpoints + IBKR snapshot feed land in the functionality
// round. Read-only, observational — no EV authority, no order routing.

import { useState } from "react";
import Link from "next/link";

import { Allocation } from "@/components/portfolio/allocation";
import { AskBar } from "@/components/portfolio/ask-bar";
import { EquityCurve } from "@/components/portfolio/equity-curve";
import { HoldingsTable } from "@/components/portfolio/holdings-table";
import { KpiCards } from "@/components/portfolio/kpi-cards";
import { RiskRadar } from "@/components/portfolio/risk-radar";
import { ACCOUNT } from "@/components/portfolio/mock";
import { fmtUsd } from "@/lib/cockpit-trust";
import { fmtSignedPct, fmtSignedUsd, pnlColor, type Period } from "@/components/portfolio/parts";

const SUBNAV = ["Overview", "Holdings", "Income", "Risk", "Ask"];

export default function PortfolioPage() {
  const [period, setPeriod] = useState<Period>("YTD");

  const asOf = new Date(ACCOUNT.asOf).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <div className="min-h-screen overflow-y-auto bg-pf-bg font-sans text-terminal-text">
      {/* Top bar */}
      <header className="sticky top-0 z-20 border-b border-white/[0.08] bg-pf-bg/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-1 px-5">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-pf-accent shadow-[0_0_8px_var(--color-pf-accent)]" />
            <span className="font-semibold tracking-tight text-terminal-text">Wheelhouse</span>
            <span className="text-[11px] text-terminal-dim">· Portfolio</span>
          </div>

          <div className="ml-auto flex items-baseline gap-3">
            <span className="text-[11px] uppercase tracking-wider text-terminal-dim">Net Liq</span>
            <span className="text-lg font-semibold tabular-nums">{fmtUsd(ACCOUNT.netLiq)}</span>
            <span className={`text-xs tabular-nums ${pnlColor(ACCOUNT.dayChangeUsd)}`}>
              {fmtSignedUsd(ACCOUNT.dayChangeUsd)} ({fmtSignedPct(ACCOUNT.dayChangePct)})
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-terminal-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-pf-ok" />
            as of {asOf}
          </div>
        </div>

        {/* Sub-nav */}
        <div className="mx-auto flex max-w-[1400px] items-center gap-1 px-5">
          {SUBNAV.map((s, i) => (
            <span
              key={s}
              className={`-mb-px border-b-2 px-2.5 py-2 text-xs font-medium ${
                i === 0
                  ? "border-pf-accent text-terminal-text"
                  : "border-transparent text-terminal-dim hover:text-terminal-text"
              }`}
            >
              {s}
            </span>
          ))}
          <Link
            href="/cockpit"
            className="ml-auto py-2 text-[11px] text-terminal-dim hover:text-terminal-text"
          >
            ← Cockpit
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] space-y-4 px-5 py-5">
        <KpiCards period={period} onPeriod={setPeriod} />

        {/* Hero chart + allocation */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <EquityCurve period={period} onPeriod={setPeriod} />
          </div>
          <Allocation />
        </div>

        {/* Holdings + risk radar */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <HoldingsTable />
          </div>
          <RiskRadar />
        </div>

        <AskBar />

        <p className="pb-2 text-center text-[10px] text-terminal-dim">
          Mock data · aesthetics preview. Read-only performance viewer (design D26) —
          live IBKR feed + /api/portfolio wiring lands next.
        </p>
      </main>
    </div>
  );
}
