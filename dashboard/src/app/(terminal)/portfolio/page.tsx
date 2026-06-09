"use client";

// Portfolio performance viewer (design D26). Read-only + observational —
// no EV authority, no order routing. Data comes live from the engine's
// read-only /api/portfolio/* endpoints (point-in-time IBKR snapshot →
// ibkr_portfolio_adapter), with components/portfolio/mock.ts as the typed
// per-slice fallback when the engine is unreachable.

import { useEffect, useState } from "react";
import Link from "next/link";

import { Allocation } from "@/components/portfolio/allocation";
import { AskBar } from "@/components/portfolio/ask-bar";
import { EquityCurve } from "@/components/portfolio/equity-curve";
import { HoldingsTable } from "@/components/portfolio/holdings-table";
import { KpiCards } from "@/components/portfolio/kpi-cards";
import { RiskRadar } from "@/components/portfolio/risk-radar";
import { usePortfolioData } from "@/components/portfolio/use-portfolio-data";
import { fmtUsd } from "@/lib/cockpit-trust";
import {
  fmtSignedPct,
  fmtSignedUsd,
  pnlColor,
  provenanceSummary,
  type Period,
} from "@/components/portfolio/parts";

const SUBNAV = ["Overview", "Holdings", "Income", "Risk", "Ask"];

export default function PortfolioPage() {
  const [period, setPeriod] = useState<Period>("YTD");
  const { data, loading, sources } = usePortfolioData();
  const account = data.account;
  // Honest header provenance (browser-QA §D): derived from the per-slice
  // sources, never hardcoded — so committed demo fixtures read "Demo data",
  // not "Live IBKR snapshot".
  const prov = provenanceSummary(Object.values(sources), loading);

  // Format the snapshot time CLIENT-SIDE only — toLocaleString is timezone-
  // dependent, so doing it during render would differ between the server (UTC)
  // and the browser (local tz) and trip a hydration mismatch. Same pattern the
  // cockpit uses for staleDays.
  const [asOf, setAsOf] = useState<string>("");
  useEffect(() => {
    if (!account.asOf) return;
    // One-shot client-only format to dodge the SSR(UTC)/CSR(local) timezone
    // hydration mismatch — toLocaleString is tz-dependent. This runs once on
    // mount / when asOf changes; it is not a cascading update.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAsOf(
      new Date(account.asOf).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    );
  }, [account.asOf]);

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
            <span className="text-lg font-semibold tabular-nums">{fmtUsd(account.netLiq)}</span>
            <span className={`text-xs tabular-nums ${pnlColor(account.dayChangeUsd ?? 0)}`}>
              {account.dayChangeUsd == null
                ? "—"
                : `${fmtSignedUsd(account.dayChangeUsd)} (${fmtSignedPct(account.dayChangePct ?? 0)})`}
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-terminal-dim">
            <span className={`h-1.5 w-1.5 rounded-full ${prov.dot}`} title={`Data source: ${prov.label}`} />
            <span className={prov.text}>{prov.label}</span>
            <span>· as of {asOf || "…"}</span>
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
        <KpiCards period={period} onPeriod={setPeriod} account={account} returns={data.returns} />

        {/* Live-NAV / as-of-curve honesty: the NAV in the header is live from
            the snapshot, but the period returns are computed off the equity
            curve, whose last point lags the live NAV until the next history
            append. Surface that explicitly so it is never silent. */}
        {data.equity.length > 0 && (
          <p className="-mt-2 text-[11px] text-terminal-dim">
            Returns as of {data.equity[data.equity.length - 1].m} · NAV live
          </p>
        )}

        {/* Hero chart + allocation */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <EquityCurve period={period} onPeriod={setPeriod} equity={data.equity} source={sources.history} />
          </div>
          <Allocation sectors={data.sectors} currency={data.currency} source={sources.risk} />
        </div>

        {/* Holdings + risk radar */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <HoldingsTable holdings={data.holdings} source={sources.positions} />
          </div>
          <RiskRadar
            account={account}
            singleName={data.singleName}
            sectorExposure={data.sectorExposure}
            caps={data.caps}
            marginHealth={data.margin.cushionPct}
            source={sources.risk}
          />
        </div>

        <AskBar />

        <p className="pb-2 text-center text-[10px] text-terminal-dim">
          {loading
            ? "Loading book…"
            : `${prov.label} · read-only performance viewer (design D26). Realized P&L shown distinctly from forward EV — observational only.`}
        </p>
      </main>
    </div>
  );
}
