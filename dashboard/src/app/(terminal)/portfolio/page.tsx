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
import { IncomePanel } from "@/components/portfolio/income-panel";
import { KpiCards } from "@/components/portfolio/kpi-cards";
import { MarginPanel } from "@/components/portfolio/margin-panel";
import { RiskRadar } from "@/components/portfolio/risk-radar";
import { usePortfolioData } from "@/components/portfolio/use-portfolio-data";
import { WheelhouseHeader } from "@/components/shell/wheelhouse-header";
import { fmtUsd } from "@/lib/cockpit-trust";
import {
  fmtSignedPct,
  fmtSignedUsd,
  pnlColor,
  provenanceSummary,
  type Period,
} from "@/components/portfolio/parts";

// In-page sections, in vertical (scroll) order. The subnav scrolls to these
// and highlights whichever is in view. `id`s are matched by the scroll-spy
// observer and the `goTo` click handler below.
const SECTIONS = [
  { label: "Overview", id: "pf-overview" },
  { label: "Performance", id: "pf-performance" },
  { label: "Income", id: "pf-income" },
  { label: "Holdings", id: "pf-holdings" },
  { label: "Risk", id: "pf-risk" },
  { label: "Ask", id: "pf-ask" },
] as const;

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

  // Which subnav section is currently in view (drives the active tab).
  const [active, setActive] = useState<string>(SECTIONS[0].id);

  // Scroll-spy: highlight the section nearest the top as the page scrolls.
  // rootMargin offsets for the sticky header (the ~h-14 bar + the subnav row)
  // so a section reads as "active" once it clears the chrome. root:null
  // observes against the viewport, which updates on window OR nested-div
  // scroll, so this is robust regardless of which element actually scrolls.
  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter(
      (el): el is HTMLElement => el != null
    );
    if (els.length === 0) return;
    const obs = new IntersectionObserver(
      (entries) => {
        const top = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (top) setActive(top.target.id);
      },
      { rootMargin: "-104px 0px -55% 0px", threshold: 0 }
    );
    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  const goTo = (id: string) => {
    setActive(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="min-h-screen overflow-y-auto bg-pf-bg font-sans text-terminal-text">
      {/* Top bar (shared Wheelhouse chrome) */}
      <WheelhouseHeader
        page="Portfolio"
        right={
          <>
            <span className="text-[11px] uppercase tracking-wider text-terminal-dim">Net Liq</span>
            <span className="text-lg font-semibold tabular-nums">{fmtUsd(account.netLiq)}</span>
            <span className={`text-xs tabular-nums ${pnlColor(account.dayChangeUsd ?? 0)}`}>
              {account.dayChangeUsd == null
                ? "—"
                : `${fmtSignedUsd(account.dayChangeUsd)} (${fmtSignedPct(account.dayChangePct ?? 0)})`}
            </span>
          </>
        }
        status={
          <>
            <span className={`h-1.5 w-1.5 rounded-full ${prov.dot}`} title={`Data source: ${prov.label}`} />
            <span className={prov.text}>{prov.label}</span>
            <span>· as of {asOf || "…"}</span>
          </>
        }
      >
        {/* In-page section tabs — click to scroll, active = section in view */}
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => goTo(s.id)}
            aria-current={active === s.id ? "location" : undefined}
            className={`-mb-px border-b-2 px-2.5 py-2 text-xs font-medium transition-colors focus-visible:rounded focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-pf-accent/50 ${
              active === s.id
                ? "border-pf-accent text-terminal-text"
                : "border-transparent text-terminal-dim hover:text-terminal-text"
            }`}
          >
            {s.label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-3 py-2 text-[11px] text-terminal-dim">
          <Link href="/cockpit" className="hover:text-terminal-text">Cockpit</Link>
          <Link href="/terminal" className="hover:text-terminal-text">Terminal</Link>
          <Link href="/top" className="hover:text-terminal-text">News</Link>
        </div>
      </WheelhouseHeader>

      <main className="mx-auto max-w-[1400px] space-y-4 px-5 py-5">
        <section id="pf-overview" className="scroll-mt-28">
          <KpiCards period={period} onPeriod={setPeriod} account={account} returns={data.returns} />
        </section>

        {/* Live-NAV / as-of-curve honesty: the NAV in the header is live from
            the snapshot, but the period returns are computed off the equity
            curve, whose last point lags the live NAV until the next history
            append. Surface that explicitly so it is never silent. The ledger
            claim is DERIVED, not hardcoded: it only renders when the summary
            actually carries ledger-backed KPIs (snapshot-only drops null them). */}
        {data.equity.length > 0 && (
          <p className="-mt-2 text-[11px] text-terminal-dim">
            Returns as of {data.equity[data.equity.length - 1].m} · NAV live
            {(account.realizedYtd != null || account.winRate != null) &&
              " · Realized, Premium & Win-Rate from the trade ledger"}
          </p>
        )}

        {/* Hero chart + allocation (Performance) */}
        <div id="pf-performance" className="scroll-mt-28 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <EquityCurve
              period={period}
              onPeriod={setPeriod}
              equity={data.equity}
              stats={data.stats}
              source={sources.history}
            />
          </div>
          <Allocation sectors={data.sectors} currency={data.currency} source={sources.risk} />
        </div>

        {/* Realized income (per-name league table + monthly bars) */}
        <section id="pf-income" className="scroll-mt-28">
          <IncomePanel income={data.income} source={sources.income} />
        </section>

        {/* Holdings + risk radar + margin */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div id="pf-holdings" className="scroll-mt-28 lg:col-span-2">
            <HoldingsTable holdings={data.holdings} source={sources.positions} />
          </div>
          <div id="pf-risk" className="scroll-mt-28 space-y-4">
            <RiskRadar
              singleName={data.singleName}
              concentration={data.concentration}
              sectorExposure={data.sectorExposure}
              caps={data.caps}
              gates={data.gates}
              ivAssumption={data.ivAssumption}
              source={sources.risk}
            />
            <MarginPanel
              account={account}
              margin={data.margin}
              holdings={data.holdings}
              source={sources.risk}
            />
          </div>
        </div>

        <section id="pf-ask" className="scroll-mt-28">
          <AskBar />
        </section>

        <p className="pb-2 text-center text-[10px] text-terminal-dim">
          {loading
            ? "Loading book…"
            : `${prov.label} · read-only performance viewer (design D26). Realized P&L shown distinctly from forward EV — observational only.`}
        </p>
      </main>
    </div>
  );
}
