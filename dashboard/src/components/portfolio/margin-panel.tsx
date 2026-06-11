"use client";

// Margin & leverage panel — the survival numbers for a margined wheel book
// (loan balance, excess liquidity vs maintenance, cushion, gross leverage).
// All fields are already served on /summary and /risk; gross exposure is
// summed client-side from the legs the holdings table renders. Display-only.

import { Scale } from "lucide-react";
import { fmtUsd } from "@/lib/cockpit-trust";
import { ACCOUNT as MOCK_ACCOUNT, HOLDINGS as MOCK_HOLDINGS, type Holding } from "./mock";
import { type Margin } from "./use-portfolio-data";
import { PfCard, ProvenanceBadge, fmtSignedUsd, type SliceSource } from "./parts";

const C = { ok: "#34d399", caution: "#f5a524", breach: "#f2495e" };

/** Semicircular margin-health gauge (SVG, sampled arc — no flag ambiguity). */
function MarginGauge({ health }: { health: number }) {
  const cx = 100;
  const cy = 96;
  const r = 78;
  const arc = (v0: number, v1: number, rr = r) => {
    const pts: string[] = [];
    const steps = 40;
    for (let i = 0; i <= steps; i++) {
      const v = v0 + ((v1 - v0) * i) / steps;
      const deg = 180 * (1 - v);
      const rad = (deg * Math.PI) / 180;
      pts.push(`${(cx + rr * Math.cos(rad)).toFixed(2)} ${(cy - rr * Math.sin(rad)).toFixed(2)}`);
    }
    return "M" + pts.join(" L");
  };
  const clamped = Math.max(0, Math.min(1, health));
  const needleDeg = 180 * (1 - clamped);
  const needleRad = (needleDeg * Math.PI) / 180;
  const nx = cx + (r - 10) * Math.cos(needleRad);
  const ny = cy - (r - 10) * Math.sin(needleRad);

  return (
    <svg viewBox="0 0 200 112" className="w-full" style={{ maxWidth: 200 }}>
      <path d={arc(0, 1)} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={9} strokeLinecap="round" />
      <path d={arc(0, 0.34)} fill="none" stroke={C.breach} strokeWidth={9} strokeLinecap="round" />
      <path d={arc(0.34, 0.66)} fill="none" stroke={C.caution} strokeWidth={9} />
      <path d={arc(0.66, 1)} fill="none" stroke={C.ok} strokeWidth={9} strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={nx.toFixed(2)} y2={ny.toFixed(2)} stroke="#e2e8f0" strokeWidth={2.5} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={4} fill="#e2e8f0" />
    </svg>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-terminal-dim">{label}</div>
      <div className={`text-xs font-semibold tabular-nums ${color ?? "text-terminal-text"}`}>
        {value}
      </div>
    </div>
  );
}

export function MarginPanel({
  account = MOCK_ACCOUNT,
  margin,
  holdings = MOCK_HOLDINGS,
  source,
}: {
  account?: typeof MOCK_ACCOUNT;
  margin?: Margin;
  holdings?: Holding[];
  source?: SliceSource;
}) {
  const stressed = account.availableFunds < 0;
  // Negative cash IS the margin loan — IBKR shows no separate field.
  const loan = account.cash < 0 ? -account.cash : 0;
  // Gross exposure = Σ|position value| over every leg, vs NAV.
  const gross = holdings.reduce((s, h) => s + Math.abs(h.mktValue), 0);
  const leverage = account.netLiq > 0 ? gross / account.netLiq : null;
  const cushion = margin?.cushionPct ?? 0;
  const bufferPct = account.netLiq > 0 ? (account.excessLiquidity / account.netLiq) * 100 : null;

  return (
    <PfCard
      pad={false}
      title="Margin & Leverage"
      right={
        <span className="flex items-center gap-1.5 text-[10px] text-terminal-dim">
          <ProvenanceBadge source={source} />
          <Scale className="h-3.5 w-3.5 text-terminal-dim" />
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
            style={{
              background: stressed ? `${C.breach}24` : `${C.ok}24`,
              color: stressed ? C.breach : C.ok,
            }}
          >
            {stressed ? "Stressed" : "Healthy"}
          </span>
        </span>
      }
    >
      <div className="px-4 pb-3 pt-1">
        <div className="flex items-center justify-center">
          <MarginGauge health={cushion} />
        </div>
        <div className="grid grid-cols-3 gap-x-2 gap-y-3 text-center">
          <Stat
            label="Margin loan"
            value={loan > 0 ? fmtUsd(loan) : "none"}
            color={loan > 0 ? "text-pf-caution" : undefined}
          />
          <Stat
            label="Avail. funds"
            value={fmtSignedUsd(account.availableFunds)}
            color={stressed ? "text-pf-loss" : undefined}
          />
          <Stat label="Excess liq." value={fmtUsd(account.excessLiquidity)} />
          <Stat label="Maint. margin" value={fmtUsd(account.maintMargin)} />
          <Stat label="Cushion" value={`${(cushion * 100).toFixed(1)}%`} />
          <Stat
            label="Gross leverage"
            value={leverage == null ? "—" : `${leverage.toFixed(2)}×`}
          />
        </div>
        <p className="mt-3 border-t border-white/[0.08] pt-2 text-[10px] leading-snug text-terminal-dim">
          NAV buffer to a margin call ≈ excess liquidity {fmtUsd(account.excessLiquidity)}
          {bufferPct != null && ` (${bufferPct.toFixed(1)}% of NAV)`}. Gross leverage is
          Σ|position value| / NAV across all legs.
        </p>
      </div>
    </PfCard>
  );
}
