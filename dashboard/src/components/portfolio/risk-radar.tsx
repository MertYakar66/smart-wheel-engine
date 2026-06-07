"use client";

import { AlertTriangle, Check, ShieldAlert } from "lucide-react";
import { fmtUsd } from "@/lib/cockpit-trust";
import {
  ACCOUNT as MOCK_ACCOUNT,
  SECTOR_CAP as MOCK_SECTOR_CAP,
  SECTOR_EXPOSURE as MOCK_SECTOR_EXPOSURE,
  SINGLE_NAME as MOCK_SINGLE_NAME,
  SINGLE_NAME_CAP as MOCK_SINGLE_NAME_CAP,
} from "./mock";
import { PfCard, ProvenanceBadge, type SliceSource } from "./parts";

const C = { ok: "#34d399", caution: "#f5a524", breach: "#f2495e" };
type Status = keyof typeof C;

function statusFor(pct: number, cap: number): Status {
  if (pct > cap) return "breach";
  if (pct > cap * 0.8) return "caution";
  return "ok";
}

/** A capped bar with a marker line at the cap threshold. */
function CapBar({ pct, cap }: { pct: number; cap: number }) {
  const status = statusFor(pct, cap);
  const fill = Math.min(pct, 100);
  const capMark = Math.min(cap, 100);
  return (
    <div className="relative h-1.5 flex-1 rounded-full bg-white/[0.07]">
      <div
        className="h-full rounded-full"
        style={{ width: `${fill}%`, background: C[status] }}
      />
      <span
        className="absolute -top-0.5 -bottom-0.5 w-px bg-white/45"
        style={{ left: `${capMark}%` }}
      />
    </div>
  );
}

function Rule({
  title,
  cap,
  rows,
}: {
  title: string;
  cap: number;
  rows: { label: string; pct: number }[];
}) {
  const breaches = rows.filter((r) => r.pct > cap).length;
  return (
    <div className="border-t border-white/[0.08] py-3 first:border-t-0 first:pt-0">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {breaches > 0 ? (
            <AlertTriangle className="h-3.5 w-3.5" style={{ color: C.breach }} />
          ) : (
            <Check className="h-3.5 w-3.5" style={{ color: C.ok }} />
          )}
          <span className="text-xs font-medium text-terminal-text">{title}</span>
          <span className="text-[10px] text-terminal-dim">cap {cap}% NAV</span>
        </div>
        <span
          className="rounded px-1.5 py-0.5 text-[10px] font-medium"
          style={{
            background: breaches > 0 ? `${C.breach}24` : `${C.ok}24`,
            color: breaches > 0 ? C.breach : C.ok,
          }}
        >
          {breaches} breach{breaches === 1 ? "" : "es"}
        </span>
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const status = statusFor(r.pct, cap);
          return (
            <div key={r.label} className="flex items-center gap-3">
              <span className="w-14 shrink-0 text-[11px] font-medium text-terminal-text">
                {r.label}
              </span>
              <CapBar pct={r.pct} cap={cap} />
              <span
                className="w-20 shrink-0 text-right text-[11px] tabular-nums"
                style={{ color: C[status] }}
              >
                {r.pct}% / {cap}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

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
  const needleDeg = 180 * (1 - health);
  const needleRad = (needleDeg * Math.PI) / 180;
  const nx = cx + (r - 10) * Math.cos(needleRad);
  const ny = cy - (r - 10) * Math.sin(needleRad);

  return (
    <svg viewBox="0 0 200 112" className="w-full" style={{ maxWidth: 220 }}>
      <path d={arc(0, 1)} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={9} strokeLinecap="round" />
      <path d={arc(0, 0.34)} fill="none" stroke={C.breach} strokeWidth={9} strokeLinecap="round" />
      <path d={arc(0.34, 0.66)} fill="none" stroke={C.caution} strokeWidth={9} />
      <path d={arc(0.66, 1)} fill="none" stroke={C.ok} strokeWidth={9} strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={nx.toFixed(2)} y2={ny.toFixed(2)} stroke="#e2e8f0" strokeWidth={2.5} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={4} fill="#e2e8f0" />
    </svg>
  );
}

export function RiskRadar({
  account = MOCK_ACCOUNT,
  singleName = MOCK_SINGLE_NAME,
  sectorExposure = MOCK_SECTOR_EXPOSURE,
  caps = { singleName: MOCK_SINGLE_NAME_CAP, sector: MOCK_SECTOR_CAP },
  marginHealth = 0.12,
  source,
}: {
  account?: typeof MOCK_ACCOUNT;
  singleName?: { sym: string; pct: number }[];
  sectorExposure?: { name: string; pct: number }[];
  caps?: { singleName: number; sector: number };
  marginHealth?: number;
  source?: SliceSource;
}) {
  const stressed = account.availableFunds < 0;
  return (
    <PfCard
      pad={false}
      title="Risk Radar"
      right={
        <span className="flex items-center gap-1.5 text-[10px] text-terminal-dim">
          <ProvenanceBadge source={source} />
          <ShieldAlert className="h-3.5 w-3.5" style={{ color: C.breach }} />
          concentration · margin
        </span>
      }
    >
      <div className="px-4 pb-2 pt-2">
        <Rule title="Single-name" cap={caps.singleName} rows={singleName.map((s) => ({ label: s.sym, pct: s.pct }))} />
        <Rule title="Sector" cap={caps.sector} rows={sectorExposure.map((s) => ({ label: s.name.slice(0, 6), pct: s.pct }))} />

        <div className="border-t border-white/[0.08] pt-3">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-terminal-text">Margin health</span>
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
              style={{
                background: stressed ? `${C.breach}24` : `${C.ok}24`,
                color: stressed ? C.breach : C.ok,
              }}
            >
              {stressed ? "Stressed" : "Healthy"}
            </span>
          </div>
          <div className="flex items-center justify-center">
            <MarginGauge health={marginHealth} />
          </div>
          <div className="mt-1 grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] uppercase tracking-wide text-terminal-dim">Avail. funds</div>
              <div className="text-xs font-semibold tabular-nums text-pf-loss">
                {fmtUsd(account.availableFunds, { signed: true })}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wide text-terminal-dim">Excess liq.</div>
              <div className="text-xs font-semibold tabular-nums text-terminal-text">
                {fmtUsd(account.excessLiquidity)}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wide text-terminal-dim">Maint. margin</div>
              <div className="text-xs font-semibold tabular-nums text-terminal-text">
                {fmtUsd(account.maintMargin)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </PfCard>
  );
}
