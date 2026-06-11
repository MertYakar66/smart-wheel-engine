"use client";

import { AlertTriangle, Check, ShieldAlert } from "lucide-react";
import {
  CONCENTRATION as MOCK_CONCENTRATION,
  GATES as MOCK_GATES,
  IV_ASSUMPTION as MOCK_IV,
  SECTOR_CAP as MOCK_SECTOR_CAP,
  SECTOR_EXPOSURE as MOCK_SECTOR_EXPOSURE,
  SINGLE_NAME as MOCK_SINGLE_NAME,
  SINGLE_NAME_CAP as MOCK_SINGLE_NAME_CAP,
  WHEEL_LABEL,
  type ConcentrationRow,
  type Gates,
} from "./mock";
import { PfCard, ProvenanceBadge, type SliceSource } from "./parts";

const C = { ok: "#34d399", caution: "#f5a524", breach: "#f2495e", dim: "#7c8696" };
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
  caption,
}: {
  title: string;
  cap: number;
  rows: { label: string; pct: number; tag?: string; tagColor?: string }[];
  caption?: string;
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
              {r.tag && (
                <span
                  className="w-16 shrink-0 text-[9px] font-semibold uppercase tracking-wide"
                  style={{ color: r.tagColor ?? C.dim }}
                >
                  {r.tag}
                </span>
              )}
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
      {caption && <p className="mt-2 text-[10px] leading-snug text-terminal-dim">{caption}</p>}
    </div>
  );
}

/** Pass/fail/skip pill for an engine gate verdict. A gate that returned
 * passed=true with reason "missing_data" did NOT evaluate — render an honest
 * SKIPPED, never a green PASS on absent evidence. */
function GatePill({ passed, reason }: { passed: boolean; reason: string | null }) {
  const skipped = passed && reason === "missing_data";
  const color = skipped ? C.dim : passed ? C.ok : C.breach;
  return (
    <span
      className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide"
      style={{ background: `${color}24`, color }}
    >
      {skipped ? "skipped" : passed ? "pass" : "fail"}
    </span>
  );
}

function GateRow({
  tag,
  label,
  detail,
  passed,
  reason,
}: {
  tag: string;
  label: string;
  detail?: string;
  passed: boolean;
  reason: string | null;
}) {
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="w-7 shrink-0 font-mono text-[10px] text-terminal-dim">{tag}</span>
      <span className="min-w-0 flex-1 truncate text-[11px] text-terminal-text">
        {label}
        {detail && <span className="ml-1.5 tabular-nums text-terminal-dim">{detail}</span>}
      </span>
      {/* The engine's structured reason, verbatim (single_name_breach /
          sector_cap_breach / missing_data) — the audit-trail vocabulary. */}
      {reason && (
        <span className="hidden font-mono text-[9px] text-terminal-dim md:inline">{reason}</span>
      )}
      <GatePill passed={passed} reason={reason} />
    </div>
  );
}

export function RiskRadar({
  singleName = MOCK_SINGLE_NAME,
  concentration = MOCK_CONCENTRATION,
  sectorExposure = MOCK_SECTOR_EXPOSURE,
  caps = { singleName: MOCK_SINGLE_NAME_CAP, sector: MOCK_SECTOR_CAP },
  gates = MOCK_GATES,
  ivAssumption = MOCK_IV,
  source,
}: {
  singleName?: { sym: string; pct: number }[];
  concentration?: ConcentrationRow[] | null;
  sectorExposure?: { name: string; pct: number }[];
  caps?: { singleName: number; sector: number };
  gates?: Gates | null;
  ivAssumption?: number | null;
  source?: SliceSource;
}) {
  return (
    <PfCard
      pad={false}
      title="Risk Radar"
      right={
        <span className="flex items-center gap-1.5 text-[10px] text-terminal-dim">
          <ProvenanceBadge source={source} />
          <ShieldAlert className="h-3.5 w-3.5" style={{ color: C.breach }} />
          concentration · engine gates
        </span>
      }
    >
      <div className="px-4 pb-3 pt-2">
        {/* All-exposure rows: stock value for CC/assigned names + short-put
            notional for CSPs — keeps CC stock at >100% NAV from hiding. The
            10% line is the R10 reference; the gate itself only caps
            short-option notional (covered-call stock never "breaches"). */}
        {concentration && concentration.length > 0 && (
          <Rule
            title="Concentration · all exposure"
            cap={caps.singleName}
            rows={concentration.map((c) => ({
              label: c.sym,
              pct: c.pct,
              tag: WHEEL_LABEL[c.state]?.short ?? c.state,
              tagColor: WHEEL_LABEL[c.state]?.color,
            }))}
            caption="Stock market value (CC/assigned) + short-put notional (CSP), % of NAV. Cap line is the R10 reference — the gate itself bounds short-option notional only."
          />
        )}
        <Rule
          title="Single-name · R10 put notional"
          cap={caps.singleName}
          rows={singleName.map((s) => ({ label: s.sym, pct: s.pct }))}
        />
        <Rule
          title="Sector"
          cap={caps.sector}
          rows={sectorExposure.map((s) => ({ label: s.name.slice(0, 6), pct: s.pct }))}
        />

        {gates && (
          <div className="border-t border-white/[0.08] pt-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium text-terminal-text">Risk gates (engine)</span>
              <span className="text-[10px] text-terminal-dim">held book · portfolio_risk_gates</span>
            </div>
            {gates.singleName.map((g) => (
              <GateRow
                key={`r10-${g.sym}`}
                tag="R10"
                label={g.sym}
                detail={`${g.pctNav.toFixed(1)}% NAV all short legs`}
                passed={g.passed}
                reason={g.reason}
              />
            ))}
            {gates.sector.map((g) => (
              <GateRow
                key={`r9-${g.sector}`}
                tag="R9"
                label={g.sector}
                detail={`${g.pctNav.toFixed(1)}% NAV`}
                passed={g.passed}
                reason={g.reason}
              />
            ))}
            <GateRow
              tag="R7"
              label="Portfolio VaR 95"
              detail={gates.var.varPct != null ? `${(gates.var.varPct * 100).toFixed(1)}% NAV` : "—"}
              passed={gates.var.passed}
              reason={gates.var.reason}
            />
            <GateRow
              tag="R8"
              label={gates.stress.scenario ?? "Stress scenario"}
              detail={
                gates.stress.drawdownPct != null
                  ? `drawdown ${(gates.stress.drawdownPct * 100).toFixed(1)}% NAV`
                  : "—"
              }
              passed={gates.stress.passed}
              reason={gates.stress.reason}
            />
            <p className="mt-1.5 text-[10px] leading-snug text-terminal-dim">
              R10 sums every short-option leg (put + call), so a strangle reads higher here than
              in the put-notional meter — both are correct.
              {ivAssumption != null &&
                ` VaR/stress assume IV ${ivAssumption.toFixed(2)} (no live greeks in the snapshot).`}
            </p>
          </div>
        )}
      </div>
    </PfCard>
  );
}
