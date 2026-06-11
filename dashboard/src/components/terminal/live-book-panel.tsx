"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import type { LiveBookSummary, LiveBookLeg } from "@/types";

/**
 * Read-only live IBKR book panel (replaces the placeholder AI-agent panel).
 * Pure display over /api/portfolio/{summary,positions} — no order surface,
 * no EV authority. Absent fields render "—"; an unreachable book renders an
 * explicit unavailable state, never stale zeros.
 */

interface LiveBookPanelProps {
  summary: LiveBookSummary | null;
  legs: LiveBookLeg[];
  loading: boolean;
  error: string | null;
  /** One-shot highlight when the command line targets this panel. */
  flash?: boolean;
}

const MONTHS: Record<string, number> = {
  JAN: 0, FEB: 1, MAR: 2, APR: 3, MAY: 4, JUN: 5,
  JUL: 6, AUG: 7, SEP: 8, OCT: 9, NOV: 10, DEC: 11,
};

/** Parse "MU 12JUN26 970 P"-style leg names → days to expiry, or null. */
function legDte(name: string): number | null {
  const m = name.match(/\b(\d{1,2})([A-Z]{3})(\d{2})\b/);
  if (!m) return null;
  const month = MONTHS[m[2]];
  if (month === undefined) return null;
  const expiry = new Date(2000 + parseInt(m[3], 10), month, parseInt(m[1], 10));
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.round((expiry.getTime() - today.getTime()) / 86_400_000);
}

function usd(v: number | null, digits = 0): string {
  if (v === null) return "—";
  return `$${v.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

function signedUsd(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : "−"}$${Math.abs(v).toLocaleString("en-US", {
    maximumFractionDigits: 0,
  })}`;
}

function pnlColor(v: number | null): string {
  if (v === null) return "text-terminal-dim";
  return v >= 0 ? "text-terminal-green" : "text-terminal-red";
}

const LEG_STATE_LABEL: Record<string, string> = {
  short_put: "SP",
  short_call: "SC",
  long_call: "LC",
  long_put: "LP",
};

export function LiveBookPanel({
  summary,
  legs,
  loading,
  error,
  flash,
}: LiveBookPanelProps) {
  const isLive = summary?.source === "live";

  // Option legs only, soonest expiry first (assignment proximity is the
  // number that should never leave the screen).
  const optionLegs = legs
    .filter((l) => l.state !== "shares")
    .map((l) => ({ ...l, dte: legDte(l.name) }))
    .sort((a, b) => (a.dte ?? Infinity) - (b.dte ?? Infinity));
  const soonestDte = optionLegs.length > 0 ? optionLegs[0].dte : null;

  // Margin cushion: excess liquidity vs maintenance margin.
  const cushionPct =
    summary?.excessLiquidity !== null &&
    summary?.excessLiquidity !== undefined &&
    summary?.maintMargin
      ? (summary.excessLiquidity / summary.maintMargin) * 100
      : null;

  return (
    <TerminalPanel
      title="Live Book"
      tag="IBKR"
      flash={flash}
      headerRight={
        loading ? (
          <TerminalBadge variant="default">…</TerminalBadge>
        ) : error || !summary ? (
          <TerminalBadge variant="amber">UNAVAILABLE</TerminalBadge>
        ) : isLive ? (
          <TerminalBadge variant="green">LIVE</TerminalBadge>
        ) : (
          // Anything not explicitly live gets labeled with its real source.
          <TerminalBadge variant="amber">
            {(summary.source || "unknown").toUpperCase()}
          </TerminalBadge>
        )
      }
    >
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Loading book…
        </div>
      ) : error || !summary ? (
        <div className="flex h-full flex-col items-center justify-center gap-1 text-terminal-dim">
          <span>Live book unavailable.</span>
          <span className="text-[10px]">
            {error || "Portfolio endpoints returned no data"}
          </span>
        </div>
      ) : (
        <>
          {/* Account */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ ACCOUNT
          </div>
          <TerminalRow label="NAV" value={usd(summary.netLiq)} />
          <TerminalRow
            label="Day change"
            value={
              summary.dayChangeUsd !== null
                ? signedUsd(summary.dayChangeUsd)
                : "—"
            }
            valueColor={pnlColor(summary.dayChangeUsd)}
          />
          <TerminalRow label="Cash" value={usd(summary.cash)} />
          <TerminalRow
            label="Unrealized P&L"
            value={signedUsd(summary.unrealizedPnl)}
            valueColor={pnlColor(summary.unrealizedPnl)}
          />
          <TerminalRow
            label="Realized YTD"
            value={signedUsd(summary.realizedYtd)}
            valueColor={pnlColor(summary.realizedYtd)}
          />
          <TerminalRow
            label="Premium 30d"
            value={usd(summary.premium30d)}
            valueColor="text-terminal-green"
          />
          <TerminalRow
            label="Win rate"
            value={
              summary.winRate !== null
                ? `${(summary.winRate * 100).toFixed(1)}%`
                : "—"
            }
          />

          <TerminalDivider />

          {/* Margin cushion */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ MARGIN
          </div>
          <TerminalRow
            label="Excess liquidity"
            value={usd(summary.excessLiquidity)}
            valueColor={
              cushionPct === null
                ? "text-terminal-dim"
                : cushionPct < 10
                  ? "text-terminal-red"
                  : cushionPct < 20
                    ? "text-terminal-amber"
                    : "text-terminal-green"
            }
          />
          <TerminalRow label="Maint margin" value={usd(summary.maintMargin)} />
          <TerminalRow
            label="Avail funds"
            value={usd(summary.availableFunds)}
            valueColor={pnlColor(summary.availableFunds)}
          />
          {cushionPct !== null && cushionPct < 10 && (
            <div className="py-[2px] text-[10px] text-terminal-red">
              ⚠ cushion {cushionPct.toFixed(1)}% of maint margin
            </div>
          )}

          <TerminalDivider />

          {/* Option legs by expiry */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ OPTION LEGS ({optionLegs.length})
          </div>
          {optionLegs.length === 0 ? (
            <div className="py-1 text-[10px] italic text-terminal-dim">
              No option legs in the book
            </div>
          ) : (
            optionLegs.map((leg, i) => (
              <div
                key={`${leg.name}-${i}`}
                className="flex items-center justify-between gap-1 py-[2px] hover:bg-terminal-border/30"
              >
                <div className="flex min-w-0 items-center gap-1.5">
                  <TerminalBadge
                    variant={leg.state.startsWith("short") ? "blue" : "default"}
                  >
                    {LEG_STATE_LABEL[leg.state] ?? leg.state.toUpperCase()}
                  </TerminalBadge>
                  <span className="truncate text-terminal-text">{leg.name}</span>
                  {leg.breach && (
                    <TerminalBadge variant="red">BREACH</TerminalBadge>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className={`tabular-nums ${pnlColor(leg.uPnl)}`}>
                    {signedUsd(leg.uPnl)}
                  </span>
                  {leg.dte !== null ? (
                    <TerminalBadge
                      variant={
                        leg.dte === soonestDte
                          ? leg.dte <= 3
                            ? "red"
                            : "amber"
                          : "default"
                      }
                    >
                      {leg.dte}d
                    </TerminalBadge>
                  ) : (
                    <span className="text-[10px] text-terminal-dim">—</span>
                  )}
                </div>
              </div>
            ))
          )}
        </>
      )}
    </TerminalPanel>
  );
}
