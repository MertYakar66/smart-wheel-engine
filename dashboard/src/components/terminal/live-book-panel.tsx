"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import { fmtUsd, fmtUsdSigned, pnlToneClass } from "@/lib/cockpit-trust";
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

// legDte and the MONTHS lookup are removed: the adapter now supplies a
// server-computed dte (expiry − snapshot as_of, clamped >= 0) so clients
// never need to re-parse name strings against browser-local time. The field
// is consumed directly from leg.dte; a labeled fallback ("—") is kept only
// when the adapter did not populate the field (old engine version).

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
  // number that should never leave the screen). DTE comes from the adapter's
  // server-computed field (snapshot as_of - anchored, clamped >= 0) so it
  // stays consistent with the holdings table even when viewing a past snapshot.
  const optionLegs = legs
    .filter((l) => l.state !== "shares")
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
          <TerminalRow label="NAV" value={fmtUsd(summary.netLiq)} />
          <TerminalRow
            label="Day change"
            value={fmtUsdSigned(summary.dayChangeUsd)}
            valueColor={pnlToneClass(summary.dayChangeUsd)}
          />
          <TerminalRow label="Cash" value={fmtUsd(summary.cash)} />
          <TerminalRow
            label="Unrealized P&L"
            value={fmtUsdSigned(summary.unrealizedPnl)}
            valueColor={pnlToneClass(summary.unrealizedPnl)}
          />
          <TerminalRow
            label="Realized YTD"
            value={fmtUsdSigned(summary.realizedYtd)}
            valueColor={pnlToneClass(summary.realizedYtd)}
          />
          <TerminalRow
            label="Premium 30d"
            value={fmtUsd(summary.premium30d)}
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
            value={fmtUsd(summary.excessLiquidity)}
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
          <TerminalRow label="Maint margin" value={fmtUsd(summary.maintMargin)} />
          <TerminalRow
            label="Avail funds"
            value={fmtUsd(summary.availableFunds)}
            valueColor={pnlToneClass(summary.availableFunds)}
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
                  <span className={`tabular-nums ${pnlToneClass(leg.uPnl)}`}>
                    {fmtUsdSigned(leg.uPnl)}
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
