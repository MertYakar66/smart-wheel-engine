"use client";

// Shared presentational primitives for the portfolio viewer. Slate-teal
// direction: flat hairline cards, optional 2px accent top-rule on KPIs,
// tabular/mono numerals, fixed semantic colors.

import { cn } from "@/lib/utils";
import { WHEEL_LABEL, type Period, type WheelState, PERIODS } from "./mock";

export function PfCard({
  children,
  className,
  accentTop = false,
  title,
  right,
  pad = true,
}: {
  children: React.ReactNode;
  className?: string;
  accentTop?: boolean;
  title?: string;
  right?: React.ReactNode;
  pad?: boolean;
}) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-white/[0.08] bg-pf-panel",
        className
      )}
    >
      {accentTop && (
        <span className="absolute inset-x-0 top-0 h-0.5 bg-pf-accent/80" />
      )}
      {(title || right) && (
        <div className="flex items-center justify-between gap-2 px-4 pt-3">
          {title && (
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-terminal-dim">
              {title}
            </h3>
          )}
          {right}
        </div>
      )}
      <div className={cn(pad && "p-4", title && pad && "pt-2")}>{children}</div>
    </div>
  );
}

/** Lifecycle pill — the wheel state of a position, color-coded + dotted. */
export function WheelBadge({ state, compact }: { state: WheelState; compact?: boolean }) {
  const m = WHEEL_LABEL[state];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide"
      style={{ backgroundColor: `${m.color}22`, color: m.color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: m.color }} />
      {compact ? m.short : m.full}
    </span>
  );
}

/** Segmented period control (drives Total-Return KPI + the equity chart). */
export function PeriodToggle({
  value,
  onChange,
  size = "sm",
}: {
  value: Period;
  onChange: (p: Period) => void;
  size?: "sm" | "xs";
}) {
  return (
    <div className="inline-flex flex-wrap items-center gap-0.5 rounded-lg border border-white/[0.08] bg-pf-bg p-0.5">
      {PERIODS.map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          className={cn(
            "rounded-md font-medium tabular-nums transition-colors",
            size === "xs" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-1 text-[11px]",
            value === p
              ? "bg-pf-accent/15 text-pf-accent"
              : "text-terminal-dim hover:text-terminal-text"
          )}
        >
          {p}
        </button>
      ))}
    </div>
  );
}

/** Tailwind text class for a signed number (gain / loss / neutral). */
export function pnlColor(v: number): string {
  if (v > 0) return "text-pf-gain";
  if (v < 0) return "text-pf-loss";
  return "text-pf-neutral";
}

export function fmtSignedUsd(v: number, decimals = 0): string {
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

export function fmtSignedPct(frac: number, decimals = 2): string {
  const sign = frac > 0 ? "+" : frac < 0 ? "−" : "";
  return `${sign}${Math.abs(frac * 100).toFixed(decimals)}%`;
}

export type { Period };
