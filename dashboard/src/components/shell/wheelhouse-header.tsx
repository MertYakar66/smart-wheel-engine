"use client";

// Shared "Wheelhouse" chrome — the sticky branding header + cross-page tab
// nav, lifted verbatim from the /portfolio reference design (design D26) so
// the Cockpit and Terminal surfaces share one visual language.
//
// Presentational only: no engine logic, no data fetching, no EV authority.
// The decision-layer trio (ev_engine / wheel_runner / candidate_dossier) is
// never imported here — this is read-only page chrome.

import Link from "next/link";

import { cn } from "@/lib/utils";

/** The top-level surfaces, in canonical order. "News" lives in the (main)
 *  route group at /top. Order matches the cross-page tab row everywhere. */
export const CROSS_PAGE_NAV: { label: string; href: string }[] = [
  { label: "Cockpit", href: "/cockpit" },
  { label: "Portfolio", href: "/portfolio" },
  { label: "Terminal", href: "/terminal" },
  { label: "News", href: "/top" },
];

/**
 * The Wheelhouse top bar: an accent-dot wordmark + page label, an optional
 * right-aligned key stat, an optional status chip, and an optional subnav row
 * (cross-page tabs or in-page section tabs) rendered beneath.
 */
export function WheelhouseHeader({
  page,
  right,
  status,
  children,
  maxW = "max-w-[1400px]",
}: {
  page: string;
  right?: React.ReactNode;
  status?: React.ReactNode;
  children?: React.ReactNode;
  maxW?: string;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.08] bg-pf-bg/90 font-sans backdrop-blur">
      <div className={cn("mx-auto flex h-14 flex-wrap items-center gap-x-6 gap-y-1 px-5", maxW)}>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-pf-accent shadow-[0_0_8px_var(--color-pf-accent)]" />
          <span className="font-semibold tracking-tight text-terminal-text">Wheelhouse</span>
          <span className="text-[11px] text-terminal-dim">· {page}</span>
        </div>

        {right && <div className="ml-auto flex items-baseline gap-3">{right}</div>}

        {status && (
          <div className="flex items-center gap-1.5 text-[11px] text-terminal-dim">{status}</div>
        )}
      </div>

      {children && (
        <div className={cn("mx-auto flex items-center gap-1 px-5", maxW)}>{children}</div>
      )}
    </header>
  );
}

/**
 * Cross-page tab nav, styled exactly like the portfolio subnav tabs. Pass the
 * label of the current page as `active`; an optional `trailing` node is pushed
 * to the far right (e.g. a secondary action or status).
 */
export function CrossPageNav({
  active,
  trailing,
}: {
  active: string;
  trailing?: React.ReactNode;
}) {
  return (
    <>
      {CROSS_PAGE_NAV.map((n) => {
        const isActive = n.label === active;
        return (
          <Link
            key={n.href}
            href={n.href}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "-mb-px border-b-2 px-2.5 py-2 text-xs font-medium transition-colors focus-visible:rounded focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-pf-accent/50",
              isActive
                ? "border-pf-accent text-terminal-text"
                : "border-transparent text-terminal-dim hover:text-terminal-text"
            )}
          >
            {n.label}
          </Link>
        );
      })}
      {trailing && <div className="ml-auto">{trailing}</div>}
    </>
  );
}
