"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

/**
 * Top status strip. Everything rendered here is a real observed value:
 * VIX/term-structure from the engine regime read, NAV from the live IBKR
 * book, the data-frontier date from action=status, and the feed badge from
 * the actual stories count. The old hardcoded index tape (fake SPX/NDX
 * quotes with no DEMO label) is gone — no realtime quote feed exists.
 */
interface StatusBarProps {
  alertCount: number;
  ollamaStatus: "connected" | "disconnected" | "checking";
  /** Real stories count from /api/stories (null while loading). */
  storyCount: number | null;
  vix?: number;
  vix3m?: number | null;
  contango?: boolean | null;
  regime?: string;
  /** Freshest OHLCV date the engine serves (action=status data_frontier). */
  dataFrontier?: string | null;
  /** Live book NAV — only shown when the book source is actually live. */
  nav?: number | null;
}

export function StatusBar({
  alertCount,
  ollamaStatus,
  storyCount,
  vix,
  vix3m,
  contango,
  regime,
  dataFrontier,
  nav,
}: StatusBarProps) {
  const [time, setTime] = useState("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTime(
        now.toLocaleDateString("en-US", {
          month: "2-digit",
          day: "2-digit",
          year: "numeric",
        }) +
          " " +
          now.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
          })
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  const vixColor =
    vix == null || vix <= 0
      ? "text-terminal-dim"
      : vix >= 25
        ? "text-terminal-red"
        : vix >= 20
          ? "text-terminal-amber"
          : "text-terminal-green";

  const regimeColor =
    regime === "HIGH_VOL" || regime === "ELEVATED"
      ? "text-terminal-amber"
      : regime === "LOW_VOL"
        ? "text-terminal-green"
        : regime === "BEAR"
          ? "text-terminal-red"
          : "text-terminal-dim";

  return (
    <div className="flex h-8 shrink-0 items-center justify-between border-b border-terminal-border bg-terminal-header px-3 font-mono text-[11px]">
      {/* Left: Brand + real engine/book reads */}
      <div className="flex min-w-0 items-center gap-4">
        <Link
          href="/cockpit"
          title="Open the Decision Cockpit"
          className="shrink-0 font-bold tracking-wider text-terminal-amber hover:text-terminal-amber/80"
        >
          ■ YAKAR TERMINAL
        </Link>
        <div className="flex min-w-0 items-center gap-3">
          {/* VIX is always visible — the single most critical number for options */}
          {vix != null && vix > 0 && (
            <span className="flex shrink-0 items-center gap-1">
              <span className="text-terminal-dim">VIX</span>
              <span className={vixColor}>{vix.toFixed(2)}</span>
              {vix >= 20 && <span className={vixColor}>⚡</span>}
            </span>
          )}
          {vix3m != null && vix3m > 0 && (
            <span className="flex shrink-0 items-center gap-1">
              <span className="text-terminal-dim">3M</span>
              <span className="text-terminal-text">{vix3m.toFixed(2)}</span>
              {contango != null && (
                <span
                  className={
                    contango ? "text-terminal-green" : "text-terminal-red"
                  }
                >
                  {contango ? "CONT" : "BACKWD"}
                </span>
              )}
            </span>
          )}
          {regime && regime !== "---" && (
            <span
              className={`shrink-0 border border-current px-1 py-[1px] text-[9px] font-bold ${regimeColor}`}
            >
              {regime}
            </span>
          )}
          {nav != null && (
            <span className="flex shrink-0 items-center gap-1">
              <span className="text-terminal-dim">NAV</span>
              <span className="text-terminal-text tabular-nums">
                ${nav.toLocaleString("en-US", { maximumFractionDigits: 0 })}
              </span>
            </span>
          )}
          {dataFrontier && (
            <span className="hidden shrink-0 items-center gap-1 sm:flex">
              <span className="text-terminal-dim">EOD</span>
              <span className="text-terminal-dim">{dataFrontier}</span>
            </span>
          )}
        </div>
      </div>

      {/* Right: Status indicators */}
      <div className="flex shrink-0 items-center gap-4">
        {/* Feed badge derives from real state: the actual stories count, not
            a hardcoded LIVE off a static provider list. */}
        <span className="flex items-center gap-1">
          <span className="text-terminal-dim">FEED:</span>
          {storyCount === null ? (
            <span className="text-terminal-dim">…</span>
          ) : storyCount > 0 ? (
            <span className="text-terminal-green">{storyCount}</span>
          ) : (
            <span className="text-terminal-amber">EMPTY</span>
          )}
        </span>
        <span className="flex items-center gap-1">
          <span className="text-terminal-dim">AI:</span>
          <span
            className={
              ollamaStatus === "connected"
                ? "text-terminal-green"
                : ollamaStatus === "checking"
                  ? "text-terminal-amber"
                  : "text-terminal-red"
            }
          >
            {ollamaStatus === "connected"
              ? "ONLINE"
              : ollamaStatus === "checking"
                ? "…"
                : "OFFLINE"}
          </span>
        </span>
        {alertCount > 0 && (
          <span className="flex items-center gap-1">
            <span className="text-terminal-amber">▲ {alertCount}</span>
          </span>
        )}
        <span className="text-terminal-dim">{time}</span>
      </div>
    </div>
  );
}
