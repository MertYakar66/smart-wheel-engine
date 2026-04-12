"use client";

import { useState, useEffect } from "react";

interface StatusBarProps {
  indices: { symbol: string; price: number; changePct: number }[];
  alertCount: number;
  ollamaStatus: "connected" | "disconnected" | "checking";
  agentStatus: "online" | "offline";
  retrievalProviders?: string[];
  vix?: number;
  regime?: string;
}

export function StatusBar({
  indices,
  alertCount,
  ollamaStatus,
  agentStatus,
  retrievalProviders,
  vix,
  regime,
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
      {/* Left: Brand + Ticker tape */}
      <div className="flex min-w-0 items-center gap-4">
        <span className="shrink-0 font-bold tracking-wider text-terminal-amber">
          ■ YAKAR TERMINAL
        </span>
        <div className="flex min-w-0 items-center gap-3">
          {indices.map((idx) => (
            <span key={idx.symbol} className="flex shrink items-center gap-1">
              <span className="text-terminal-dim">{idx.symbol}</span>
              <span className="text-terminal-text">
                {idx.price.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </span>
              <span
                className={
                  idx.changePct >= 0
                    ? "text-terminal-green"
                    : "text-terminal-red"
                }
              >
                {idx.changePct >= 0 ? "+" : ""}
                {idx.changePct.toFixed(2)}%
              </span>
            </span>
          ))}
          {/* VIX is always visible — the single most critical number for options */}
          {vix != null && vix > 0 && (
            <span className="flex shrink-0 items-center gap-1">
              <span className="text-terminal-dim">VIX</span>
              <span className={vixColor}>{vix.toFixed(2)}</span>
              {vix >= 20 && <span className={vixColor}>⚡</span>}
            </span>
          )}
          {regime && regime !== "---" && (
            <span
              className={`shrink-0 border border-current px-1 py-[1px] text-[9px] font-bold ${regimeColor}`}
            >
              {regime}
            </span>
          )}
        </div>
      </div>

      {/* Right: Status indicators */}
      <div className="flex shrink-0 items-center gap-4">
        {retrievalProviders && retrievalProviders.length > 0 && (
          <span className="flex items-center gap-1">
            <span className="text-terminal-dim">FEED:</span>
            <span className="text-terminal-green">LIVE</span>
          </span>
        )}
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
        <span className="flex items-center gap-1">
          <span className="text-terminal-dim">AGENT:</span>
          <span
            className={
              agentStatus === "online"
                ? "text-terminal-green"
                : "text-terminal-red"
            }
          >
            {agentStatus === "online" ? "●" : "○"}
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
