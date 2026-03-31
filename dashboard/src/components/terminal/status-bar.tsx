"use client";

import { useState, useEffect } from "react";

interface StatusBarProps {
  indices: { symbol: string; price: number; changePct: number }[];
  alertCount: number;
  ollamaStatus: "connected" | "disconnected" | "checking";
  agentStatus: "online" | "offline";
  retrievalProviders?: string[];
}

export function StatusBar({
  indices,
  alertCount,
  ollamaStatus,
  agentStatus,
  retrievalProviders,
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

  return (
    <div className="flex h-8 shrink-0 items-center justify-between border-b border-terminal-border bg-terminal-header px-3 font-mono text-[11px]">
      {/* Left: Brand + Ticker tape */}
      <div className="flex items-center gap-4">
        <span className="font-bold tracking-wider text-terminal-amber">
          ■ YAKAR TERMINAL
        </span>
        <div className="flex items-center gap-3">
          {indices.map((idx) => (
            <span key={idx.symbol} className="flex items-center gap-1">
              <span className="text-terminal-dim">{idx.symbol}</span>
              <span className="text-terminal-text">
                {idx.price.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </span>
              <span
                className={
                  idx.changePct >= 0 ? "text-terminal-green" : "text-terminal-red"
                }
              >
                {idx.changePct >= 0 ? "+" : ""}
                {idx.changePct.toFixed(2)}%
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* Right: Status indicators */}
      <div className="flex items-center gap-4">
        {retrievalProviders && retrievalProviders.length > 0 && (
          <span className="flex items-center gap-1">
            <span className="text-terminal-dim">DATA:</span>
            <span className="text-terminal-green">
              {retrievalProviders.join("+")}
            </span>
          </span>
        )}
        <span className="flex items-center gap-1">
          <span className="text-terminal-dim">OLLAMA:</span>
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
              ? "●"
              : ollamaStatus === "checking"
              ? "◌"
              : "○"}
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
