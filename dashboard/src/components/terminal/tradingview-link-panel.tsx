"use client";

import { ExternalLink } from "lucide-react";

interface TradingViewLinkRowProps {
  symbol: string;
  onClose: () => void;
}

/**
 * Compact TradingView handoff strip. Charting lives in TradingView (via the
 * TradingView MCP); the terminal no longer renders its own charts. This used
 * to occupy the dominant grid cell as a full panel — now it's a slim row so
 * the freed space can show engine substance (analysis + dealer positioning).
 */
export function TradingViewLinkRow({ symbol, onClose }: TradingViewLinkRowProps) {
  const sym = symbol.toUpperCase();
  const tvUrl = `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(
    sym
  )}`;

  return (
    <div className="flex h-9 shrink-0 items-center justify-between border border-terminal-border bg-terminal-header px-2">
      <div className="flex min-w-0 items-center gap-2">
        <span className="text-[13px] font-bold tracking-tight text-terminal-amber">
          {sym}
        </span>
        <a
          href={tvUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 border border-terminal-border px-2 py-0.5 text-[10px] font-semibold uppercase text-terminal-green transition-colors hover:bg-terminal-border/40"
        >
          Chart in TradingView
          <ExternalLink className="h-3 w-3" />
        </a>
        <span className="hidden text-[10px] text-terminal-dim md:inline">
          charting is handled in TradingView — panels below are engine reads
        </span>
      </div>
      <button
        onClick={onClose}
        className="shrink-0 text-[10px] uppercase text-terminal-dim transition-colors hover:text-terminal-text"
      >
        ✕ Close
      </button>
    </div>
  );
}
