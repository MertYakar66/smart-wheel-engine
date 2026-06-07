"use client";

import { ExternalLink } from "lucide-react";

import { TerminalPanel } from "./panel";

interface TradingViewLinkPanelProps {
  symbol: string;
  onClose: () => void;
}

/**
 * Replaces the old in-terminal ChartPanel. Charting now lives in TradingView
 * (navigated via the TradingView MCP), so the terminal no longer renders its
 * own price/RSI/ATR charts — selecting a ticker just hands off to TradingView
 * for that symbol. The OptionsPanel + Research panels stay alongside.
 */
export function TradingViewLinkPanel({
  symbol,
  onClose,
}: TradingViewLinkPanelProps) {
  const sym = symbol.toUpperCase();
  const tvUrl = `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(
    sym
  )}`;

  return (
    <TerminalPanel
      title={`${sym} — Chart`}
      tag="TRADINGVIEW"
      headerRight={
        <button
          onClick={onClose}
          className="text-[10px] uppercase text-terminal-dim transition-colors hover:text-terminal-text"
        >
          ✕ Close
        </button>
      }
    >
      <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center">
        <span className="text-[10px] uppercase tracking-wider text-terminal-dim">
          Charts live in TradingView
        </span>
        <span className="text-3xl font-bold tracking-tight text-terminal-amber">
          {sym}
        </span>
        <a
          href={tvUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 border border-terminal-border bg-terminal-header px-3 py-1.5 text-[11px] font-semibold uppercase text-terminal-green transition-colors hover:bg-terminal-border/40"
        >
          Open {sym} in TradingView
          <ExternalLink className="h-3 w-3" />
        </a>
        <p className="max-w-xs text-[10px] leading-relaxed text-terminal-dim">
          In-terminal charts were removed — charting is handled in TradingView
          (drive it via the TradingView MCP). This opens the symbol on
          tradingview.com.
        </p>
      </div>
    </TerminalPanel>
  );
}
