"use client";

import { TerminalPanel, TerminalDivider, TerminalBadge } from "./panel";

interface WatchlistItem {
  ticker: string;
  price: number | null;
  changePct: number | null;
  addedAt: string;
}

interface WatchlistPanelProps {
  items: WatchlistItem[];
  loading: boolean;
  onRefresh: () => void;
  onAddTicker: (ticker: string) => void;
}

export function WatchlistPanel({
  items,
  loading,
  onRefresh,
  onAddTicker,
}: WatchlistPanelProps) {
  return (
    <TerminalPanel
      title="Watchlist"
      tag="PORTFOLIO"
      headerRight={
        <button
          onClick={onRefresh}
          className="text-[10px] text-terminal-amber hover:text-terminal-text"
        >
          REFRESH
        </button>
      }
    >
      {/* Header */}
      <div className="flex items-center justify-between py-[1px] text-[10px] text-terminal-dim">
        <span className="w-14">SYM</span>
        <span className="w-20 text-right">PRICE</span>
        <span className="w-16 text-right">CHG%</span>
        <span className="w-14 text-right">STATUS</span>
      </div>
      <TerminalDivider />

      {loading ? (
        <div className="flex h-16 items-center justify-center text-terminal-dim">
          Fetching prices...
        </div>
      ) : items.length === 0 ? (
        <div className="flex h-16 flex-col items-center justify-center gap-1 text-terminal-dim">
          <span>No tickers tracked.</span>
          <span className="text-[10px]">
            Type <span className="text-terminal-amber">WATCH AAPL</span> to add
          </span>
        </div>
      ) : (
        <div>
          {items.map((item) => {
            const isPositive = (item.changePct ?? 0) >= 0;
            return (
              <div
                key={item.ticker}
                className="flex items-center justify-between py-[2px] hover:bg-terminal-border/30"
              >
                <span className="w-14 text-terminal-amber font-semibold">
                  {item.ticker}
                </span>
                <span className="w-20 text-right text-terminal-text">
                  {item.price !== null
                    ? `$${item.price.toFixed(2)}`
                    : "---"}
                </span>
                <span
                  className={`w-16 text-right ${
                    item.changePct !== null
                      ? isPositive
                        ? "text-terminal-green"
                        : "text-terminal-red"
                      : "text-terminal-dim"
                  }`}
                >
                  {item.changePct !== null
                    ? `${isPositive ? "+" : ""}${item.changePct.toFixed(2)}%`
                    : "---"}
                </span>
                <span className="w-14 text-right">
                  {item.price !== null ? (
                    <TerminalBadge variant={isPositive ? "green" : "red"}>
                      {isPositive ? "▲" : "▼"}
                    </TerminalBadge>
                  ) : (
                    <TerminalBadge variant="default">LOADING</TerminalBadge>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <TerminalDivider />

      {/* Summary — only render the up/down split once prices are actually populated */}
      <div className="flex justify-between text-[10px]">
        <span className="text-terminal-dim">
          {items.length} ticker{items.length !== 1 ? "s" : ""} tracked
        </span>
        {items.some((i) => i.price !== null) && (
          <span className="text-terminal-dim">
            {items.filter((i) => (i.changePct ?? 0) > 0).length}↑{" "}
            {items.filter((i) => (i.changePct ?? 0) < 0).length}↓
          </span>
        )}
      </div>
    </TerminalPanel>
  );
}
