"use client";

import { TerminalPanel, TerminalDivider, TerminalBadge } from "./panel";
import type { MarketIndex } from "@/types";

interface MarketOverviewProps {
  indices: MarketIndex[];
  futures: MarketIndex[];
  commodities: MarketIndex[];
  loading: boolean;
}

function PriceRow({ item }: { item: MarketIndex }) {
  const isPositive = item.changePct >= 0;
  return (
    <div className="flex items-center justify-between py-[1px]">
      <span className="w-16 text-terminal-amber">{item.symbol}</span>
      <span className="w-24 text-right text-terminal-text">
        {item.price.toLocaleString("en-US", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
      </span>
      <span
        className={`w-16 text-right ${
          isPositive ? "text-terminal-green" : "text-terminal-red"
        }`}
      >
        {isPositive ? "+" : ""}
        {item.change.toFixed(2)}
      </span>
      <span
        className={`w-16 text-right ${
          isPositive ? "text-terminal-green" : "text-terminal-red"
        }`}
      >
        {isPositive ? "+" : ""}
        {item.changePct.toFixed(2)}%
      </span>
    </div>
  );
}

export function MarketOverview({
  indices,
  futures,
  commodities,
  loading,
}: MarketOverviewProps) {
  return (
    <TerminalPanel
      title="Market Overview"
      tag="INDICES"
      headerRight={
        <TerminalBadge variant="green">LIVE</TerminalBadge>
      }
    >
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Loading market data...
        </div>
      ) : (
        <>
          {/* Header row */}
          <div className="flex items-center justify-between py-[1px] text-[10px] text-terminal-dim">
            <span className="w-16">SYM</span>
            <span className="w-24 text-right">LAST</span>
            <span className="w-16 text-right">CHG</span>
            <span className="w-16 text-right">CHG%</span>
          </div>
          <TerminalDivider />

          {/* Indices */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ INDICES
          </div>
          {indices.map((item) => (
            <PriceRow key={item.symbol} item={item} />
          ))}

          <TerminalDivider />

          {/* Futures */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ FUTURES
          </div>
          {futures.map((item) => (
            <PriceRow key={item.symbol} item={item} />
          ))}

          <TerminalDivider />

          {/* Commodities */}
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ COMMODITIES
          </div>
          {commodities.map((item) => (
            <PriceRow key={item.symbol} item={item} />
          ))}
        </>
      )}
    </TerminalPanel>
  );
}
