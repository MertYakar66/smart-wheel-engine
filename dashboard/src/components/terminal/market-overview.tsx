"use client";

import { useState, useEffect } from "react";
import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import type { MarketRegime } from "@/types";

/**
 * Honest market/vol panel. Every number here is a real engine read:
 * VIX complex + term structure from action=regime, EOD last closes for a few
 * liquid universe names from action=chart (chart_type=ohlcv). No placeholder
 * index/futures/commodities quotes — absent data renders as an explicit
 * empty state, never a static number dressed up as live.
 */

interface MarketOverviewProps {
  regime: MarketRegime;
  /** Freshest OHLCV date served by the engine (action=status). */
  dataFrontier: string | null;
  connected: boolean;
  /** One-shot highlight when the command line targets this panel. */
  flash?: boolean;
}

// Liquid wheel-universe names for the EOD strip. EOD closes only — this is
// not a realtime tape and is labeled with the data frontier date.
const EOD_TICKERS = ["AAPL", "MSFT", "JPM"];

interface EodQuote {
  ticker: string;
  close: number | null;
  changePct: number | null;
  // /api/market resolution provenance: "live" (Finnhub realtime), "snapshot"
  // (<24h cached), or "eod" (engine close at the data frontier). The strip
  // may only claim "EOD · <frontier>" when every quote is actually
  // eod-sourced — a realtime quote labeled as a frontier-dated close is
  // fabricated provenance (PR #406 audit).
  source: string | null;
}

function fmt2(v: number | null | undefined): string {
  return typeof v === "number" && Number.isFinite(v)
    ? v.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    : "—";
}

export function MarketOverview({
  regime,
  dataFrontier,
  connected,
  flash,
}: MarketOverviewProps) {
  const [eod, setEod] = useState<EodQuote[]>([]);
  const [eodLoading, setEodLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function fetchEod() {
      setEodLoading(true);
      // Route through /api/market so the last-two-closes derivation and
      // null-honest changePct contract live in one place (fetchEngineEodQuote
      // via the server route). changePct is null when only one close exists —
      // never fabricated 0.
      const results = await Promise.all(
        EOD_TICKERS.map(async (ticker): Promise<EodQuote> => {
          try {
            const res = await fetch(`/api/market?ticker=${encodeURIComponent(ticker)}`);
            if (!res.ok) return { ticker, close: null, changePct: null, source: null };
            const body = await res.json();
            const close =
              typeof body?.price === "number" && Number.isFinite(body.price)
                ? body.price
                : null;
            const changePct =
              typeof body?.changePct === "number" && Number.isFinite(body.changePct)
                ? body.changePct
                : null;
            const source = typeof body?.source === "string" ? body.source : null;
            return { ticker, close, changePct, source };
          } catch {
            return { ticker, close: null, changePct: null, source: null };
          }
        })
      );
      if (!cancelled) {
        setEod(results);
        setEodLoading(false);
      }
    }
    fetchEod();
    return () => {
      cancelled = true;
    };
  }, []);

  const hasVol = regime.vix > 0;
  // contango === null means unknown/flat — never render it as backwardation.
  // Only confirmed contango === false is backwardation (stress-red badge).
  const contangoKnown = regime.contango !== null;
  const backwardation = regime.contango === false;

  return (
    <TerminalPanel
      title="Market / Vol"
      tag="ENGINE"
      flash={flash}
      headerRight={
        connected ? (
          dataFrontier ? (
            <span className="text-[10px] text-terminal-dim">
              EOD · as of {dataFrontier}
            </span>
          ) : (
            <TerminalBadge variant="green">LIVE</TerminalBadge>
          )
        ) : (
          <TerminalBadge variant="amber">OFFLINE</TerminalBadge>
        )
      }
    >
      {/* VIX complex + term structure — the premium seller's first read */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ VOL COMPLEX
      </div>
      {hasVol ? (
        <>
          <TerminalRow
            label="VIX"
            value={fmt2(regime.vix)}
            valueColor={
              regime.vix >= 25
                ? "text-terminal-red"
                : regime.vix >= 20
                  ? "text-terminal-amber"
                  : "text-terminal-green"
            }
          />
          <TerminalRow label="VIX 3M" value={fmt2(regime.vix3m)} />
          <TerminalRow label="VIX 6M" value={fmt2(regime.vix6m)} />
          <TerminalRow
            label="1Y percentile"
            value={
              regime.vixPercentile !== null
                ? `${(regime.vixPercentile * 100).toFixed(0)}%`
                : "—"
            }
          />
          <div className="flex items-center justify-between py-[2px]">
            <span className="text-terminal-dim">Term structure</span>
            {regime.termStructure ? (
              // Only confirmed backwardation (contango===false) is stress-red.
              // contango===null means the server could not determine structure
              // (unknown/flat) — render neutral, never loss-red.
              <TerminalBadge
                variant={
                  contangoKnown
                    ? backwardation ? "red" : "green"
                    : "default"
                }
              >
                {regime.termStructure}
              </TerminalBadge>
            ) : (
              <span className="text-terminal-dim">—</span>
            )}
          </div>
        </>
      ) : (
        <div className="py-1 text-[10px] italic text-terminal-dim">
          No vol data — engine offline or VIX feed absent
        </div>
      )}

      <TerminalDivider />

      {/* Regime bucket — honest label: this endpoint is a VIX-band heuristic,
          not the engine's 4-state HMM regime layer. */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ REGIME (VIX BAND)
      </div>
      <div className="flex items-center gap-2 py-[2px]">
        {regime.regime !== "---" ? (
          <TerminalBadge
            variant={
              regime.regime === "HIGH_VOL" || regime.regime === "ELEVATED"
                ? "amber"
                : regime.regime === "BEAR"
                  ? "red"
                  : regime.regime === "LOW_VOL"
                    ? "green"
                    : "blue"
            }
          >
            {regime.regime}
          </TerminalBadge>
        ) : (
          <span className="text-terminal-dim">—</span>
        )}
        <span className="text-[10px] text-terminal-dim">
          VIX-band heuristic, not the HMM
        </span>
      </div>

      <TerminalDivider />

      {/* Closes for liquid universe names — header claims "EOD · <frontier>"
          ONLY when every quote actually resolved from the engine's EOD path;
          /api/market prefers Finnhub realtime / <24h snapshots when available,
          and labeling those as frontier-dated closes would fabricate
          provenance. */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        {eodLoading || eod.length === 0
          ? "─ CLOSES …"
          : eod.every((q) => q.source === "eod" || q.close === null)
            ? `─ EOD CLOSES${dataFrontier ? ` · ${dataFrontier}` : ""}`
            : `─ LAST QUOTES · ${[...new Set(eod.filter((q) => q.source).map((q) => q.source))].join("/") || "unknown source"}`}
      </div>
      <div className="flex items-center justify-between py-[1px] text-[10px] text-terminal-dim">
        <span className="w-14">SYM</span>
        <span className="w-24 text-right">CLOSE</span>
        <span className="w-16 text-right">1D CHG%</span>
      </div>
      {eodLoading ? (
        <div className="py-1 text-[10px] text-terminal-dim">
          Loading EOD closes…
        </div>
      ) : (
        eod.map((q) => {
          const up = (q.changePct ?? 0) >= 0;
          return (
            <div
              key={q.ticker}
              className="flex items-center justify-between py-[1px]"
            >
              <span className="w-14 text-terminal-amber">{q.ticker}</span>
              <span className="w-24 text-right tabular-nums text-terminal-text">
                {q.close !== null ? fmt2(q.close) : "—"}
              </span>
              <span
                className={`w-16 text-right tabular-nums ${
                  q.changePct === null
                    ? "text-terminal-dim"
                    : up
                      ? "text-terminal-green"
                      : "text-terminal-red"
                }`}
              >
                {q.changePct !== null
                  ? `${up ? "+" : ""}${q.changePct.toFixed(2)}%`
                  : "—"}
              </span>
            </div>
          );
        })
      )}
    </TerminalPanel>
  );
}
