"use client";

// Trade history tab — every IBKR execution (buys / sells / expiries / assignments)
// from the Flex export, with per-ticker realized-P&L broken out by asset class.
// Read-only + observational (CLAUDE.md §2/§3): reports what happened, never a
// forward score, never an order. The headline "how much did I make/lose on
// <ticker> options" is realized P&L INCLUDING expiries — the whole reason this
// uses Flex (the connector feed misses expired-worthless premium).
//
// P&L is reported in USD-equivalent (each trade converted from its native
// currency via the live snapshot's FX) so totals sum across currencies. CLS is
// dual-listed (NYSE-USD / TSX-CAD); such names are flagged `mixed`.

import { useMemo, useState } from "react";
import { ArrowLeftRight, Filter, X } from "lucide-react";

import { fmtUsd } from "@/lib/cockpit-trust";
import { PfCard, ProvenanceBadge, fmtSignedUsd, pnlColor } from "./parts";
import { useTradesData, type Trade } from "./use-trades-data";

type AssetClass = "ALL" | "OPT" | "STK" | "CASH";

const CLASS_TABS: { key: AssetClass; label: string }[] = [
  { key: "ALL", label: "All" },
  { key: "STK", label: "Stock" },
  { key: "OPT", label: "Options" },
  { key: "CASH", label: "Cash / FX" },
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function shortExpiry(iso: string | null): string {
  if (!iso) return "";
  const [y, m, d] = iso.split("-");
  const mi = Number(m) - 1;
  return `${d}${MONTHS[mi] ?? m}${y.slice(2)}`;
}

/** Compact contract descriptor for the Type column. */
function contractLabel(t: Trade): string {
  if (t.sec_type === "OPT") {
    const strike = t.strike != null ? `${t.strike}` : "";
    return `${t.right ?? "?"} ${strike} ${shortExpiry(t.expiry)}`.trim();
  }
  if (t.sec_type === "STK") return "Shares";
  return t.sec_type;
}

/** Human tag for the FIFO close code (Ep = expired, A = assigned). */
function codeTag(code: string | null): { label: string; color: string } | null {
  if (!code) return null;
  const c = code.toUpperCase();
  if (c.includes("EP")) return { label: "expired", color: "#34d399" };
  if (c === "A" || c.includes("IA")) return { label: "assigned", color: "#f5a524" };
  if (c === "EX") return { label: "exercised", color: "#f5a524" };
  return null;
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-terminal-dim">{label}</div>
      <div className={`text-sm font-semibold tabular-nums ${color ?? "text-terminal-text"}`}>{value}</div>
    </div>
  );
}

const MAX_ROWS = 500;

export function TradesPanel() {
  const { trades, tickers, totals, coverage, source, loading } = useTradesData();

  const [cls, setCls] = useState<AssetClass>("ALL");
  const [ticker, setTicker] = useState<string | null>(null);
  const [from, setFrom] = useState<string>("");
  const [to, setTo] = useState<string>("");

  const selected = useMemo(
    () => tickers.find((t) => t.symbol === ticker) ?? null,
    [tickers, ticker]
  );

  const filtered = useMemo(() => {
    const rows = trades.filter((t) => {
      if (ticker && t.symbol !== ticker) return false;
      if (cls !== "ALL" && t.sec_type !== cls) return false;
      if (from && (t.date ?? "") < from) return false;
      if (to && (t.date ?? "") > to) return false;
      return true;
    });
    // most-recent first
    rows.sort((a, b) => (b.datetime ?? "").localeCompare(a.datetime ?? ""));
    return rows;
  }, [trades, ticker, cls, from, to]);

  const filteredPnl = useMemo(
    () => filtered.reduce((s, t) => s + (t.realized_pnl || 0), 0),
    [filtered]
  );

  const hasFilters = ticker || cls !== "ALL" || from || to;
  const clear = () => {
    setTicker(null);
    setCls("ALL");
    setFrom("");
    setTo("");
  };

  const coverageNote = coverage
    ? `${totals.trade_count.toLocaleString()} trades · ${coverage.from ?? "?"} → ${coverage.to ?? "?"}`
    : "";

  return (
    <PfCard
      pad={false}
      title="Trades"
      right={
        <span className="flex items-center gap-1.5 text-[10px] text-terminal-dim">
          <ProvenanceBadge source={loading ? undefined : source} />
          <ArrowLeftRight className="h-3.5 w-3.5 text-terminal-dim" />
          realized history · Flex
        </span>
      }
    >
      <div className="px-4 pb-4 pt-2">
        {loading ? (
          <p className="py-8 text-center text-xs text-terminal-dim">Loading trades…</p>
        ) : trades.length === 0 ? (
          <p className="py-8 text-center text-xs text-terminal-dim">
            No trade history available (engine offline or trades.json not yet ingested).
          </p>
        ) : (
          <>
            {/* Totals strip */}
            <div className="mb-3 grid grid-cols-2 gap-3 border-b border-white/[0.08] pb-3 sm:grid-cols-4">
              <Stat
                label="Realized (USD)"
                value={fmtSignedUsd(totals.realized_usd)}
                color={pnlColor(totals.realized_usd)}
              />
              <Stat
                label="Options P&L"
                value={fmtSignedUsd(totals.opt_realized_usd)}
                color={pnlColor(totals.opt_realized_usd)}
              />
              <Stat
                label="Stock P&L"
                value={fmtSignedUsd(totals.stk_realized_usd)}
                color={pnlColor(totals.stk_realized_usd)}
              />
              <Stat label="Coverage" value={coverageNote} />
            </div>

            {/* Controls */}
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {/* asset-class toggle */}
              <div className="inline-flex items-center gap-0.5 rounded-lg border border-white/[0.08] bg-pf-bg p-0.5">
                {CLASS_TABS.map((c) => (
                  <button
                    key={c.key}
                    type="button"
                    onClick={() => setCls(c.key)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                      cls === c.key
                        ? "bg-pf-accent/15 text-pf-accent"
                        : "text-terminal-dim hover:text-terminal-text"
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>

              {/* ticker dropdown */}
              <div className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] bg-pf-bg px-2 py-1">
                <Filter className="h-3 w-3 text-terminal-dim" />
                <select
                  value={ticker ?? ""}
                  onChange={(e) => setTicker(e.target.value || null)}
                  className="bg-transparent text-[11px] text-terminal-text outline-none"
                  aria-label="Filter by ticker"
                >
                  <option value="">All tickers</option>
                  {tickers.map((t) => (
                    <option key={t.symbol} value={t.symbol}>
                      {t.symbol}
                    </option>
                  ))}
                </select>
              </div>

              {/* date range */}
              <label className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] bg-pf-bg px-2 py-1 text-[11px] text-terminal-dim">
                from
                <input
                  type="date"
                  value={from}
                  min={coverage?.from ?? undefined}
                  max={coverage?.to ?? undefined}
                  onChange={(e) => setFrom(e.target.value)}
                  className="bg-transparent text-terminal-text outline-none"
                />
              </label>
              <label className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] bg-pf-bg px-2 py-1 text-[11px] text-terminal-dim">
                to
                <input
                  type="date"
                  value={to}
                  min={coverage?.from ?? undefined}
                  max={coverage?.to ?? undefined}
                  onChange={(e) => setTo(e.target.value)}
                  className="bg-transparent text-terminal-text outline-none"
                />
              </label>

              {hasFilters && (
                <button
                  type="button"
                  onClick={clear}
                  className="inline-flex items-center gap-1 rounded-lg border border-white/[0.08] px-2 py-1 text-[11px] text-terminal-dim hover:text-terminal-text"
                >
                  <X className="h-3 w-3" /> clear
                </button>
              )}
            </div>

            {/* Selected-ticker P&L card */}
            {selected && (
              <div className="mb-3 rounded-lg border border-pf-accent/25 bg-pf-accent/[0.04] p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-semibold text-terminal-text">
                    {selected.symbol}
                    {selected.mixed_currency && (
                      <span className="ml-2 text-[10px] font-normal text-terminal-dim">
                        (dual-listed — USD-equivalent)
                      </span>
                    )}
                  </span>
                  <span className="text-[10px] text-terminal-dim">
                    {selected.first} → {selected.last}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat
                    label="Options P&L"
                    value={fmtSignedUsd(selected.opt_pnl_usd)}
                    color={pnlColor(selected.opt_pnl_usd)}
                  />
                  <Stat
                    label="Stock P&L"
                    value={fmtSignedUsd(selected.stk_pnl_usd)}
                    color={pnlColor(selected.stk_pnl_usd)}
                  />
                  <Stat
                    label="Total P&L"
                    value={fmtSignedUsd(selected.total_pnl_usd)}
                    color={pnlColor(selected.total_pnl_usd)}
                  />
                  <Stat
                    label="Premium collected"
                    value={fmtUsd(selected.premium_collected)}
                  />
                </div>
                <p className="mt-2 text-[10px] text-terminal-dim">
                  {selected.opt_count} option · {selected.stk_count} stock trades. Options P&L
                  includes expiries &amp; assignments (premium kept at expiry counts).
                </p>
              </div>
            )}

            {/* Per-ticker league table (click a row to filter) */}
            {!selected && (
              <div className="mb-3 overflow-x-auto">
                <table className="w-full min-w-[540px] text-[11px]">
                  <thead>
                    <tr className="text-left text-[10px] uppercase tracking-wide text-terminal-dim">
                      <th className="pb-1 font-medium">Ticker</th>
                      <th className="pb-1 text-right font-medium">Options P&L</th>
                      <th className="pb-1 text-right font-medium">Stock P&L</th>
                      <th className="pb-1 text-right font-medium">Total (USD)</th>
                      <th className="pb-1 text-right font-medium">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tickers.map((t) => (
                      <tr
                        key={t.symbol}
                        onClick={() => setTicker(t.symbol)}
                        className="cursor-pointer border-t border-white/[0.05] hover:bg-white/[0.03]"
                      >
                        <td className="py-1.5 font-medium text-terminal-text">
                          {t.symbol}
                          {t.mixed_currency ? (
                            <span className="ml-1 text-[9px] text-terminal-dim">±fx</span>
                          ) : t.currency !== "USD" ? (
                            <span className="ml-1 text-[9px] text-terminal-dim">{t.currency}</span>
                          ) : null}
                        </td>
                        <td className={`py-1.5 text-right tabular-nums ${pnlColor(t.opt_pnl_usd)}`}>
                          {fmtSignedUsd(t.opt_pnl_usd)}
                        </td>
                        <td className={`py-1.5 text-right tabular-nums ${pnlColor(t.stk_pnl_usd)}`}>
                          {fmtSignedUsd(t.stk_pnl_usd)}
                        </td>
                        <td className={`py-1.5 text-right tabular-nums ${pnlColor(t.total_pnl_usd)}`}>
                          {fmtSignedUsd(t.total_pnl_usd)}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-terminal-dim">
                          {t.opt_count + t.stk_count + t.other_count}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Trade table */}
            <div className="mb-1 flex items-center justify-between text-[10px] text-terminal-dim">
              <span>
                {filtered.length.toLocaleString()} trade{filtered.length === 1 ? "" : "s"}
                {filtered.length > MAX_ROWS && ` (showing first ${MAX_ROWS})`}
              </span>
              <span>
                filtered realized:{" "}
                <span className={`tabular-nums ${pnlColor(filteredPnl)}`}>
                  {fmtSignedUsd(filteredPnl)}
                </span>{" "}
                <span className="text-terminal-dim">(native ccy sum)</span>
              </span>
            </div>
            <div className="max-h-[520px] overflow-auto rounded-lg border border-white/[0.06]">
              <table className="w-full min-w-[720px] text-[11px]">
                <thead className="sticky top-0 z-10 bg-pf-panel">
                  <tr className="text-left text-[10px] uppercase tracking-wide text-terminal-dim">
                    <th className="px-2 py-1.5 font-medium">Date</th>
                    <th className="px-2 py-1.5 font-medium">Ticker</th>
                    <th className="px-2 py-1.5 font-medium">Type</th>
                    <th className="px-2 py-1.5 font-medium">Side</th>
                    <th className="px-2 py-1.5 text-right font-medium">Qty</th>
                    <th className="px-2 py-1.5 text-right font-medium">Price</th>
                    <th className="px-2 py-1.5 text-right font-medium">Proceeds</th>
                    <th className="px-2 py-1.5 text-right font-medium">Realized</th>
                    <th className="px-2 py-1.5 font-medium">Ccy</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, MAX_ROWS).map((t) => {
                    const tag = codeTag(t.code);
                    return (
                      <tr key={t.id} className="border-t border-white/[0.05] hover:bg-white/[0.03]">
                        <td className="whitespace-nowrap px-2 py-1 text-terminal-dim">{t.date}</td>
                        <td className="px-2 py-1 font-medium text-terminal-text">{t.symbol}</td>
                        <td className="whitespace-nowrap px-2 py-1 text-terminal-text">
                          {contractLabel(t)}
                          {tag && (
                            <span
                              className="ml-1.5 rounded px-1 py-0.5 text-[9px] font-semibold uppercase"
                              style={{ background: `${tag.color}22`, color: tag.color }}
                            >
                              {tag.label}
                            </span>
                          )}
                        </td>
                        <td
                          className={`px-2 py-1 ${
                            t.side === "SELL" ? "text-pf-gain" : "text-terminal-text"
                          }`}
                        >
                          {t.side}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums text-terminal-dim">
                          {t.qty}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums text-terminal-dim">
                          {t.price}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums text-terminal-dim">
                          {t.proceeds != null ? fmtSignedUsd(t.proceeds) : "—"}
                        </td>
                        <td
                          className={`px-2 py-1 text-right tabular-nums ${
                            t.realized_pnl ? pnlColor(t.realized_pnl) : "text-terminal-dim"
                          }`}
                        >
                          {t.realized_pnl ? fmtSignedUsd(t.realized_pnl) : "—"}
                        </td>
                        <td className="px-2 py-1 text-terminal-dim">{t.currency}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-[10px] leading-snug text-terminal-dim">
              Realized column is each trade&apos;s FIFO P&L in its native currency; the summary
              cards above convert to USD-equivalent (current FX) so cross-currency totals sum.
              Includes expiries (kept premium) &amp; assignments — the accurate options view.
            </p>
          </>
        )}
      </div>
    </PfCard>
  );
}
