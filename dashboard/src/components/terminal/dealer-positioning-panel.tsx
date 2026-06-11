"use client";

import { useState, useEffect } from "react";
import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";

/**
 * Dealer positioning read for one underlying: GEX, put/call walls,
 * gamma-flip distance, dealer regime — via /api/engine?action=
 * dealer_positioning (engine /api/tv/dealer_positioning).
 *
 * Display-only context: the dealer multiplier itself is applied engine-side
 * (clamped [0.70, 1.05]) and R6 downgrades fire in the reviewer chain —
 * nothing here scores or upgrades anything.
 *
 * The Bloomberg provider has NO option-chain access, so on that rig the
 * engine answers 404 "option chain unavailable" for every ticker — render
 * that as an explicit unavailable state, never fabricated walls.
 */

interface Wall {
  strike?: number;
  distance_pct?: number;
}

interface DealerPositioning {
  ticker?: string;
  as_of?: string;
  spot?: number;
  expiry?: string;
  gex_total?: number;
  regime?: string;
  confidence?: number;
  flip_level?: number;
  flip_distance_pct?: number;
  nearest_call_wall?: Wall | null;
  nearest_put_wall?: Wall | null;
  n_strikes?: number;
}

interface DealerPositioningPanelProps {
  ticker: string;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Compact $ for large gamma-exposure notionals. */
function fmtGex(v: number | null): string {
  if (v === null) return "—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "−" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

function fmtPct(v: number | null): string {
  return v !== null ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` : "—";
}

export function DealerPositioningPanel({ ticker }: DealerPositioningPanelProps) {
  const [data, setData] = useState<DealerPositioning | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchDealer() {
      setLoading(true);
      setData(null);
      setUnavailable(null);
      try {
        const res = await fetch(
          `/api/engine?action=dealer_positioning&ticker=${encodeURIComponent(
            ticker
          )}&dte=35`
        );
        if (cancelled) return;
        if (!res.ok) {
          // The proxy folds engine errors into 503 with the upstream status in
          // `detail`; a 404 there is the engine's structured "no chain" answer.
          const body = await res.json().catch(() => null);
          const detail = typeof body?.detail === "string" ? body.detail : "";
          setUnavailable(
            detail.includes("404")
              ? `Option chain unavailable for ${ticker} on the current data provider`
              : "Engine unavailable"
          );
          return;
        }
        setData(await res.json());
      } catch {
        if (!cancelled) setUnavailable("Engine unavailable");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchDealer();
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  const gex = num(data?.gex_total);
  const callWall = data?.nearest_call_wall ?? null;
  const putWall = data?.nearest_put_wall ?? null;

  return (
    <TerminalPanel
      title={`${ticker} — Dealer Positioning`}
      tag="GEX"
      headerRight={
        data?.as_of ? (
          <span className="text-[10px] text-terminal-dim">
            as of {data.as_of}
          </span>
        ) : undefined
      }
    >
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Reading option chain…
        </div>
      ) : unavailable ? (
        <div className="flex h-full flex-col items-center justify-center gap-1 px-4 text-center text-terminal-dim">
          <span>{unavailable}</span>
          <span className="text-[10px]">
            Dealer GEX/walls need a chain-capable provider (Theta); the
            Bloomberg IV file carries no per-strike open interest.
          </span>
        </div>
      ) : data ? (
        <>
          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ DEALER REGIME
          </div>
          <div className="mb-1 flex items-center gap-2">
            {data.regime ? (
              <TerminalBadge
                variant={
                  String(data.regime).toLowerCase().includes("short")
                    ? "amber"
                    : "green"
                }
              >
                {data.regime}
              </TerminalBadge>
            ) : (
              <span className="text-terminal-dim">—</span>
            )}
            {num(data.confidence) !== null && (
              <span className="text-[10px] text-terminal-dim">
                conf {(num(data.confidence)! * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <TerminalRow
            label="GEX total"
            value={fmtGex(gex)}
            valueColor={
              gex === null
                ? "text-terminal-dim"
                : gex >= 0
                  ? "text-terminal-green"
                  : "text-terminal-amber"
            }
          />
          <TerminalRow
            label="Spot"
            value={num(data.spot) !== null ? `$${num(data.spot)!.toFixed(2)}` : "—"}
          />
          <TerminalRow label="Chain expiry" value={data.expiry || "—"} />

          <TerminalDivider />

          <div className="mb-1 text-[10px] font-bold text-terminal-blue">
            ─ GAMMA FLIP / WALLS
          </div>
          <TerminalRow
            label="Flip level"
            value={
              num(data.flip_level) !== null
                ? `$${num(data.flip_level)!.toFixed(2)}`
                : "—"
            }
          />
          <TerminalRow
            label="Flip distance"
            value={fmtPct(num(data.flip_distance_pct))}
          />
          <TerminalRow
            label="Put wall"
            value={
              num(putWall?.strike) !== null
                ? `$${num(putWall?.strike)} (${fmtPct(num(putWall?.distance_pct))})`
                : "—"
            }
          />
          <TerminalRow
            label="Call wall"
            value={
              num(callWall?.strike) !== null
                ? `$${num(callWall?.strike)} (${fmtPct(num(callWall?.distance_pct))})`
                : "—"
            }
          />
          <div className="mt-1 text-[10px] text-terminal-dim">
            Display-only context — the dealer multiplier is applied engine-side
            (clamped 0.70–1.05) and never rescues negative EV.
          </div>
        </>
      ) : (
        <span className="text-terminal-dim">No data</span>
      )}
    </TerminalPanel>
  );
}
