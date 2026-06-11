// The funnel — makes the silent filtering visible: how many names the engine
// started from, how many it scanned, how many survived the gates, how many
// came back after the top-N cap.
//
// HONESTY NOTES:
//   * `/api/candidates` now serializes the ranker's `attrs["drops_summary"]`
//     ({ total_dropped, by_gate }), so "passed gates" is the engine's own
//     number (scanned − gate-dropped), not a client invention. With an older
//     engine payload (no drops_summary) the funnel falls back to the single
//     collapsed "returned (top-N)" stage rather than implying a distinction
//     the API did not expose.
//   * `count` still counts the already-top-N-capped array, so it renders as
//     the final "returned (top-N)" stage, never as a separate "ranked" bar.

export interface DropsSummary {
  total_dropped: number;
  by_gate: Record<string, number>;
}

interface FunnelProps {
  universeTotal?: number;
  universeScanned?: number;
  ranked?: number;
  shown?: number;
  dropsSummary?: DropsSummary | null;
}

export function Funnel({
  universeTotal,
  universeScanned,
  ranked,
  shown,
  dropsSummary,
}: FunnelProps) {
  const dropped =
    dropsSummary && typeof dropsSummary.total_dropped === "number"
      ? dropsSummary.total_dropped
      : null;
  // Engine semantics: drops accumulate over the scanned names (chain-quality /
  // event-gate / EV-floor); survivors received an EV row, then the top-N cap
  // applies. Guard against payload skew rather than rendering a negative bar.
  const passedGates =
    dropped !== null && typeof universeScanned === "number"
      ? Math.max(0, universeScanned - dropped)
      : null;

  const gateBits = dropsSummary
    ? Object.entries(dropsSummary.by_gate)
        .sort((a, b) => b[1] - a[1])
        .map(([gate, n]) => `${gate} ${n}`)
    : [];

  const stages: { label: string; value: number | undefined | null; hint: string }[] = [
    {
      label: "S&P universe",
      value: universeTotal,
      hint: "tradeable names tracked by the connector",
    },
    {
      label: "scanned",
      value: universeScanned,
      hint: "names the ranker evaluated this run (universe_limit cap)",
    },
    ...(passedGates !== null
      ? [
          {
            label: "passed gates",
            value: passedGates,
            hint: `scanned − gate-dropped (engine drops_summary: ${
              gateBits.join(", ") || "0 drops"
            }) — before the top-N cap`,
          },
          {
            label: "returned (top-N)",
            value: shown ?? ranked,
            hint: "rows returned after the engine's top-N cap",
          },
        ]
      : [
          {
            label: "returned (top-N)",
            value: shown ?? ranked,
            hint: "ranked rows returned after the engine's top-N cap. This engine payload carries no drops_summary, so the pre-cap gate funnel is not shown rather than invented.",
          },
        ]),
  ];

  const max = universeTotal || universeScanned || ranked || 1;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-pf-panel p-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
          Selection funnel
        </span>
        {dropsSummary ? (
          <span
            className="text-[9px] text-terminal-dim"
            title={`Engine per-gate drops this run: ${gateBits.join(" · ") || "none"}`}
          >
            drops: {dropped?.toLocaleString()}
            {gateBits.length > 0 ? ` (${gateBits.slice(0, 3).join(" · ")}${gateBits.length > 3 ? " · …" : ""})` : ""}
          </span>
        ) : (
          <span
            className="text-[9px] text-terminal-dim"
            title="This payload carries no drops_summary (older engine) — per-gate breakdown unavailable."
          >
            per-gate drops: unavailable ⓘ
          </span>
        )}
      </div>
      <div className="flex flex-col gap-1">
        {stages.map((s) => {
          const v = typeof s.value === "number" ? s.value : null;
          const pct = v !== null ? Math.max(2, (v / max) * 100) : 0;
          return (
            <div key={s.label} className="flex items-center gap-2" title={s.hint}>
              <span className="w-24 shrink-0 text-[10px] text-terminal-dim">{s.label}</span>
              <div className="relative h-3.5 flex-1 rounded-sm bg-terminal-border/40">
                <div
                  className="absolute inset-y-0 left-0 rounded-sm bg-terminal-blue/40"
                  style={{ width: `${pct}%` }}
                />
                <span className="absolute inset-y-0 left-1 flex items-center text-[10px] tabular-nums text-terminal-text">
                  {v !== null ? v.toLocaleString() : "—"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
