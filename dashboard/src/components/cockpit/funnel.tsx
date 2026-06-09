// The funnel — makes the silent filtering visible: how many names the engine
// started from, how many it scanned, how many it ranked, how many you see.
//
// HONESTY NOTE: the engine computes a per-gate drop breakdown
// (`frame.attrs["drops_summary"]`: chain-quality / event-gate / ev-threshold)
// but `/api/candidates` does NOT serialize `.attrs`, so the per-gate reasons
// are not available to the dashboard. We show the stages the API DOES expose
// and flag the missing breakdown as a follow-up rather than inventing numbers.

interface FunnelProps {
  universeTotal?: number;
  universeScanned?: number;
  ranked?: number;
  shown?: number;
}

export function Funnel({ universeTotal, universeScanned, ranked, shown }: FunnelProps) {
  const stages: { label: string; value: number | undefined; hint: string }[] = [
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
    {
      label: "ranked (EV)",
      value: ranked,
      hint: "passed chain-quality + event-gate + EV floor → got an EV row",
    },
    { label: "shown", value: shown, hint: "top-N returned to the cockpit" },
  ];

  const max = universeTotal || universeScanned || ranked || 1;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-pf-panel p-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] font-bold uppercase tracking-wider text-terminal-blue">
          Selection funnel
        </span>
        <span
          className="text-[9px] text-terminal-dim"
          title="The engine computes per-gate drop reasons (chain-quality / event-gate / EV-threshold) on the DataFrame's .attrs, but /api/candidates does not serialize them. Per-gate breakdown is a follow-up."
        >
          per-gate drops: API follow-up ⓘ
        </span>
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
