"use client";

import { useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { ASK_CHIPS } from "./mock";
import { PfCard } from "./parts";

// Visual affordance only for the aesthetics round — the conversational layer
// (engine + IBKR-backed queries) is wired in the functionality round.
export function AskBar() {
  const [q, setQ] = useState("");
  return (
    <PfCard>
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 shrink-0 text-pf-accent" />
        <span className="text-xs font-medium text-terminal-text">Ask your portfolio</span>
        <span className="rounded bg-pf-accent/15 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-pf-accent">
          beta
        </span>
      </div>
      <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/[0.08] bg-pf-bg px-3 py-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. how did MU do this month?"
          className="min-w-0 flex-1 bg-transparent text-sm text-terminal-text outline-none placeholder:text-terminal-dim"
        />
        <button
          className="flex h-7 w-7 items-center justify-center rounded-md bg-pf-accent/15 text-pf-accent transition-colors hover:bg-pf-accent/25"
          aria-label="Ask"
        >
          <Send className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {ASK_CHIPS.map((c) => (
          <button
            key={c}
            onClick={() => setQ(c)}
            className="rounded-full border border-white/[0.08] px-2.5 py-1 text-[11px] text-terminal-dim transition-colors hover:border-pf-accent/40 hover:text-terminal-text"
          >
            {c}
          </button>
        ))}
      </div>
    </PfCard>
  );
}
