"use client";

import { useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { fmtUsd } from "@/lib/cockpit-trust";
import { HOLDINGS as MOCK_HOLDINGS, type Holding } from "./mock";
import { PfCard, WheelBadge, fmtSignedUsd, pnlColor } from "./parts";

type SortKey = "sym" | "mktValue" | "uPnl" | "pctNav";
type Sort = { key: SortKey; dir: 1 | -1 };

// Module-scope so the sortable header isn't a component created during the
// parent's render (react-hooks/cannot-create-components-during-render); sort
// state + toggle flow in as props.
function Th({
  label,
  k,
  align = "right",
  sort,
  toggle,
}: {
  label: string;
  k: SortKey;
  align?: "left" | "right";
  sort: Sort;
  toggle: (k: SortKey) => void;
}) {
  return (
    <th className={`px-3 py-2 ${align === "left" ? "text-left" : "text-right"}`}>
      <button
        onClick={() => toggle(k)}
        className={`inline-flex items-center gap-1 transition-colors hover:text-terminal-text ${
          sort.key === k ? "text-terminal-text" : ""
        } ${align === "right" ? "flex-row-reverse" : ""}`}
      >
        {label}
        {sort.key === k &&
          (sort.dir === -1 ? (
            <ArrowDown className="h-3 w-3" />
          ) : (
            <ArrowUp className="h-3 w-3" />
          ))}
      </button>
    </th>
  );
}

export function HoldingsTable({ holdings = MOCK_HOLDINGS }: { holdings?: Holding[] }) {
  const [sort, setSort] = useState<Sort>({
    key: "pctNav",
    dir: -1,
  });

  const rows = [...holdings].sort((a, b) => {
    if (sort.key === "sym") return a.sym.localeCompare(b.sym) * sort.dir;
    return (a[sort.key] - b[sort.key]) * sort.dir;
  });

  const toggle = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: (s.dir * -1) as 1 | -1 } : { key, dir: -1 }));

  return (
    <PfCard pad={false} title="Holdings" right={<span className="text-[10px] text-terminal-dim">{rows.length} positions</span>}>
      <div className="overflow-x-auto px-2 pb-2">
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-terminal-dim">
              <Th label="Symbol" k="sym" align="left" sort={sort} toggle={toggle} />
              <th className="px-3 py-2 text-left">State</th>
              <th className="px-3 py-2 text-right">Qty</th>
              <th className="px-3 py-2 text-right">Mark</th>
              <Th label="Value" k="mktValue" sort={sort} toggle={toggle} />
              <Th label="Unreal. P&L" k="uPnl" sort={sort} toggle={toggle} />
              <Th label="% NAV" k="pctNav" sort={sort} toggle={toggle} />
            </tr>
          </thead>
          <tbody>
            {rows.map((h) => (
              <Row key={h.sym} h={h} />
            ))}
          </tbody>
        </table>
      </div>
    </PfCard>
  );
}

function Row({ h }: { h: Holding }) {
  const [hover, setHover] = useState(false);
  const barColor = h.breach ? "#f2495e" : "var(--color-pf-accent)";
  return (
    <tr
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="border-t border-white/[0.05]"
      style={{ background: hover ? "rgba(255,255,255,0.03)" : "transparent" }}
    >
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-terminal-text">{h.sym}</span>
          {h.currency === "CAD" && (
            <span className="rounded bg-white/[0.06] px-1 text-[9px] text-terminal-dim">CAD</span>
          )}
        </div>
        <div className="text-[10px] text-terminal-dim">{h.name}</div>
      </td>
      <td className="px-3 py-2">
        <WheelBadge state={h.state} compact />
      </td>
      <td className="px-3 py-2 text-right font-mono tabular-nums text-terminal-text">{h.qty}</td>
      <td className="px-3 py-2 text-right font-mono tabular-nums text-terminal-text">
        {fmtUsd(h.mark, { decimals: 2 })}
      </td>
      <td className="px-3 py-2 text-right font-mono tabular-nums text-terminal-text">
        {fmtUsd(h.mktValue)}
      </td>
      <td className={`px-3 py-2 text-right font-mono tabular-nums ${pnlColor(h.uPnl)}`}>
        {fmtSignedUsd(h.uPnl)}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center justify-end gap-2">
          <div className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-white/[0.07] sm:block">
            <div
              className="h-full rounded-full"
              style={{ width: `${Math.min(h.pctNav, 100)}%`, background: barColor }}
            />
          </div>
          <span
            className="w-10 text-right font-mono tabular-nums"
            style={{ color: h.breach ? "#f2495e" : "#94a3b8" }}
          >
            {h.pctNav}%
          </span>
        </div>
      </td>
    </tr>
  );
}
