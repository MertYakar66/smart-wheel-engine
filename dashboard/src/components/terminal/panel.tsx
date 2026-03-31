"use client";

import { cn } from "@/lib/utils";

interface TerminalPanelProps {
  title: string;
  tag?: string;
  children: React.ReactNode;
  className?: string;
  headerRight?: React.ReactNode;
  flash?: boolean;
}

export function TerminalPanel({
  title,
  tag,
  children,
  className,
  headerRight,
  flash,
}: TerminalPanelProps) {
  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden border border-terminal-border bg-terminal-panel",
        flash && "animate-terminal-flash",
        className
      )}
    >
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-terminal-border bg-terminal-header px-2">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-terminal-amber">
            {title}
          </span>
          {tag && (
            <span className="text-[10px] text-terminal-dim">[{tag}]</span>
          )}
        </div>
        {headerRight && (
          <div className="flex items-center gap-1">{headerRight}</div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 font-mono text-[12px] leading-[18px] text-terminal-text">
        {children}
      </div>
    </div>
  );
}

export function TerminalRow({
  label,
  value,
  valueColor,
  className,
}: {
  label: string;
  value: string | number;
  valueColor?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex justify-between", className)}>
      <span className="text-terminal-dim">{label}</span>
      <span className={valueColor || "text-terminal-text"}>{value}</span>
    </div>
  );
}

export function TerminalDivider() {
  return (
    <div className="my-1 border-t border-terminal-border opacity-50" />
  );
}

export function TerminalBadge({
  children,
  variant = "default",
}: {
  children: React.ReactNode;
  variant?: "default" | "green" | "red" | "amber" | "blue";
}) {
  const colors = {
    default: "bg-terminal-border text-terminal-text",
    green: "bg-green-900/50 text-terminal-green",
    red: "bg-red-900/50 text-terminal-red",
    amber: "bg-amber-900/50 text-terminal-amber",
    blue: "bg-blue-900/50 text-terminal-blue",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center px-1.5 py-0.5 text-[10px] font-semibold uppercase",
        colors[variant]
      )}
    >
      {children}
    </span>
  );
}
