"use client";

import {
  TerminalPanel,
  TerminalDivider,
  TerminalRow,
  TerminalBadge,
} from "./panel";
import type { AgentStatus, AgentTask } from "@/types";

interface AgentPanelProps {
  status: AgentStatus;
  tasks: AgentTask[];
  connected: boolean;
}

const STATUS_ICON: Record<string, string> = {
  queued: "○",
  running: "▶",
  completed: "✓",
  failed: "✗",
};

const STATUS_COLOR: Record<string, string> = {
  queued: "text-terminal-dim",
  running: "text-terminal-amber",
  completed: "text-terminal-green",
  failed: "text-terminal-red",
};

export function AgentPanel({ status, tasks, connected }: AgentPanelProps) {
  const runningTasks = tasks.filter((t) => t.status === "running");
  const queuedTasks = tasks.filter((t) => t.status === "queued");
  const recentCompleted = tasks
    .filter((t) => t.status === "completed" || t.status === "failed")
    .slice(0, 5);

  return (
    <TerminalPanel
      title="AI Agent"
      tag="LOCAL"
      headerRight={
        connected ? (
          <TerminalBadge variant="green">ONLINE</TerminalBadge>
        ) : (
          <TerminalBadge variant="amber">PLACEHOLDER</TerminalBadge>
        )
      }
    >
      {/* Agent Status */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ SYSTEM
      </div>
      <TerminalRow
        label="Status"
        value={status.online ? "ONLINE" : "OFFLINE"}
        valueColor={
          status.online ? "text-terminal-green" : "text-terminal-red"
        }
      />
      <TerminalRow label="Model" value={status.model} />
      <TerminalRow
        label="VRAM"
        value={`${status.vramUsage.toFixed(1)}/${status.vramTotal}GB`}
        valueColor={
          status.vramUsage / status.vramTotal > 0.9
            ? "text-terminal-red"
            : status.vramUsage / status.vramTotal > 0.7
            ? "text-terminal-amber"
            : "text-terminal-green"
        }
      />
      <TerminalRow label="RAM" value={`${status.ramUsage.toFixed(1)}GB`} />
      <TerminalRow label="Browser Tabs" value={status.activeTabs} />
      <TerminalRow label="Uptime" value={status.uptime} />

      <TerminalDivider />

      {/* Pipeline */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ PIPELINE
      </div>

      {/* Progress bar for VRAM */}
      <div className="mb-2">
        <div className="flex justify-between text-[10px] text-terminal-dim mb-0.5">
          <span>VRAM USAGE</span>
          <span>
            {((status.vramUsage / status.vramTotal) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-1.5 w-full bg-terminal-border">
          <div
            className={`h-full transition-all ${
              status.vramUsage / status.vramTotal > 0.9
                ? "bg-terminal-red"
                : status.vramUsage / status.vramTotal > 0.7
                ? "bg-terminal-amber"
                : "bg-terminal-green"
            }`}
            style={{
              width: `${(status.vramUsage / status.vramTotal) * 100}%`,
            }}
          />
        </div>
      </div>

      {/* Running tasks */}
      {runningTasks.length > 0 && (
        <>
          <div className="text-[10px] text-terminal-amber mb-0.5">
            ACTIVE ({runningTasks.length}):
          </div>
          {runningTasks.map((task) => (
            <div key={task.id} className="flex items-center gap-1 py-[1px]">
              <span className="text-terminal-amber animate-pulse">▶</span>
              <span className="text-terminal-text truncate">
                {task.description}
              </span>
            </div>
          ))}
        </>
      )}

      {/* Queued tasks */}
      {queuedTasks.length > 0 && (
        <>
          <div className="text-[10px] text-terminal-dim mt-1 mb-0.5">
            QUEUED ({queuedTasks.length}):
          </div>
          {queuedTasks.map((task) => (
            <div key={task.id} className="flex items-center gap-1 py-[1px]">
              <span className="text-terminal-dim">○</span>
              <span className="text-terminal-dim truncate">
                {task.description}
              </span>
            </div>
          ))}
        </>
      )}

      <TerminalDivider />

      {/* Recent activity */}
      <div className="mb-1 text-[10px] font-bold text-terminal-blue">
        ─ RECENT
      </div>
      <TerminalRow
        label="Tasks Completed"
        value={status.tasksCompleted}
        valueColor="text-terminal-green"
      />

      {recentCompleted.length > 0 && (
        <div className="mt-1 space-y-0.5">
          {recentCompleted.map((task) => (
            <div key={task.id} className="flex items-center gap-1 py-[1px]">
              <span className={STATUS_COLOR[task.status]}>
                {STATUS_ICON[task.status]}
              </span>
              <span className="text-terminal-dim truncate text-[11px]">
                {task.description}
              </span>
            </div>
          ))}
        </div>
      )}
    </TerminalPanel>
  );
}
