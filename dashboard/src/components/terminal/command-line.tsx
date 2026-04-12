"use client";

import { useState, useRef, useEffect } from "react";

interface CommandLineProps {
  onCommand: (command: string) => void;
  history: string[];
}

const COMMANDS = [
  { cmd: "NEWS", desc: "Open news panel" },
  { cmd: "OPTIONS", desc: "Open options engine" },
  { cmd: "AGENT", desc: "Open AI agent panel" },
  { cmd: "WATCH <SYM>", desc: "Add ticker to watchlist" },
  { cmd: "UNWATCH <SYM>", desc: "Remove from watchlist" },
  { cmd: "QUOTE <SYM>", desc: "Get stock quote" },
  { cmd: "REFRESH", desc: "Ingest RSS feeds" },
  { cmd: "ENGINE", desc: "Refresh options engine data" },
  { cmd: "CALENDAR", desc: "Show macro calendar" },
  { cmd: "RESEARCH <Q>", desc: "Ask AI a question" },
  { cmd: "CLEAR", desc: "Clear command history" },
  { cmd: "HELP", desc: "Show available commands" },
];

export function CommandLine({ onCommand, history }: CommandLineProps) {
  const [input, setInput] = useState("");
  const [historyIdx, setHistoryIdx] = useState(-1);
  const [showHelp, setShowHelp] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = () => {
    if (!input.trim()) return;
    const cmd = input.trim().toUpperCase();
    onCommand(cmd);
    setInput("");
    setHistoryIdx(-1);
    if (cmd === "HELP") {
      setShowHelp(true);
    } else {
      setShowHelp(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSubmit();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length > 0) {
        const newIdx = Math.min(historyIdx + 1, history.length - 1);
        setHistoryIdx(newIdx);
        setInput(history[history.length - 1 - newIdx]);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (historyIdx > 0) {
        const newIdx = historyIdx - 1;
        setHistoryIdx(newIdx);
        setInput(history[history.length - 1 - newIdx]);
      } else {
        setHistoryIdx(-1);
        setInput("");
      }
    } else if (e.key === "Escape") {
      setInput("");
      setShowHelp(false);
    } else if (e.key === "Tab") {
      e.preventDefault();
      // Auto-complete from commands
      const partial = input.toUpperCase();
      const match = COMMANDS.find((c) =>
        c.cmd.split(" ")[0].startsWith(partial)
      );
      if (match) {
        setInput(match.cmd.split(" ")[0]);
      }
    }
  };

  return (
    <div className="relative shrink-0 border-t border-terminal-border bg-terminal-header">
      {/* Help popup — high z so it sits above chart panels and grids */}
      {showHelp && (
        <div className="absolute bottom-full left-0 right-0 z-50 border border-terminal-border bg-terminal-panel p-3 shadow-2xl">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[11px] font-bold text-terminal-amber">
              AVAILABLE COMMANDS
            </span>
            <button
              onClick={() => setShowHelp(false)}
              className="text-terminal-dim hover:text-terminal-text text-[11px]"
            >
              [ESC to close]
            </button>
          </div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-1">
            {COMMANDS.map((cmd) => (
              <div key={cmd.cmd} className="flex gap-2 text-[11px]">
                <span className="text-terminal-amber font-mono w-28">
                  {cmd.cmd}
                </span>
                <span className="text-terminal-dim">{cmd.desc}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 pt-2 border-t border-terminal-border text-[10px] text-terminal-dim">
            Tip: type any ticker symbol (e.g. <span className="text-terminal-amber">AAPL</span>) to open its chart.
            Type <span className="text-terminal-amber">BACK</span> to return.
          </div>
        </div>
      )}

      {/* Command input */}
      <div className="flex h-9 items-center px-3 font-mono text-[12px]">
        <span className="mr-2 text-terminal-amber font-bold">&gt;</span>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            setShowHelp(false);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Enter command... (HELP for list, TAB to autocomplete)"
          className="flex-1 bg-transparent text-terminal-text placeholder:text-terminal-dim/50 outline-none caret-terminal-amber"
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
        />
        <div className="flex items-center gap-2 text-[10px] text-terminal-dim">
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="hover:text-terminal-amber transition-colors"
          >
            [F1 HELP]
          </button>
          <span>[ESC]</span>
        </div>
      </div>
    </div>
  );
}
