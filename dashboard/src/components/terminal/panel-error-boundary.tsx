"use client";

import { Component, type ReactNode } from "react";

interface PanelErrorBoundaryProps {
  /** Short panel name shown in the fallback header (e.g. "Options Engine"). */
  label?: string;
  /**
   * Data epoch for auto-recovery. When this changes (e.g. the poll's
   * lastUpdated timestamp) a latched error state resets, so one transient
   * malformed payload doesn't kill the panel until a manual Retry click.
   */
  resetKey?: unknown;
  children: ReactNode;
}

interface PanelErrorBoundaryState {
  hasError: boolean;
  message?: string;
}

/**
 * Per-panel error isolation for the terminal grid.
 *
 * The (terminal) route also has an App-Router `error.tsx`, but that is a
 * WHOLE-ROUTE boundary: one panel throwing (e.g. an OptionsPanel `.toFixed`
 * on a missing engine field) would replace the entire terminal — all panels —
 * and crash-loop on the 60s poll. Wrapping each panel in its own boundary
 * keeps a single misbehaving panel contained: it degrades to a small "panel
 * error" card while the other panels keep rendering.
 *
 * Must be a class component — React error boundaries cannot be hooks.
 */
export class PanelErrorBoundary extends Component<
  PanelErrorBoundaryProps,
  PanelErrorBoundaryState
> {
  state: PanelErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(error: unknown): PanelErrorBoundaryState {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : String(error),
    };
  }

  componentDidCatch(error: unknown) {
    // Keep the diagnostic in the console; do not rethrow (that would bubble to
    // the whole-route error.tsx and defeat the per-panel isolation).
    console.error(`[${this.props.label ?? "panel"}] render error:`, error);
  }

  componentDidUpdate(prevProps: PanelErrorBoundaryProps) {
    // Auto-reset on a new data epoch: the next healthy poll un-latches the
    // fallback and re-attempts the children render.
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, message: undefined });
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, message: undefined });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="flex flex-col overflow-hidden border border-terminal-border bg-terminal-panel">
        <div className="flex h-7 shrink-0 items-center justify-between gap-2 border-b border-terminal-border bg-terminal-header px-2">
          <span className="truncate text-[11px] font-bold uppercase tracking-wider text-terminal-red">
            {this.props.label ?? "Panel"} — error
          </span>
          <button
            onClick={this.handleRetry}
            className="shrink-0 border border-terminal-border px-1.5 py-0.5 text-[10px] uppercase text-terminal-text transition-colors hover:bg-terminal-border/40"
          >
            Retry
          </button>
        </div>
        <div className="flex flex-1 flex-col gap-1 overflow-auto p-2 font-mono text-[11px] text-terminal-dim">
          <span>
            This panel failed to render; the rest of the terminal is
            unaffected.
          </span>
          {this.state.message ? (
            <span className="break-all text-terminal-red">
              {this.state.message}
            </span>
          ) : null}
        </div>
      </div>
    );
  }
}
