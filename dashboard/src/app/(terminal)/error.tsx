"use client";

// Error boundary for the terminal/cockpit routes. Kept monospace to match the
// terminal shell; offers a retry without forcing a full reload.
import { useEffect } from "react";

export default function TerminalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center font-mono">
      <p className="text-sm font-semibold uppercase tracking-widest text-red-600">
        [ engine view error ]
      </p>
      <h1 className="mt-3 text-base font-semibold text-zinc-900 dark:text-zinc-50">
        This terminal view failed to render
      </h1>
      <p className="mt-2 max-w-md text-sm text-zinc-500 dark:text-zinc-400">
        If this is the cockpit, the engine API may be unreachable — confirm it is
        running on :8787, then retry. Otherwise see the message below.
      </p>
      {error.message ? (
        <pre className="mt-3 max-w-md overflow-x-auto whitespace-pre-wrap rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-left text-xs text-red-300">
          {error.message}
        </pre>
      ) : null}
      {error.digest ? (
        <p className="mt-3 text-xs text-zinc-400 dark:text-zinc-600">
          ref: {error.digest}
        </p>
      ) : null}
      <button
        onClick={() => reset()}
        className="mt-6 inline-flex h-9 items-center justify-center rounded-md border border-zinc-700 px-4 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800"
      >
        Retry
      </button>
    </div>
  );
}
