"use client";

// Root-level error boundary. Catches errors thrown in the root layout itself,
// so it must render its own <html>/<body> (it replaces the root layout).
// Segment-level errors are handled by the per-group error.tsx files instead.
import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface to the console / any attached logger; no PII in the message.
    console.error(error);
  }, [error]);

  return (
    <html lang="en">
      <body className="antialiased">
        <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
          <p className="font-mono text-sm font-semibold uppercase tracking-widest text-red-600">
            Application error
          </p>
          <h1 className="mt-3 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Something went wrong
          </h1>
          <p className="mt-2 max-w-md text-sm text-zinc-500 dark:text-zinc-400">
            The terminal hit an unexpected error and couldn&apos;t render this
            view. You can try again, or reload the app.
          </p>
          {error.digest ? (
            <p className="mt-3 font-mono text-xs text-zinc-400 dark:text-zinc-600">
              ref: {error.digest}
            </p>
          ) : null}
          <button
            onClick={() => reset()}
            className="mt-6 inline-flex h-9 items-center justify-center rounded-md bg-zinc-900 px-4 text-sm font-medium text-zinc-50 shadow transition-colors hover:bg-zinc-800 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
