"use client";

// Error boundary for the news-app routes. Rendered inside the (main) layout,
// so the Nav remains usable — the user can navigate away even if this view
// failed to load.
import { useEffect } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function MainError({
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
    <div className="flex min-h-[50vh] flex-col items-center justify-center px-4 text-center">
      <AlertTriangle className="h-9 w-9 text-amber-500" />
      <h1 className="mt-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
        Couldn&apos;t load this view
      </h1>
      <p className="mt-2 max-w-md text-sm text-zinc-500 dark:text-zinc-400">
        Something went wrong fetching this page. This is usually transient —
        try again, or pick another section from the nav above.
      </p>
      {error.digest ? (
        <p className="mt-3 font-mono text-xs text-zinc-400 dark:text-zinc-600">
          ref: {error.digest}
        </p>
      ) : null}
      <Button onClick={() => reset()} className="mt-6">
        Try again
      </Button>
    </div>
  );
}
