import { Skeleton } from "@/components/ui/skeleton";

// Suspense fallback for the terminal/cockpit routes. The (terminal) group
// layout applies the monospace `terminal-body` shell around this.
export default function Loading() {
  return (
    <div
      className="mx-auto max-w-7xl px-4 py-6"
      aria-busy="true"
      aria-label="Loading terminal"
    >
      <div className="flex items-center gap-3">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-6 w-24" />
        <Skeleton className="ml-auto h-6 w-28" />
      </div>
      <Skeleton className="mt-4 h-px w-full" />
      <div className="mt-4 space-y-2">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-full" />
        ))}
      </div>
    </div>
  );
}
