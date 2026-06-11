import Link from "next/link";
import { Compass, LayoutDashboard } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
      <Compass className="h-10 w-10 text-zinc-300 dark:text-zinc-700" />
      <p className="mt-4 font-mono text-6xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
        404
      </p>
      <h1 className="mt-2 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
        Page not found
      </h1>
      <p className="mt-2 max-w-md text-sm text-zinc-500 dark:text-zinc-400">
        That page doesn&apos;t exist or has moved. The Decision Cockpit is the
        best place to start.
      </p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        <Link href="/cockpit" className={cn(buttonVariants({ variant: "default" }))}>
          <LayoutDashboard className="mr-2 h-4 w-4" />
          Go to Cockpit
        </Link>
        <Link href="/top" className={cn(buttonVariants({ variant: "outline" }))}>
          Browse TOP news
        </Link>
      </div>
    </div>
  );
}
