import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-zinc-900 text-zinc-50 shadow dark:bg-zinc-50 dark:text-zinc-900",
        secondary: "border-transparent bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50",
        destructive: "border-transparent bg-red-500 text-zinc-50 shadow",
        outline: "text-zinc-950 dark:text-zinc-50",
        ticker: "border-transparent bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
        positive: "border-transparent bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
        negative: "border-transparent bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
