"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Newspaper,
  TrendingUp,
  Star,
  Calendar,
  MessageSquare,
  Bell,
  Zap,
  Monitor,
  Gauge,
} from "lucide-react";

const navItems = [
  // The Decision Cockpit is the flagship engine surface — list it first.
  { href: "/cockpit", label: "Cockpit", icon: Gauge, accent: true },
  { href: "/top", label: "TOP", icon: Zap },
  { href: "/feed", label: "Feed", icon: Newspaper },
  { href: "/watchlist", label: "Watchlist", icon: Star },
  { href: "/calendar", label: "Calendar", icon: Calendar },
  { href: "/research", label: "Research", icon: MessageSquare },
  { href: "/terminal", label: "Terminal", icon: Monitor },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mx-auto flex h-14 max-w-7xl items-center px-4">
        <Link href="/cockpit" className="mr-8 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-blue-600" />
          <span className="font-semibold text-zinc-900 dark:text-zinc-50">
            YAKAR TERMINAL
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50"
                    : item.accent
                      ? "text-amber-700 hover:bg-amber-50 hover:text-amber-800 dark:text-amber-500 dark:hover:bg-amber-950/40"
                      : "text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-50"
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Link
            href="/feed"
            className="relative rounded-md p-2 text-zinc-600 hover:bg-zinc-50 dark:text-zinc-400 dark:hover:bg-zinc-900"
          >
            <Bell className="h-4 w-4" />
          </Link>
        </div>
      </div>
    </nav>
  );
}
