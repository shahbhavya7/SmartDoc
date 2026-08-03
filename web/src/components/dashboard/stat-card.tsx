"use client";

import type { LucideIcon } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

/**
 * One headline number.
 *
 * The figure gets the champagne gradient and the largest type on the page; the
 * label and hint stay muted, so a row of these reads as four values rather than
 * four boxes. `tabular-nums` keeps the digits from reflowing when a count
 * changes after an upload or delete.
 */
export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  loading = false,
  accent = false,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: LucideIcon;
  loading?: boolean;
  /** Tints the tile; reserved for the single most important number. */
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl p-5 transition-transform duration-200 hover:-translate-y-0.5",
        accent ? "glass-accent" : "glass",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-[13px] font-medium text-muted-foreground">{label}</p>
        <span
          className={cn(
            "grid size-8 shrink-0 place-items-center rounded-lg",
            accent ? "bg-primary/20 text-primary" : "bg-white/[0.07] text-muted-foreground",
          )}
        >
          <Icon className="size-4" aria-hidden />
        </span>
      </div>

      {loading ? (
        <Skeleton className="mt-3 h-9 w-24 bg-white/10" />
      ) : (
        <p
          className={cn(
            "mt-2.5 text-3xl font-semibold tabular-nums tracking-tight",
            accent && "text-accent-gradient",
          )}
        >
          {value}
        </p>
      )}

      {hint ? (
        <p className="mt-1.5 text-xs text-muted-foreground/85">{hint}</p>
      ) : null}
    </div>
  );
}
