"use client";

/**
 * colpali branch experiment: a two-way switch for which pipeline answers the
 * next question -- "Hybrid RAG" (the default text pipeline) or "ColPali
 * (visual)" (colpali_experiment, a separate page-image pipeline). Selection
 * is per-request (see `api.ask`'s `backend` param), so switching mid-session
 * needs no restart and takes effect on the very next question.
 *
 * Deliberately a plain two-segment control rather than the DropdownMenu
 * primitive elsewhere in this app: a binary choice is more scannable as two
 * always-visible buttons than as a menu that has to be opened to see the
 * current value.
 */

import { Eye, FileText } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RetrievalBackend } from "@/lib/types";

const OPTIONS: { value: RetrievalBackend; label: string; icon: typeof FileText }[] = [
  { value: "hybrid", label: "Hybrid RAG", icon: FileText },
  { value: "colpali", label: "ColPali (visual)", icon: Eye },
];

export function BackendToggle({
  value,
  onChange,
  disabled = false,
}: {
  value: RetrievalBackend;
  onChange: (value: RetrievalBackend) => void;
  disabled?: boolean;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Answer using"
      className="inline-flex items-center gap-0.5 rounded-lg border border-white/10 bg-white/[0.04] p-0.5"
    >
      {OPTIONS.map((option) => {
        const active = option.value === value;
        const Icon = option.icon;
        return (
          <Button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            variant={active ? "secondary" : "ghost"}
            size="xs"
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={cn("gap-1", !active && "text-muted-foreground")}
          >
            <Icon aria-hidden />
            {option.label}
          </Button>
        );
      })}
    </div>
  );
}
