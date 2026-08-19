"use client";

/**
 * table-router branch: a three-way switch for how the next question is
 * answered -- "Auto" (default: backend.router_graph classifies the question
 * and picks Hybrid or ColPali itself), or a manual override to either single
 * pipeline for direct A/B testing (the colpali branch's original toggle,
 * unchanged in behaviour). Selection is per-request (see `api.ask`'s
 * `backend` param -- "auto" simply omits it), so switching mid-session needs
 * no restart and takes effect on the very next question.
 *
 * Deliberately a plain segmented control rather than the DropdownMenu
 * primitive elsewhere in this app: a small, fixed set of choices is more
 * scannable as always-visible buttons than as a menu that has to be opened
 * to see the current value.
 */

import { Eye, FileText, Wand2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RetrievalMode } from "@/lib/types";

const OPTIONS: { value: RetrievalMode; label: string; icon: typeof FileText }[] = [
  { value: "auto", label: "Auto", icon: Wand2 },
  { value: "hybrid", label: "Hybrid RAG", icon: FileText },
  { value: "colpali", label: "ColPali (visual)", icon: Eye },
];

export function BackendToggle({
  value,
  onChange,
  disabled = false,
}: {
  value: RetrievalMode;
  onChange: (value: RetrievalMode) => void;
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
