"use client";

/**
 * The citations under an answer.
 *
 * These come from retrieval metadata, not from the model the document, page,
 * and section are structural facts about which passage was read, which is why
 * they cannot be hallucinated. Presenting them as the primary way to check an
 * answer is the point, so each one shows the actual snippet that entered the
 * prompt rather than just a filename.
 *
 * Collapsed by default: an answer with six citations would otherwise bury the
 * next question below a wall of quoted text.
 */

import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";

import { displayFilename } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";

function pageLabel(source: Source): string {
  if (source.page_end && source.page_end !== source.page) {
    return `pp. ${source.page}–${source.page_end}`;
  }
  return `p. ${source.page}`;
}

export function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  if (sources.length === 0) return null;

  // A refusal has no sources, so anything here is genuinely cited material.
  const documentCount = new Set(sources.map((source) => source.source)).size;

  return (
    <div className="mt-4 border-t border-white/[0.08] pt-3">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="group flex w-full items-center gap-2 rounded-lg text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-primary/90">
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
        <span className="text-xs text-muted-foreground">
          across {documentCount} {documentCount === 1 ? "document" : "documents"}
        </span>
        <ChevronDown
          className={cn(
            "ml-auto size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {/* Always render the compact chip row: even collapsed, the user should see
          WHICH documents the answer came from without an extra click. */}
      <ul className="mt-2.5 flex flex-wrap gap-1.5">
        {sources.map((source, index) => (
          <li key={`${source.source}-${source.page}-${index}`}>
            <span className="flex items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.05] px-2 py-1 text-[11px] text-muted-foreground">
              <FileText className="size-3 shrink-0 text-primary/80" aria-hidden />
              <span className="max-w-[16rem] truncate text-foreground/85">
                {displayFilename(source.source)}
              </span>
              <span className="shrink-0 tabular-nums">{pageLabel(source)}</span>
            </span>
          </li>
        ))}
      </ul>

      {open ? (
        <ul className="mt-3 space-y-2.5">
          {sources.map((source, index) => (
            <li
              key={`detail-${source.source}-${source.page}-${index}`}
              className="animate-rise rounded-lg border border-white/[0.08] bg-black/25 p-3"
            >
              <p className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs">
                <span className="font-medium text-foreground/90">{source.source}</span>
                <span className="tabular-nums text-muted-foreground">
                  {pageLabel(source)}
                </span>
                {source.section ? (
                  <span className="text-muted-foreground">· {source.section}</span>
                ) : null}
              </p>
              <p className="mt-2 border-l-2 border-primary/35 pl-2.5 text-[13px] leading-relaxed text-foreground/75">
                {source.snippet}
              </p>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
