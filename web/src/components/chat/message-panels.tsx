"use client";

/**
 * The message panels.
 *
 * The two roles are given deliberately different weight. A question is a compact
 * accent-tinted card aligned right; an answer is a full-width glass panel with
 * its citations and any grounding flags attached — because the answer is the
 * artifact the user has to be able to check, and the question is just the prompt
 * that produced it.
 */

import { Sparkles, User as UserIcon } from "lucide-react";

import { AnswerText } from "@/components/chat/answer-text";
import { GroundingNotice } from "@/components/chat/grounding-notice";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { useProgressiveText } from "@/components/chat/use-progressive-text";
import { cn } from "@/lib/utils";
import type { Grounding, Source } from "@/lib/types";

export function QuestionPanel({ content }: { content: string }) {
  return (
    <div className="flex animate-rise justify-end gap-3">
      <div className="glass-accent max-w-[min(46rem,88%)] rounded-2xl rounded-br-md px-4 py-3">
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-foreground">
          {content}
        </p>
      </div>
      <span
        className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-white/[0.07] text-muted-foreground"
        aria-hidden
      >
        <UserIcon className="size-4" />
      </span>
    </div>
  );
}

export function AnswerPanel({
  content,
  sources,
  grounding,
  queryType,
  /** True for the turn just answered: reveals the text progressively. */
  animate = false,
}: {
  content: string;
  sources: Source[];
  grounding: Grounding | null;
  queryType?: string;
  animate?: boolean;
}) {
  const { visible, done } = useProgressiveText(content, animate);

  // Citations and flags appear only once the prose has finished revealing —
  // attaching them to a half-written answer reads as a rendering glitch.
  const showMeta = done;

  return (
    <div className="flex animate-rise gap-3">
      <span
        className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-primary/15 text-primary"
        aria-hidden
      >
        <Sparkles className="size-4" />
      </span>

      <div className="glass min-w-0 max-w-[min(52rem,92%)] flex-1 rounded-2xl rounded-bl-md px-4 py-3.5">
        <AnswerText text={visible} />
        {!done ? (
          <span
            className="caret-blink ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] bg-primary align-middle"
            aria-hidden
          />
        ) : null}

        {showMeta ? (
          <>
            <GroundingNotice grounding={grounding} />
            <SourcesPanel sources={sources} />
            {queryType ? (
              <p className="mt-3 text-[11px] uppercase tracking-wider text-muted-foreground/60">
                {queryType.replace(/_/g, " ")}
              </p>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

/** The placeholder shown while a question is in flight. */
export function ThinkingPanel({ elapsedSeconds }: { elapsedSeconds: number }) {
  return (
    <div className="flex animate-rise gap-3">
      <span
        className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-primary/15 text-primary"
        aria-hidden
      >
        <Sparkles className="size-4 animate-pulse" />
      </span>
      <div className="glass rounded-2xl rounded-bl-md px-4 py-3.5">
        <p className="shimmer-text text-sm font-medium" role="status" aria-live="polite">
          {/* Honest about what is happening, and never claims a percentage: the
              pipeline is one server call and reports no intermediate progress. */}
          Searching your documents and verifying the answer…
        </p>
        {elapsedSeconds >= 4 ? (
          <p className="mt-1.5 text-xs tabular-nums text-muted-foreground">
            {elapsedSeconds}s · thorough questions take longer to check
          </p>
        ) : null}
      </div>
    </div>
  );
}

/** A failed turn, shown in place of an answer with the option to retry. */
export function ErrorPanel({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex animate-rise gap-3">
      <span
        className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-destructive/20 text-destructive"
        aria-hidden
      >
        <Sparkles className="size-4" />
      </span>
      <div
        role="alert"
        className={cn(
          "max-w-[min(52rem,92%)] rounded-2xl rounded-bl-md border border-destructive/30",
          "bg-destructive/[0.09] px-4 py-3.5",
        )}
      >
        <p className="text-sm font-medium text-red-200">{message}</p>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 rounded-md text-xs font-medium text-primary underline-offset-4 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring"
          >
            Ask again
          </button>
        ) : null}
      </div>
    </div>
  );
}
