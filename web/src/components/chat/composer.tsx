"use client";

/**
 * The question input.
 *
 * Enter sends and Shift+Enter breaks the line — a question about a policy is
 * often one line, and the alternative (Enter always breaking) makes the common
 * case need a mouse.
 *
 * The textarea autogrows to a cap rather than scrolling from the first line, so a
 * long multi-part question stays fully visible while being written, and the panel
 * never grows tall enough to push the conversation off screen.
 */

import { useEffect, useRef } from "react";
import { CornerDownLeft, Loader2 } from "lucide-react";

import { UploadControl } from "@/components/upload-button";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

/** Mirrors MAX_QUESTION_CHARS in backend/config.py. */
const MAX_QUESTION_CHARS = 4000;
const MAX_TEXTAREA_PX = 200;

export function Composer({
  value,
  onChange,
  onSubmit,
  onUploaded,
  busy,
  disabled = false,
  placeholder = "Ask a question about your documents…",
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onUploaded: () => void;
  busy: boolean;
  disabled?: boolean;
  placeholder?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Autogrow. Height is reset to `auto` first so the box also SHRINKS when text
  // is deleted, rather than staying at its high-water mark.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, MAX_TEXTAREA_PX)}px`;
  }, [value]);

  // Return focus after a turn completes so the next question can just be typed.
  useEffect(() => {
    if (!busy && !disabled) textareaRef.current?.focus();
  }, [busy, disabled]);

  const tooLong = value.length > MAX_QUESTION_CHARS;
  const canSend = value.trim().length > 0 && !busy && !disabled && !tooLong;

  function onKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (canSend) onSubmit();
    }
  }

  return (
    <div className="glass-raised rounded-2xl p-2">
      <div className="flex items-end gap-1.5">
        <UploadControl
          variant="compact"
          onUploaded={onUploaded}
          disabled={busy}
          className="mb-0.5"
        />

        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
          rows={1}
          placeholder={placeholder}
          aria-label="Your question"
          className="min-h-10 flex-1 resize-none border-0 bg-transparent px-1.5 py-2 text-[15px] shadow-none focus-visible:ring-0 dark:bg-transparent"
        />

        <Button
          type="button"
          size="icon-lg"
          onClick={onSubmit}
          disabled={!canSend}
          aria-label="Send question"
          className="mb-0.5 size-9 shrink-0"
        >
          {busy ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <CornerDownLeft className="size-4" aria-hidden />
          )}
        </Button>
      </div>

      <div className="flex items-center justify-between gap-3 px-2 pb-0.5 pt-1">
        <p className="text-[11px] text-muted-foreground">
          <kbd className="rounded border border-white/12 bg-white/[0.06] px-1 py-px font-sans">
            Enter
          </kbd>{" "}
          to send ·{" "}
          <kbd className="rounded border border-white/12 bg-white/[0.06] px-1 py-px font-sans">
            Shift+Enter
          </kbd>{" "}
          for a new line
        </p>
        {/* The counter appears only when it starts to matter, so it is a warning
            rather than permanent chrome. */}
        {value.length > MAX_QUESTION_CHARS * 0.8 ? (
          <p
            className={`text-[11px] tabular-nums ${
              tooLong ? "font-medium text-red-300" : "text-muted-foreground"
            }`}
          >
            {value.length.toLocaleString()} / {MAX_QUESTION_CHARS.toLocaleString()}
          </p>
        ) : null}
      </div>
    </div>
  );
}
