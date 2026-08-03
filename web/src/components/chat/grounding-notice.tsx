"use client";

/**
 * Surfaces the grounding verdict when — and only when — it means something.
 *
 * The backend verifies every answer's claims against the retrieved context and
 * remediates before responding: it regenerates, or prunes the offending
 * sentences, or withdraws to the refusal. So a verdict of `faithful` needs no
 * badge; saying "verified" on every answer trains the user to ignore the one
 * place it matters.
 *
 * Two cases are worth showing:
 *
 * **`unsupported_claims` on a returned answer.** This is the "declined" path:
 * remediation found a claim the context does not support but pruning it would
 * have cost supported content, so the original answer was returned with the flag
 * attached. That flag is exactly the thing a UI must not silently drop.
 *
 * **`removed_claims`.** Sentences were pruned. The user is reading a deliberately
 * shortened answer, and should know why it may feel incomplete.
 *
 * `unverified_numbers` is deliberately NOT surfaced as a warning. A legitimately
 * derived figure ("a difference of eight days" from 20 and 28) lands there too,
 * so flagging it would cry wolf on correct arithmetic.
 */

import { ShieldAlert, Scissors } from "lucide-react";

import type { Grounding } from "@/lib/types";

export function GroundingNotice({ grounding }: { grounding: Grounding | null }) {
  if (!grounding || !grounding.checked) return null;

  const unsupported = grounding.unsupported_claims ?? [];
  const removed = grounding.removed_claims ?? [];

  if (unsupported.length === 0 && removed.length === 0) return null;

  return (
    <div className="mt-3.5 space-y-2">
      {unsupported.length > 0 ? (
        <div className="rounded-lg border border-amber-400/25 bg-amber-400/[0.07] p-3">
          <p className="flex items-center gap-2 text-xs font-semibold text-amber-200/95">
            <ShieldAlert className="size-3.5 shrink-0" aria-hidden />
            Not confirmed by your documents
          </p>
          <ul className="mt-2 space-y-1.5 text-[13px] leading-relaxed text-amber-100/80">
            {unsupported.map((claim, index) => (
              <li key={index} className="border-l-2 border-amber-400/30 pl-2.5">
                {claim}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-amber-100/60">
            The rest of the answer verified against the cited passages. Treat the
            above as unverified.
          </p>
        </div>
      ) : null}

      {removed.length > 0 ? (
        <div className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
          <p className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
            <Scissors className="size-3.5 shrink-0" aria-hidden />
            {removed.length} {removed.length === 1 ? "sentence" : "sentences"} removed
          </p>
          <p className="mt-1.5 text-[12px] leading-relaxed text-muted-foreground">
            Content the cited passages did not support was pruned before this
            answer was shown, so it may read as incomplete.
          </p>
        </div>
      ) : null}
    </div>
  );
}
