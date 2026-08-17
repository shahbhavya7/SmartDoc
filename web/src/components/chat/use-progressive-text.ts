"use client";

/**
 * Reveals an answer progressively instead of dropping it in as one block.
 *
 * Why the reveal is client-side rather than a token stream from the model
 * ----------------------------------------------------------------------
 * SmartDoc's grounding stage reads the COMPLETE answer, and may regenerate it,
 * prune sentences from it, or withdraw it entirely to the refusal string
 * (DECISIONS.md C14). Tokens streamed straight from the model would therefore be
 * text the server has not yet decided to stand behind and retracting an answer
 * a user has already read is a worse experience than waiting for a verified one.
 *
 * So the network trip stays a single request for a verified answer, and the
 * reveal happens here. It starts at the same instant a token stream's first
 * useful token would have arrived (the answer is not knowable earlier), and the
 * text that appears is final: nothing on screen is ever taken back.
 *
 * The rate is length-aware a one-line refusal should not crawl, and a
 * three-paragraph synthesis should not take ten seconds so total reveal time
 * is bounded to roughly `MAX_MS` regardless of size.
 *
 * What gets revealed is a BLOCK-SAFE prefix, not a raw character slice. Answers
 * are now formatted to their content (Phase 4, Part A), and a raw prefix of a
 * markdown table is invalid markdown: it renders as a paragraph of literal pipe
 * characters that snaps into a table a few frames later. `blockSafeSlice`
 * clamps back to the last point where the markdown still parses, so a table
 * appears row by row and never as debris. See `lib/markdown-reveal.ts`.
 */

import { useEffect, useState } from "react";

import { blockSafeSlice } from "@/lib/markdown-reveal";

const MIN_CHARS_PER_SECOND = 420;
const MAX_MS = 1600;

/**
 * Progress is stored WITH the text it describes, so a changed `text` reads as
 * "nothing revealed yet" during render instead of needing a reset write. That is
 * what keeps this hook free of any synchronous state update inside its effect
 * the only writes happen inside a requestAnimationFrame callback.
 */
interface Progress {
  text: string;
  revealed: number;
}

export function useProgressiveText(
  text: string,
  /** False shows the text in full with no animation used for history. */
  animate: boolean,
): { visible: string; done: boolean } {
  const [progress, setProgress] = useState<Progress>({ text: "", revealed: 0 });

  useEffect(() => {
    // History renders fully from the derivation below; there is nothing to drive.
    if (!animate || !text) return;

    // Honour the OS setting rather than animating anyway at a reduced rate: a
    // typewriter effect is exactly the kind of motion this preference is for.
    const reducedMotion = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    )?.matches;

    if (reducedMotion) {
      const frame = requestAnimationFrame(() =>
        setProgress({ text, revealed: text.length }),
      );
      return () => cancelAnimationFrame(frame);
    }

    const perSecond = Math.max(MIN_CHARS_PER_SECOND, (text.length / MAX_MS) * 1000);
    const start = performance.now();
    let frame = 0;

    function step(now: number) {
      const elapsed = (now - start) / 1000;
      const next = Math.min(text.length, Math.ceil(elapsed * perSecond));
      setProgress({ text, revealed: next });
      if (next < text.length) frame = requestAnimationFrame(step);
    }
    frame = requestAnimationFrame(step);

    return () => cancelAnimationFrame(frame);
  }, [text, animate]);

  if (!animate) return { visible: text, done: true };

  const revealed = progress.text === text ? progress.revealed : 0;
  return {
    // `done` stays keyed to the character count, not to the clamped string: a
    // table held back for one more frame is still mid-reveal, and citations
    // must not attach until the answer is whole.
    visible: blockSafeSlice(text, revealed),
    done: revealed >= text.length,
  };
}
