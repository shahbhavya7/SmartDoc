/**
 * Clamps a partially-revealed answer to the last point where its markdown is
 * still well-formed.
 *
 * Why this exists
 * ---------------
 * The progressive reveal (see `use-progressive-text.ts`) hands `AnswerText` a
 * character-count prefix of the final answer. That was harmless while answers
 * were prose. Once the model is asked to format an answer to its content
 * (Phase 4, Part A), a raw prefix is routinely *invalid* markdown, and the
 * result is not a slightly-early render but a visibly broken one:
 *
 * - A GFM table is only a table once its `|---|---|` divider row exists. Before
 *   that, remark renders the header as a paragraph of literal pipe characters
 *   so a table would appear as `| Band | Annual leave |` text, then snap into a
 *   table a few frames later.
 * - Half a row (`| Senior | 28 d`) is a row with the wrong number of cells, so
 *   the table's column count jumps while it fills in.
 * - An unclosed ``` fence renders everything after it as a code block.
 * - A just-typed `**` renders as literal asterisks.
 *
 * The fix is to reveal at *structural* boundaries: never show a partial block
 * line, never show a table before it can render as one, and close what is still
 * open. Nothing is ever taken back the clamp only ever shows LESS than the
 * character count, and the final call returns the text verbatim.
 */

const TABLE_LINE = /^\s*\|/;
const TABLE_DIVIDER = /^\s*\|[\s:|-]+\|?\s*$/;
/** A block marker typed but not yet followed by any content. */
const MARKER_ONLY = /^\s*(?:[-*+]|\d+[.)]|#{1,6}|>|\|)\s*$/;

export function blockSafeSlice(text: string, revealed: number): string {
  if (revealed >= text.length) return text;
  if (revealed <= 0) return "";

  const slice = text.slice(0, revealed);
  const lines = slice.split("\n");

  // Drop the trailing line while it is still mid-block. Prose is left alone:
  // a half-written sentence is exactly what a reveal is supposed to look like.
  const endsOnNewline = slice.endsWith("\n");
  const partial = lines[lines.length - 1];
  if (!endsOnNewline && (TABLE_LINE.test(partial) || MARKER_ONLY.test(partial))) {
    lines.pop();
  }

  // Hide a table until its divider row has arrived, otherwise GFM renders the
  // header row as literal text.
  let end = lines.length;
  while (end > 0 && lines[end - 1].trim() === "") end--;
  let start = end;
  while (start > 0 && TABLE_LINE.test(lines[start - 1])) start--;
  if (start < end && !lines.slice(start, end).some((line) => TABLE_DIVIDER.test(line))) {
    lines.splice(start, end - start);
  }

  let out = lines.join("\n");

  // Close what is still open. An odd count means one unmatched opener; `***`
  // would miscount, but the answer prompt asks for bold lead-ins, not triples.
  // Inline backticks are counted with the fences removed first, or a closed
  // ```-block would read as three stray inline markers.
  if (count(out, "**") % 2 === 1) out += "**";
  if (count(out, "```") % 2 === 1) out += "\n```";
  else if (count(out.replaceAll("```", ""), "`") % 2 === 1) out += "`";

  return out;
}

function count(haystack: string, needle: string): number {
  let n = 0;
  let at = haystack.indexOf(needle);
  while (at !== -1) {
    n += 1;
    at = haystack.indexOf(needle, at + needle.length);
  }
  return n;
}
