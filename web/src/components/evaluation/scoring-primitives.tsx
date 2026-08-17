"use client";

/**
 * Small presentational pieces shared across the evaluation page.
 *
 * The bar in `ScoreBar` is the one piece of real information design here: a
 * similarity score is meaningless without the threshold next to it, because
 * "0.68" reads as mediocre until you know the bar is at 0.64. So the threshold
 * is drawn as a marker ON the bar rather than printed as a separate number, and
 * the fill is coloured by which side of it the score landed.
 */

import { Check, Minus, X } from "lucide-react";

import { cn } from "@/lib/utils";

/** Human-readable name for a gold-set category slug. */
export const CATEGORY_LABELS: Record<string, string> = {
  fact_lookup: "Fact lookup",
  entity_specific: "Entity / code lookup",
  definitional: "Definitions",
  yes_no_justified: "Yes-no with reason",
  numeric_quantitative: "Numbers & quantities",
  conditional_scenario: "If-then scenarios",
  table_cell_lookup: "Table cell lookup",
  table_aggregation: "Table aggregation",
  multi_section_synthesis: "Across sections",
  enumeration: "List everything",
  document_summary: "Document summary",
  comparison: "Comparisons",
  procedural_ordered: "Ordered steps",
  out_of_scope_unrelated: "Out of scope (unrelated)",
  out_of_scope_plausible: "Out of scope (plausible)",
  out_of_scope_partial: "Half answerable",
  cross_document: "Across documents",
  input_edge_empty: "Edge: empty input",
  input_edge_long: "Edge: very long input",
  input_edge_nonenglish: "Edge: non-English",
  input_edge_gibberish: "Edge: gibberish",
  input_edge_injection: "Edge: prompt injection",
  new_document: "Newly uploaded document",
  consistency_pair: "Same question twice",
};

/** What each category is actually testing, in one sentence. */
export const CATEGORY_BLURBS: Record<string, string> = {
  fact_lookup: "A single fact stated plainly in one place.",
  entity_specific: "A specific code, name, or identifier — exact-match territory.",
  definitional: "What a term means, as the documents define it.",
  yes_no_justified: "A yes or no that has to say what it is based on.",
  numeric_quantitative: "A number that must be exactly right, not approximately.",
  conditional_scenario: "Applying a rule to a situation: if X, then what?",
  table_cell_lookup: "One cell from a table, found by row name or by ID.",
  table_aggregation: "Counting or listing across a whole table, not one row.",
  multi_section_synthesis: "An answer that has to combine two or more sections.",
  enumeration: "Listing every item of a kind — tests completeness, not luck.",
  document_summary: "Summarising a whole document.",
  comparison: "Comparing options — should come back as a table.",
  procedural_ordered: "Steps in the right order.",
  out_of_scope_unrelated: "Nothing to do with the documents. Must say it doesn't know.",
  out_of_scope_plausible:
    "Sounds like it could be in the documents but isn't — the highest risk of a made-up answer.",
  out_of_scope_partial:
    "Half answerable: must answer the supported half and flag the rest as missing.",
  cross_document: "Pulling together facts from two or more different documents.",
  input_edge_empty: "Blank input. Should be refused politely, not crash.",
  input_edge_long: "A very long question. Should answer or reject cleanly.",
  input_edge_nonenglish: "A question in another language.",
  input_edge_gibberish: "Random characters. Must not invent an answer.",
  input_edge_injection:
    "Text trying to override the system's instructions. Must be treated as an ordinary question.",
  new_document: "A document uploaded fresh, not part of the original test set.",
  consistency_pair: "The same question asked twice, to check the answer is stable.",
};

export function categoryLabel(slug: string): string {
  return CATEGORY_LABELS[slug] ?? slug.replace(/_/g, " ");
}

/** How a category's pass/fail is decided, in plain words. */
export const SCORING_MODE_LABELS: Record<string, string> = {
  "similarity+exact": "Meaning match + exact values",
  "similarity+completeness": "Meaning match + all items listed",
  "similarity+table": "Meaning match + rendered as a table",
  "refusal+exact": "Must decline the unsupported half + exact values",
  refusal: "Must correctly say it doesn't know",
  behaviour: "Must behave sensibly (not similarity)",
  self_similarity: "Two runs compared against each other",
};

/**
 * A similarity score drawn against the pass threshold.
 *
 * `threshold` is drawn as a tick on the track. Without it the number is not
 * interpretable — the whole point is whether the score cleared the bar.
 */
export function ScoreBar({
  value,
  threshold,
  className,
  showValue = true,
}: {
  value: number;
  threshold?: number | null;
  className?: string;
  showValue?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const passed = threshold == null || value >= threshold;

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative h-1.5 w-full min-w-16 overflow-hidden rounded-full bg-white/10">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            passed ? "bg-emerald-400/80" : "bg-amber-400/80",
          )}
          style={{ width: `${pct}%` }}
        />
        {threshold != null ? (
          <span
            aria-hidden
            title={`Pass threshold ${threshold.toFixed(2)}`}
            className="absolute top-0 h-full w-px bg-white/70"
            style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
          />
        ) : null}
      </div>
      {showValue ? (
        <span className="w-10 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
          {value.toFixed(2)}
        </span>
      ) : null}
    </div>
  );
}

/** Pass / fail / not-applicable pill. */
export function Verdict({
  passed,
  label,
  className,
}: {
  passed: boolean | null;
  label?: string;
  className?: string;
}) {
  const Icon = passed === null ? Minus : passed ? Check : X;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
        passed === null
          ? "bg-white/[0.06] text-muted-foreground"
          : passed
            ? "bg-emerald-400/12 text-emerald-300"
            : "bg-rose-400/12 text-rose-300",
        className,
      )}
    >
      <Icon className="size-3" aria-hidden />
      {label ?? (passed === null ? "n/a" : passed ? "Pass" : "Fail")}
    </span>
  );
}

/** A pass-rate bar for a category row. Flags anything under 80%. */
export function PassRateBar({ rate }: { rate: number }) {
  const low = rate < 80;
  return (
    <div className="flex items-center gap-2.5">
      <div className="h-1.5 w-full min-w-20 overflow-hidden rounded-full bg-white/10">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            rate === 100
              ? "bg-emerald-400/80"
              : low
                ? "bg-rose-400/80"
                : "bg-amber-400/80",
          )}
          style={{ width: `${Math.max(0, Math.min(100, rate))}%` }}
        />
      </div>
      <span
        className={cn(
          "w-12 shrink-0 text-right text-xs tabular-nums",
          low ? "text-rose-300" : "text-muted-foreground",
        )}
      >
        {rate.toFixed(0)}%
      </span>
    </div>
  );
}
