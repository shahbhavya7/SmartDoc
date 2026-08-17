"use client";

/**
 * One question, expanded: everything the harness measured, and why it landed
 * where it did.
 *
 * The ordering is deliberate. The VERDICT and the reason come first, then the
 * two answers side by side, then the individual checks. Someone opening a failed
 * row wants "why did this fail" answered in the first line, not after scrolling
 * past two paragraphs of answer text.
 *
 * Checks that did not apply are omitted rather than shown as "n/a" — a
 * consistency row has no exact-value check, and printing an empty one for it
 * just adds noise to every card.
 */

import { FileText } from "lucide-react";

import {
  ScoreBar,
  Verdict,
  categoryLabel,
} from "@/components/evaluation/scoring-primitives";
import type { EvalQuestionResult } from "@/lib/types";
import { cn } from "@/lib/utils";

function Field({
  label,
  children,
  tone = "default",
}: {
  label: string;
  children: React.ReactNode;
  tone?: "default" | "good" | "bad";
}) {
  return (
    <div
      className={cn(
        "rounded-xl p-3.5",
        tone === "good"
          ? "bg-emerald-400/[0.07]"
          : tone === "bad"
            ? "bg-rose-400/[0.07]"
            : "glass-raised",
      )}
    >
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <div className="mt-1.5 text-sm leading-relaxed">{children}</div>
    </div>
  );
}

function ValuePill({ value, tone }: { value: string; tone: "good" | "bad" }) {
  return (
    <code
      className={cn(
        "rounded px-1.5 py-0.5 font-mono text-xs",
        tone === "good"
          ? "bg-emerald-400/15 text-emerald-300"
          : "bg-rose-400/15 text-rose-300",
      )}
    >
      {value}
    </code>
  );
}

export function QuestionDetail({
  result,
  threshold,
}: {
  result: EvalQuestionResult;
  threshold: number;
}) {
  const isEdge = result.category.startsWith("input_edge_");
  const isConsistency = result.category === "consistency_pair";

  return (
    <div className="space-y-3 border-t border-white/[0.06] bg-black/20 p-4 sm:p-5">
      {/* Verdict first: why did this land the way it did? */}
      <div className="flex flex-wrap items-center gap-2">
        <Verdict passed={result.passed} />
        <span className="text-xs text-muted-foreground">
          {categoryLabel(result.category)}
        </span>
        {result.fail_reason ? (
          <span className="text-xs text-rose-300">— {result.fail_reason}</span>
        ) : null}
        <span className="ml-auto text-[11px] tabular-nums text-muted-foreground/70">
          {result.latency_ms > 0
            ? `${(result.latency_ms / 1000).toFixed(1)}s`
            : null}
          {result.query_type ? ` · ${result.query_type}` : null}
        </span>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Edge cases: behaviour, not similarity                             */}
      {/* ---------------------------------------------------------------- */}
      {isEdge ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Field label="Expected behaviour">
            <span className="text-muted-foreground">
              {result.expected_behavior}
            </span>
          </Field>
          <Field
            label="What actually happened"
            tone={result.passed ? "good" : "bad"}
          >
            <span className="break-words font-mono text-xs">
              {result.actual_behavior}
            </span>
          </Field>
        </div>
      ) : isConsistency ? (
        /* ---------------------------------------------------------------- */
        /* Consistency: the two runs compared to each other                  */
        /* ---------------------------------------------------------------- */
        <>
          <div className="grid gap-3 lg:grid-cols-2">
            <Field label="First run">{result.run1_answer || "—"}</Field>
            <Field label="Second run (5 minutes later)">
              {result.run2_answer || "—"}
            </Field>
          </div>
          <div className="glass-raised rounded-xl p-3.5">
            <div className="flex items-center justify-between gap-3">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                How alike the two runs were
              </p>
              <Verdict
                passed={result.stable}
                label={result.stable ? "Stable" : "Drifted"}
              />
            </div>
            <ScoreBar
              className="mt-2"
              value={result.self_similarity ?? 0}
              threshold={null}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              Compared against each other, not against the expected answer — this
              measures whether the system is stable, not whether it is right.
            </p>
          </div>
        </>
      ) : (
        /* ---------------------------------------------------------------- */
        /* Everything else: expected vs actual, then the checks              */
        /* ---------------------------------------------------------------- */
        <>
          <div className="grid gap-3 lg:grid-cols-2">
            <Field label="Known-correct answer">
              {result.expected_answer}
            </Field>
            <Field
              label="What the system answered"
              tone={result.passed ? "good" : "bad"}
            >
              {result.generated_answer || (
                <span className="text-muted-foreground">(no answer returned)</span>
              )}
            </Field>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {/* Similarity */}
            <div className="glass-raised rounded-xl p-3.5">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Meaning match
              </p>
              <ScoreBar
                className="mt-2"
                value={result.similarity}
                threshold={threshold}
              />
              <p className="mt-1.5 text-[11px] text-muted-foreground/75">
                Needs {threshold.toFixed(2)} or higher to pass.
              </p>
            </div>

            {/* Exact values */}
            {result.exact_match_applicable ? (
              <div className="glass-raised rounded-xl p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Exact values
                  </p>
                  <Verdict passed={result.exact_match_passed} />
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  {result.expected_values.map((value) => (
                    <ValuePill
                      key={value}
                      value={value}
                      tone={
                        result.missing_values.includes(value) ? "bad" : "good"
                      }
                    />
                  ))}
                </div>
                {result.missing_values.length > 0 ? (
                  <p className="mt-1.5 text-[11px] text-rose-300">
                    Missing from the answer:{" "}
                    {result.missing_values.join(", ")}
                  </p>
                ) : (
                  <p className="mt-1.5 text-[11px] text-muted-foreground/75">
                    All required values present.
                  </p>
                )}
              </div>
            ) : null}

            {/* Completeness */}
            {result.completeness_applicable ? (
              <div className="glass-raised rounded-xl p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Items found
                  </p>
                  <span className="text-xs font-semibold tabular-nums">
                    {result.items_found}/{result.items_expected}
                  </span>
                </div>
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      result.items_found === result.items_expected
                        ? "bg-emerald-400/80"
                        : "bg-amber-400/80",
                    )}
                    style={{
                      width: `${
                        result.items_expected
                          ? (result.items_found / result.items_expected) * 100
                          : 0
                      }%`,
                    }}
                  />
                </div>
                {result.missing_items.length > 0 ? (
                  <p className="mt-1.5 text-[11px] text-amber-300">
                    Missed: {result.missing_items.join(", ")}
                  </p>
                ) : (
                  <p className="mt-1.5 text-[11px] text-muted-foreground/75">
                    Nothing missing.
                  </p>
                )}
              </div>
            ) : null}

            {/* Declined correctly */}
            {result.correctly_declined !== null ? (
              <div className="glass-raised rounded-xl p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Said &ldquo;I don&rsquo;t know&rdquo;
                  </p>
                  <Verdict
                    passed={result.correctly_declined}
                    label={result.correctly_declined ? "Yes" : "No"}
                  />
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/75">
                  {result.correctly_declined
                    ? "Correctly declined instead of inventing an answer."
                    : "Answered when it should have declined — this is a made-up answer."}
                </p>
              </div>
            ) : null}

            {/* Table rendering */}
            {result.rendered_table !== null ? (
              <div className="glass-raised rounded-xl p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Rendered a table
                  </p>
                  <Verdict
                    passed={result.rendered_table}
                    label={result.rendered_table ? "Yes" : "No"}
                  />
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/75">
                  Comparisons are expected to come back as a table, not a
                  paragraph.
                </p>
              </div>
            ) : null}
          </div>

          {/* Citations */}
          {result.expected_source || result.retrieved_sources.length > 0 ? (
            <div className="glass-raised flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl p-3.5">
              <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                <FileText className="size-3.5" aria-hidden />
                Sources
              </span>
              {result.expected_source ? (
                <span className="text-xs text-muted-foreground">
                  Expected:{" "}
                  <code className="font-mono text-[11px]">
                    {result.expected_source}
                  </code>
                  {result.cited_expected_source !== null ? (
                    <Verdict
                      className="ml-1.5"
                      passed={result.cited_expected_source}
                      label={result.cited_expected_source ? "cited" : "not cited"}
                    />
                  ) : null}
                </span>
              ) : null}
              <span className="text-xs text-muted-foreground">
                Actually used:{" "}
                {result.retrieved_sources.length > 0 ? (
                  <span className="font-mono text-[11px]">
                    {Array.from(new Set(result.retrieved_sources)).join(", ")}
                  </span>
                ) : (
                  <span className="text-rose-300">nothing retrieved</span>
                )}
              </span>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
