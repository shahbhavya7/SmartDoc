"use client";

/**
 * "How this is scored", in plain English.
 *
 * The wording comes from the BACKEND (`GET /eval/method`), not from constants
 * here. That is deliberate: an explanation of the scoring rules that lives in
 * the frontend drifts away from the rules themselves the first time someone
 * changes the scorer, and a confidently wrong explanation of a metric is worse
 * than no explanation.
 *
 * The calibration section is the part that earns the page its "transparent"
 * claim: it shows the measured distributions the threshold was derived from, so
 * a reader can see WHY the bar sits where it does instead of taking 0.64 on
 * faith.
 */

import { AlertTriangle, ArrowRight, Ruler } from "lucide-react";

import { ScoreBar } from "@/components/evaluation/scoring-primitives";
import { Skeleton } from "@/components/ui/skeleton";
import type { EvalCalibration, EvalMethod } from "@/lib/types";

/** Labels for the corruption types calibration measures. */
const KIND_LABELS: Record<string, string> = {
  wrong_number: "Wrong number (20 → 28)",
  wrong_entity: "Wrong entity (Tier 3 → Tier 1)",
  wrong_section: "Answer to a different question",
};

export function MethodPanel({
  method,
  calibration,
  loading,
}: {
  method: EvalMethod | null;
  calibration: EvalCalibration | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="glass space-y-4 rounded-2xl p-6">
        <Skeleton className="h-5 w-56 bg-white/10" />
        <Skeleton className="h-20 w-full bg-white/10" />
        <Skeleton className="h-20 w-full bg-white/10" />
      </div>
    );
  }
  if (!method) return null;

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- */}
      {/* The five steps                                                    */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass rounded-2xl p-6">
        <h2 className="text-base font-semibold">How a question is scored</h2>
        <p className="mt-1.5 max-w-3xl text-sm text-muted-foreground">
          {method.summary}
        </p>

        <ol className="mt-5 grid gap-3 lg:grid-cols-5">
          {method.steps.map((step, index) => (
            <li
              key={step.title}
              className="glass-raised relative rounded-xl p-4"
            >
              <span className="text-[11px] font-semibold tabular-nums text-primary">
                Step {index + 1}
              </span>
              <h3 className="mt-1 text-sm font-medium">{step.title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                {step.body}
              </p>
              {index < method.steps.length - 1 ? (
                <ArrowRight
                  className="absolute -right-2.5 top-1/2 hidden size-4 -translate-y-1/2 text-muted-foreground/40 lg:block"
                  aria-hidden
                />
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Why the exact-match guard exists                                  */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass-accent rounded-2xl p-6">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-primary/20 text-primary">
            <AlertTriangle className="size-4" aria-hidden />
          </span>
          <div className="min-w-0">
            <h2 className="text-base font-semibold">
              Why meaning-matching alone is not enough
            </h2>
            <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-muted-foreground">
              {method.why_exact_match}
            </p>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="glass-raised rounded-xl p-3.5">
                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Correct answer
                </p>
                <p className="mt-1.5 text-sm">
                  Standard band employees accrue{" "}
                  <span className="rounded bg-emerald-400/15 px-1 font-semibold text-emerald-300">
                    20
                  </span>{" "}
                  days of paid annual leave.
                </p>
              </div>
              <div className="glass-raised rounded-xl p-3.5">
                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Wrong answer but 0.98 similar
                </p>
                <p className="mt-1.5 text-sm">
                  Standard band employees accrue{" "}
                  <span className="rounded bg-rose-400/15 px-1 font-semibold text-rose-300">
                    28
                  </span>{" "}
                  days of paid annual leave.
                </p>
              </div>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Only the exact-value check separates these two. It can fail an
              answer on its own, however high the similarity score.
            </p>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* The metrics                                                       */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass rounded-2xl p-6">
        <h2 className="text-base font-semibold">What each number means</h2>
        <dl className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {method.metrics.map((metric) => (
            <div key={metric.name} className="glass-raised rounded-xl p-4">
              <dt className="text-sm font-medium">{metric.name}</dt>
              <dd className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                {metric.plain}
              </dd>
              <dd className="mt-2 border-t border-white/[0.06] pt-2 text-[11px] leading-relaxed text-muted-foreground/75">
                {metric.detail}
              </dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 rounded-lg bg-white/[0.04] p-3 text-xs leading-relaxed text-muted-foreground">
          {method.categories_note}
        </p>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Calibration: where the threshold came from                        */}
      {/* ---------------------------------------------------------------- */}
      {calibration ? (
        <CalibrationPanel calibration={calibration} />
      ) : null}
    </div>
  );
}

function CalibrationPanel({ calibration }: { calibration: EvalCalibration }) {
  const { distributions } = calibration;
  const threshold = calibration.current_threshold;

  const rows = [
    {
      key: "correct",
      label: "Correct answers",
      tone: "good" as const,
      mean: distributions.correct.mean,
      n: distributions.correct.n,
    },
    ...Object.entries(distributions.wrong_by_kind).map(([kind, stats]) => ({
      key: kind,
      label: KIND_LABELS[kind] ?? kind,
      tone: "bad" as const,
      mean: stats.mean,
      n: stats.n,
    })),
  ];

  return (
    <section className="glass rounded-2xl p-6">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-white/[0.07] text-muted-foreground">
          <Ruler className="size-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold">
            Where the pass mark came from
          </h2>
          <p className="mt-1.5 max-w-3xl text-sm text-muted-foreground">
            The threshold was not chosen by intuition. Every question was scored
            against its correct answer, and then against deliberately wrong ones,
            so the two sets of scores could be compared directly. The pass mark
            sits where they separate.
          </p>

          <div className="mt-5 space-y-2.5">
            {rows.map((row) => (
              <div
                key={row.key}
                className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 sm:grid-cols-[220px_minmax(0,1fr)_auto]"
              >
                <div className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className={
                      row.tone === "good"
                        ? "size-2 shrink-0 rounded-full bg-emerald-400"
                        : "size-2 shrink-0 rounded-full bg-rose-400/70"
                    }
                  />
                  <span className="truncate text-xs text-muted-foreground">
                    {row.label}
                  </span>
                </div>
                <div className="col-span-2 sm:col-span-1">
                  <ScoreBar value={row.mean} threshold={threshold} />
                </div>
                <span className="hidden text-[11px] tabular-nums text-muted-foreground/70 sm:block">
                  n={row.n}
                </span>
              </div>
            ))}
          </div>

          <p className="mt-4 rounded-lg bg-white/[0.04] p-3 text-xs leading-relaxed text-muted-foreground">
            The white tick marks the pass threshold ({threshold.toFixed(2)}).
            Notice that <strong className="text-foreground">wrong numbers and
            wrong entities score about as high as correct answers</strong> no
            threshold can separate them, which is exactly why the exact-value
            check is mandatory. What the threshold does catch is an answer to a
            different question, which scores far lower.
          </p>
        </div>
      </div>
    </section>
  );
}
