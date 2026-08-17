"use client";

/**
 * One evaluation run, rendered in full.
 *
 * Three layers, in the order someone actually reads them:
 *   1. the headline numbers,
 *   2. the per-category breakdown, weakest first — the question "what should I
 *      fix" is answered by ordering, not by making the reader scan,
 *   3. every individual question, filterable, each expandable into the full
 *      scoring detail.
 *
 * Every question is listed, not just the failures. A page that only shows what
 * broke cannot be checked for the thing that matters most here — that the
 * passes were earned rather than scored generously.
 */

import { useMemo, useState } from "react";
import { ChevronRight, Search } from "lucide-react";

import { QuestionDetail } from "@/components/evaluation/question-detail";
import {
  CATEGORY_BLURBS,
  PassRateBar,
  Verdict,
  categoryLabel,
} from "@/components/evaluation/scoring-primitives";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { EvalRun } from "@/lib/types";

type Filter = "all" | "failed" | "passed";

export function RunResults({
  run,
  /**
   * True while a PDF is being produced. Every question is rendered expanded,
   * because a printed report whose detail is collapsed behind a chevron is
   * useless -- the detail IS the report. Interactive chrome is hidden by the
   * print stylesheet via `data-print="hide"`.
   */
  printing = false,
}: {
  run: EvalRun;
  printing?: boolean;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [category, setCategory] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const threshold = run.meta.threshold;

  // Weakest categories first: the report's job is to point at what to fix.
  const categories = useMemo(
    () =>
      Object.entries(run.by_category).sort(
        (a, b) => a[1].pass_rate - b[1].pass_rate,
      ),
    [run.by_category],
  );

  const visible = useMemo(() => {
    // A PDF is a record of the whole run, so on-screen filters are ignored
    // while printing -- otherwise the file silently documents a subset and
    // nothing in it says so.
    if (printing) return run.results;

    const needle = search.trim().toLowerCase();
    return run.results.filter((result) => {
      if (filter === "failed" && result.passed) return false;
      if (filter === "passed" && !result.passed) return false;
      if (category && result.category !== category) return false;
      if (
        needle &&
        !result.question.toLowerCase().includes(needle) &&
        !result.generated_answer.toLowerCase().includes(needle) &&
        !result.id.toLowerCase().includes(needle)
      ) {
        return false;
      }
      return true;
    });
  }, [run.results, filter, category, search, printing]);

  const failed = run.summary.total - run.summary.passed;

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- */}
      {/* Headline                                                          */}
      {/* ---------------------------------------------------------------- */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div data-print="keep" className="glass-accent rounded-2xl p-5">
          <p className="text-[13px] font-medium text-muted-foreground">
            Questions passed
          </p>
          <p className="mt-2.5 text-3xl font-semibold tabular-nums tracking-tight text-accent-gradient">
            {run.summary.passed}/{run.summary.total}
          </p>
          <p className="mt-1.5 text-xs text-muted-foreground/85">
            {run.summary.pass_rate.toFixed(1)}% pass rate
          </p>
        </div>
        <div data-print="keep" className="glass rounded-2xl p-5">
          <p className="text-[13px] font-medium text-muted-foreground">
            Average meaning match
          </p>
          <p className="mt-2.5 text-3xl font-semibold tabular-nums tracking-tight">
            {run.summary.mean_similarity?.toFixed(3) ?? "—"}
          </p>
          <p className="mt-1.5 text-xs text-muted-foreground/85">
            Pass mark is {threshold.toFixed(2)}
          </p>
        </div>
        <div data-print="keep" className="glass rounded-2xl p-5">
          <p className="text-[13px] font-medium text-muted-foreground">
            Needs attention
          </p>
          <p
            className={cn(
              "mt-2.5 text-3xl font-semibold tabular-nums tracking-tight",
              failed > 0 && "text-rose-300",
            )}
          >
            {failed}
          </p>
          <p className="mt-1.5 text-xs text-muted-foreground/85">
            {failed === 0 ? "Everything passed" : "questions did not pass"}
          </p>
        </div>
        <div data-print="keep" className="glass rounded-2xl p-5">
          <p className="text-[13px] font-medium text-muted-foreground">
            Categories covered
          </p>
          <p className="mt-2.5 text-3xl font-semibold tabular-nums tracking-tight">
            {categories.length}
          </p>
          <p className="mt-1.5 text-xs text-muted-foreground/85">
            {categories.filter(([, s]) => s.pass_rate < 80).length} below 80%
          </p>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Per-category, weakest first                                       */}
      {/* ---------------------------------------------------------------- */}
      {/* Deliberately NOT data-print="keep": at 24 rows this section is taller
          than the space left on page 1, so forcing it whole pushes it to a
          fresh page and leaves a half-empty one behind. Individual rows are
          kept intact instead (below), which is the unit that actually matters. */}
      <section className="glass rounded-2xl p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-base font-semibold">
            Results by question type
          </h2>
          <p className="text-xs text-muted-foreground">
            <span data-print="hide">Weakest first · click to filter</span>
            <span className="hidden print:inline">Weakest first</span>
          </p>
        </div>

        <div className="mt-4 space-y-1">
          {categories.map(([name, stat]) => {
            const active = category === name;
            return (
              <button
                key={name}
                type="button"
                data-print="keep"
                onClick={() => setCategory(active ? null : name)}
                aria-pressed={active}
                className={cn(
                  "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 rounded-lg px-3 py-2.5 text-left transition-colors",
                  "sm:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_auto]",
                  "outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active ? "bg-white/10" : "hover:bg-white/[0.05]",
                )}
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">
                    {categoryLabel(name)}
                  </p>
                  <p className="truncate text-[11px] text-muted-foreground/75">
                    {CATEGORY_BLURBS[name] ?? ""}
                  </p>
                </div>
                <div className="col-span-2 sm:col-span-1">
                  <PassRateBar rate={stat.pass_rate} />
                </div>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {stat.passed}/{stat.total}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Every question                                                    */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass rounded-2xl">
        <div className="flex flex-wrap items-center gap-3 border-b border-white/[0.06] p-4 sm:p-5">
          <h2 className="text-base font-semibold">
            Every question
            <span className="ml-2 text-sm font-normal text-muted-foreground">
              {visible.length} shown
            </span>
          </h2>

          <div
            data-print="hide"
            className="ml-auto flex flex-wrap items-center gap-2"
          >
            {category ? (
              <button
                type="button"
                onClick={() => setCategory(null)}
                className="rounded-lg bg-white/10 px-2.5 py-1 text-xs text-foreground hover:bg-white/[0.14]"
              >
                {categoryLabel(category)} ✕
              </button>
            ) : null}

            <div className="flex rounded-lg bg-white/[0.06] p-0.5">
              {(["all", "failed", "passed"] as Filter[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setFilter(option)}
                  aria-pressed={filter === option}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-xs font-medium capitalize transition-colors",
                    filter === option
                      ? "bg-white/12 text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {option}
                </button>
              ))}
            </div>

            <div className="relative">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                aria-hidden
              />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search questions…"
                aria-label="Search questions"
                className="h-8 w-44 pl-8 text-xs"
              />
            </div>
          </div>
        </div>

        <ul className="divide-y divide-white/[0.06]">
          {visible.map((result) => {
            // While printing, every row is open: the scoring detail is the
            // substance of the report, not an optional drill-down.
            const open = printing || expanded === result.id;
            return (
              <li key={result.id} data-print="keep">
                <button
                  type="button"
                  onClick={() => setExpanded(expanded === result.id ? null : result.id)}
                  aria-expanded={open}
                  className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-white/[0.04] focus-visible:bg-white/[0.04] focus-visible:outline-none sm:px-5"
                >
                  <ChevronRight
                    data-print="hide"
                    className={cn(
                      "size-4 shrink-0 text-muted-foreground transition-transform",
                      open && "rotate-90",
                    )}
                    aria-hidden
                  />
                  <Verdict passed={result.passed} label="" />
                  <span className="min-w-0 flex-1 truncate text-sm">
                    {result.question || (
                      <span className="text-muted-foreground italic">
                        (empty input)
                      </span>
                    )}
                  </span>
                  <span className="hidden shrink-0 text-[11px] text-muted-foreground sm:block">
                    {categoryLabel(result.category)}
                  </span>
                  {!result.category.startsWith("input_edge_") ? (
                    <span className="w-10 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                      {result.similarity.toFixed(2)}
                    </span>
                  ) : (
                    <span className="w-10 shrink-0" />
                  )}
                </button>

                {open ? (
                  <QuestionDetail result={result} threshold={threshold} />
                ) : null}
              </li>
            );
          })}
          {visible.length === 0 ? (
            <li className="p-8 text-center text-sm text-muted-foreground">
              No questions match those filters.
            </li>
          ) : null}
        </ul>
      </section>
    </div>
  );
}
