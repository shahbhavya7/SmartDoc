"use client";

/**
 * ColPali vs Hybrid RAG comparison (colpali branch experiment).
 *
 * Everything shown here comes straight from the saved comparison JSON
 * (GET /eval/comparison/latest, itself built by
 * eval.eval_tool.run_comparison from two saved eval runs) -- nothing is
 * recomputed client-side. The two hypothesis conclusions are rendered FIRST,
 * above the summary numbers and the per-category table, because they are the
 * headline finding of the whole exercise, not a footnote to a data table.
 *
 * Deliberately reuses the existing evaluation page's presentational pieces
 * (categoryLabel, PassRateBar, Verdict, the "glass" card classes) rather than
 * inventing a second visual language for what is, at heart, the same kind of
 * report the /evaluation page already renders.
 */

import { useEffect, useState } from "react";
import { Eye, FileText, TriangleAlert } from "lucide-react";

import { useAuth } from "@/components/auth-provider";
import {
  categoryLabel,
  PassRateBar,
  Verdict,
} from "@/components/evaluation/scoring-primitives";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, NetworkError, api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { ComparisonHypothesis, ComparisonReport } from "@/lib/types";

function describe(error: unknown, fallback: string): string {
  if (error instanceof ApiError || error instanceof NetworkError) return error.message;
  return fallback;
}

function HypothesisCard({
  title,
  hypothesis,
}: {
  title: string;
  hypothesis: ComparisonHypothesis;
}) {
  const verdict =
    hypothesis.held === null ? "Not measurable" : hypothesis.held ? "Held" : "Did not hold";
  const tone =
    hypothesis.held === null
      ? "bg-white/[0.06] text-muted-foreground"
      : hypothesis.held
        ? "bg-emerald-400/12 text-emerald-300"
        : "bg-rose-400/12 text-rose-300";

  return (
    <div className="glass-accent rounded-2xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className={cn("rounded-md px-2 py-0.5 text-[11px] font-medium", tone)}>
          {verdict}
        </span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
        {hypothesis.detail}
      </p>
      {hypothesis.comparison_formatting_note ? (
        <p className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/[0.06] p-2.5 text-[11px] leading-relaxed text-amber-200/90">
          {hypothesis.comparison_formatting_note}
        </p>
      ) : null}
      <p className="mt-2 text-[11px] text-muted-foreground/70">
        Categories: {hypothesis.categories.map(categoryLabel).join(", ")}
      </p>
    </div>
  );
}

function StatPair({
  label,
  hybrid,
  colpali,
  note,
}: {
  label: string;
  hybrid: string;
  colpali: string;
  note?: string;
}) {
  return (
    <div className="glass rounded-2xl p-5">
      <p className="text-[13px] font-medium text-muted-foreground">{label}</p>
      <div className="mt-2.5 flex items-baseline gap-4">
        <span className="flex items-center gap-1.5 text-lg font-semibold tabular-nums">
          <FileText className="size-3.5 text-muted-foreground" aria-hidden />
          {hybrid}
        </span>
        <span className="flex items-center gap-1.5 text-lg font-semibold tabular-nums text-accent-gradient">
          <Eye className="size-3.5" aria-hidden />
          {colpali}
        </span>
      </div>
      {note ? <p className="mt-1.5 text-[11px] text-muted-foreground/70">{note}</p> : null}
    </div>
  );
}

export function Comparison() {
  const { authorizedFetch } = useAuth();
  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    authorizedFetch((token) => api.evalComparisonLatest(token))
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((caught) => {
        if (cancelled) return;
        if (caught instanceof ApiError && caught.status === 404) {
          setNotFound(true);
          return;
        }
        setError(describe(caught, "Could not load the comparison report."));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [authorizedFetch]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-2xl" />
        <Skeleton className="h-24 rounded-2xl" />
        <Skeleton className="h-64 rounded-2xl" />
      </div>
    );
  }

  if (notFound) {
    return (
      <div className="glass rounded-2xl p-8 text-center">
        <TriangleAlert className="mx-auto size-6 text-amber-300" aria-hidden />
        <p className="mt-3 text-sm text-muted-foreground">
          No ColPali-vs-Hybrid comparison has been generated yet. Run{" "}
          <code className="rounded bg-white/[0.06] px-1.5 py-0.5 text-xs">
            python -m eval.eval_tool.run_comparison --latest
          </code>{" "}
          after producing a hybrid and a colpali eval run.
        </p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="glass rounded-2xl p-8 text-center">
        <TriangleAlert className="mx-auto size-6 text-rose-300" aria-hidden />
        <p className="mt-3 text-sm text-muted-foreground">{error}</p>
      </div>
    );
  }

  const { hybrid, colpali } = report;
  const costNote =
    "colpali uses vision-input pricing (page images), not normalized against hybrid's text-token cost";

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- */}
      {/* Hypothesis conclusions -- FIRST, above everything else            */}
      {/* ---------------------------------------------------------------- */}
      <section className="grid gap-4 md:grid-cols-2">
        <HypothesisCard
          title="Hybrid's SQL-backed exact lookup outperforms ColPali on numeric/table-cell questions"
          hypothesis={report.hypotheses.sql_exact_match_outperforms}
        />
        <HypothesisCard
          title="ColPali outperforms hybrid on layout/table/comparison-heavy questions"
          hypothesis={report.hypotheses.colpali_layout_outperforms}
        />
      </section>

      {report.surprising_categories.length > 0 ? (
        <section className="glass rounded-2xl p-5">
          <h3 className="text-sm font-semibold">
            Surprising categories (large gap, not one of the two hypotheses above)
          </h3>
          <ul className="mt-3 space-y-1.5">
            {report.surprising_categories.map((row) => (
              <li key={row.category} className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{categoryLabel(row.category)}</span>
                <span className="tabular-nums">
                  hybrid {row.hybrid_pass_rate}% vs colpali {row.colpali_pass_rate}%{" "}
                  <span
                    className={cn(
                      "ml-1 font-medium",
                      (row.delta ?? 0) > 0 ? "text-accent-gradient" : "text-rose-300",
                    )}
                  >
                    ({row.delta! > 0 ? "+" : ""}
                    {row.delta}pp)
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      {/* Overall summary                                                   */}
      {/* ---------------------------------------------------------------- */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatPair
          label="Overall pass rate"
          hybrid={`${hybrid.pass_rate.toFixed(1)}%`}
          colpali={`${colpali.pass_rate.toFixed(1)}%`}
        />
        <StatPair
          label="Mean similarity"
          hybrid={hybrid.mean_similarity?.toFixed(4) ?? "n/a"}
          colpali={colpali.mean_similarity?.toFixed(4) ?? "n/a"}
        />
        <StatPair
          label="Mean latency / query"
          hybrid={hybrid.mean_latency_ms != null ? `${hybrid.mean_latency_ms.toFixed(0)}ms` : "n/a"}
          colpali={
            colpali.mean_latency_ms != null ? `${colpali.mean_latency_ms.toFixed(0)}ms` : "n/a"
          }
        />
        <StatPair
          label="Mean cost / query"
          hybrid={hybrid.mean_cost_usd != null ? `$${hybrid.mean_cost_usd.toFixed(6)}` : "n/a"}
          colpali={colpali.mean_cost_usd != null ? `$${colpali.mean_cost_usd.toFixed(6)}` : "n/a"}
          note={costNote}
        />
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Per-category, sorted by gap size                                  */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass rounded-2xl p-6">
        <h2 className="text-base font-semibold">By category, sorted by gap size</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                <th className="pb-2 font-medium">Category</th>
                <th className="pb-2 font-medium">Hybrid</th>
                <th className="pb-2 font-medium">ColPali</th>
                <th className="pb-2 text-right font-medium">Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.06]">
              {report.by_category.map((row) => (
                <tr key={row.category}>
                  <td className="py-2 pr-3">
                    <span className="text-sm">{categoryLabel(row.category)}</span>
                    {row.watch ? (
                      <span className="ml-2 rounded-md bg-amber-400/12 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                        WATCH
                      </span>
                    ) : null}
                  </td>
                  <td className="py-2 pr-3">
                    {row.hybrid_pass_rate != null ? (
                      <PassRateBar rate={row.hybrid_pass_rate} />
                    ) : (
                      <span className="text-xs text-muted-foreground">n/a</span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    {row.colpali_pass_rate != null ? (
                      <PassRateBar rate={row.colpali_pass_rate} />
                    ) : (
                      <span className="text-xs text-muted-foreground">n/a</span>
                    )}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {row.delta != null ? (
                      <span
                        className={cn(
                          "font-medium",
                          row.delta > 0
                            ? "text-accent-gradient"
                            : row.delta < 0
                              ? "text-rose-300"
                              : "text-muted-foreground",
                        )}
                      >
                        {row.delta > 0 ? "+" : ""}
                        {row.delta}pp
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">n/a</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Intent-classifier asymmetry                                       */}
      {/* ---------------------------------------------------------------- */}
      <section className="glass rounded-2xl p-6">
        <h2 className="text-base font-semibold">
          Intent-classifier asymmetry (aggregation / enumeration questions)
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Each backend detects "this needs exhaustive coverage" with a different
          mechanism -- logged per question so a category gap can be attributed to
          retrieval architecture, not classifier tuning.
        </p>

        {report.intent_classifier_asymmetry.per_question.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">
            No shared aggregation/enumeration questions between the two runs.
          </p>
        ) : (
          <ul className="mt-4 space-y-3">
            {report.intent_classifier_asymmetry.per_question.map((row) => (
              <li
                key={row.id}
                className="rounded-lg border border-white/[0.08] bg-black/25 p-3"
              >
                <p className="text-sm">{row.question}</p>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  <div className="flex items-center gap-2 text-xs">
                    <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                    <span className="text-muted-foreground">
                      fired={String(row.hybrid_fired)}
                    </span>
                    <Verdict passed={row.hybrid_passed} />
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <Eye className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                    <span className="text-muted-foreground">
                      fired={String(row.colpali_fired)}
                    </span>
                    <Verdict passed={row.colpali_passed} />
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}

        {report.intent_classifier_asymmetry.hybrid_blind_spot_questions.length > 0 ? (
          <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-400/[0.06] p-3">
            <p className="flex items-center gap-1.5 text-xs font-medium text-amber-300">
              <TriangleAlert className="size-3.5" aria-hidden />
              Production-relevant finding (not fixed on this branch)
            </p>
            <p className="mt-1.5 text-xs text-muted-foreground">
              Hybrid&apos;s own exhaustive-intent classifier did not fire expansion
              for these &quot;how many X&quot;-style questions -- the same blind
              spot found in ColPali&apos;s regex also affects the classifier
              hybrid actually uses in production.
            </p>
            <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
              {report.intent_classifier_asymmetry.hybrid_blind_spot_questions.map((q) => (
                <li key={q.id}>&bull; {q.question}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <p className="text-center text-[11px] text-muted-foreground/60">
        hybrid run: {hybrid.run_path.split("/").pop()} &middot; colpali run:{" "}
        {colpali.run_path.split("/").pop()}
        {report.generated_at ? ` · generated ${report.generated_at}` : ""}
      </p>
    </div>
  );
}
