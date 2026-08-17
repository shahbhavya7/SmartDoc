"use client";

/**
 * The evaluation page.
 *
 * Two tabs: the results of a run, and how the scoring works. They are separate
 * because they answer different questions ("how is the system doing" vs "why
 * should I believe that number"), and interleaving them would bury both.
 *
 * The page opens on the most recent run. Everything shown comes from the saved
 * run file — nothing is recomputed here, so what the page displays and what the
 * CLI printed for the same run are the same numbers by construction.
 *
 * A run in flight is polled, and the poll is re-attached on mount rather than
 * only being started by the button, so reloading mid-run picks the job back up
 * instead of appearing to lose it.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, FlaskConical, History, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";
import { MethodPanel } from "@/components/evaluation/method-panel";
import { RunPanel } from "@/components/evaluation/run-panel";
import { RunResults } from "@/components/evaluation/run-results";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, NetworkError, api } from "@/lib/api";
import { formatRelativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  EvalCalibration,
  EvalGoldSetOverview,
  EvalJob,
  EvalMethod,
  EvalRun,
  EvalRunSummary,
} from "@/lib/types";

/** "20260817T094624Z" -> a Date. */
function parseRunTimestamp(stamp: string): Date | null {
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(stamp);
  if (!match) return null;
  const [, y, mo, d, h, mi, s] = match;
  return new Date(
    Date.UTC(+y, +mo - 1, +d, +h, +mi, +s),
  );
}

function runLabel(run: EvalRunSummary): string {
  if (run.label) return run.label;
  const date = parseRunTimestamp(run.timestamp);
  return date ? formatRelativeTime(date.toISOString()) : run.run_id;
}

type Tab = "results" | "method";

export function Evaluation() {
  const { authorizedFetch } = useAuth();

  const [tab, setTab] = useState<Tab>("results");
  const [runs, setRuns] = useState<EvalRunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [run, setRun] = useState<EvalRun | null>(null);
  const [method, setMethod] = useState<EvalMethod | null>(null);
  const [calibration, setCalibration] = useState<EvalCalibration | null>(null);
  const [goldSet, setGoldSet] = useState<EvalGoldSetOverview | null>(null);
  const [job, setJob] = useState<EvalJob | null>(null);

  const [loading, setLoading] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [printing, setPrinting] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Produce the PDF via the browser's own print-to-PDF.
   *
   * `printing` expands all 115 questions, and `window.print()` is blocking, so
   * it must not be called until React has actually committed that render --
   * otherwise the dialog captures the collapsed DOM and the PDF is a list of
   * one-line rows. Two rAFs put the call after paint of the state change
   * (one schedules before the commit's paint, the second lands after it).
   */
  const downloadPdf = useCallback(() => {
    setPrinting(true);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.print();
        // Restoring immediately is safe: print() does not return until the
        // dialog is dismissed, so the expanded DOM has already been captured.
        setPrinting(false);
      });
    });
  }, []);

  const refreshRuns = useCallback(async () => {
    const { runs: rows } = await authorizedFetch((token) => api.evalRuns(token, 30));
    setRuns(rows);
    return rows;
  }, [authorizedFetch]);

  /* ---------------------------------------------------------------- */
  /* Initial load                                                      */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // The method text and gold-set overview are cheap and always wanted.
        // Calibration is optional: a project that has never calibrated should
        // still get a working page, just without that section.
        const [rows, methodText, gold] = await Promise.all([
          authorizedFetch((token) => api.evalRuns(token, 30)),
          authorizedFetch((token) => api.evalMethod(token)),
          authorizedFetch((token) => api.evalGoldSet(token)),
        ]);
        if (cancelled) return;
        setRuns(rows.runs);
        setMethod(methodText);
        setGoldSet(gold);
        if (rows.runs.length > 0) setSelected(rows.runs[0].run_id);

        try {
          const calib = await authorizedFetch((token) => api.evalCalibration(token));
          if (!cancelled) setCalibration(calib);
        } catch {
          /* Never calibrated: the panel is simply omitted. */
        }

        // Re-attach to a run that is already going.
        try {
          const { job: active } = await authorizedFetch((token) =>
            api.evalActiveJob(token),
          );
          if (!cancelled && active) setJob(active);
        } catch {
          /* Not fatal. */
        }
      } catch (caught) {
        if (cancelled) return;
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        setError(
          caught instanceof ApiError || caught instanceof NetworkError
            ? caught.message
            : "Could not load evaluation results.",
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authorizedFetch]);

  /* ---------------------------------------------------------------- */
  /* Load whichever run is selected                                    */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;

    (async () => {
      // Set inside the async body rather than in the effect body: a synchronous
      // setState during an effect triggers a cascading render, and the flag is
      // only meaningful once the request is actually in flight anyway.
      setLoadingRun(true);
      try {
        const payload = await authorizedFetch((token) =>
          api.evalRun(token, selected),
        );
        if (!cancelled) setRun(payload);
      } catch (caught) {
        if (cancelled) return;
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        setError(
          caught instanceof ApiError || caught instanceof NetworkError
            ? caught.message
            : "Could not load that run.",
        );
      } finally {
        if (!cancelled) setLoadingRun(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authorizedFetch, selected]);

  /* ---------------------------------------------------------------- */
  /* Poll an in-flight run                                             */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    const active = job?.status === "queued" || job?.status === "running";
    if (!active || !job) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const next = await authorizedFetch((token) => api.evalJob(token, job.job_id));
        setJob(next);

        if (next.status === "done" && next.run_id) {
          toast.success("Evaluation finished", {
            description: "Showing the new results.",
          });
          await refreshRuns();
          // Select the run this job produced, so the page lands on it rather
          // than on whatever was previously open.
          setSelected(next.run_id);
        } else if (next.status === "error") {
          toast.error("Evaluation failed", {
            description: next.error || "The server did not say why.",
            duration: 10000,
          });
        }
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        /* A dropped poll is not fatal; the next tick retries. */
      }
    }, 2500);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [authorizedFetch, job, refreshRuns]);

  /* ---------------------------------------------------------------- */

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-9 w-64 bg-white/10" />
        <Skeleton className="h-40 w-full rounded-2xl bg-white/10" />
        <Skeleton className="h-64 w-full rounded-2xl bg-white/10" />
      </div>
    );
  }

  return (
    <div className="space-y-7">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <section className="animate-rise space-y-1.5">
        <p className="text-sm text-muted-foreground">Quality</p>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Evaluation
        </h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Every question below was asked to the real system and its answer
          compared against a known-correct one. Nothing here is estimated —
          <span data-print="hide"> open any question to see exactly how it was
          judged.</span>
          <span className="hidden print:inline">
            {" "}
            each question below shows exactly how it was judged.
          </span>
        </p>
        {/* Print-only provenance line. On screen this context comes from the
            nav bar and the run picker, both of which are hidden on paper, so
            without it the PDF would not say which run it documents. */}
        {run ? (
          <p className="hidden text-xs text-muted-foreground print:block">
            {run.meta.label ? `${run.meta.label} · ` : ""}
            Run {run.meta.timestamp} · {run.meta.question_count} questions ·
            pass mark {run.meta.threshold.toFixed(2)} ·{" "}
            {run.meta.embedding_model}
          </p>
        ) : null}
      </section>

      {error ? (
        <div className="glass flex items-start gap-3 rounded-2xl p-5">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-300" aria-hidden />
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      {/* Run controls                                                      */}
      {/* ---------------------------------------------------------------- */}
      <div data-print="hide">
        <RunPanel goldSet={goldSet} job={job} onStarted={setJob} />
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Tabs                                                              */}
      {/* ---------------------------------------------------------------- */}
      <div data-print="hide" className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-xl bg-white/[0.06] p-0.5">
          {(
            [
              ["results", "Results", FlaskConical],
              ["method", "How this is measured", History],
            ] as const
          ).map(([value, text, Icon]) => (
            <button
              key={value}
              type="button"
              onClick={() => setTab(value)}
              aria-pressed={tab === value}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors",
                "outline-none focus-visible:ring-2 focus-visible:ring-ring",
                tab === value
                  ? "bg-white/12 text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="size-4" aria-hidden />
              {text}
            </button>
          ))}
        </div>

        {/* Run picker + PDF */}
        {tab === "results" && runs.length > 0 ? (
          <div className="ml-auto flex items-center gap-2">
            <label
              htmlFor="eval-run-picker"
              className="text-xs text-muted-foreground"
            >
              Showing
            </label>
            <select
              id="eval-run-picker"
              value={selected ?? ""}
              onChange={(event) => setSelected(event.target.value)}
              className="h-9 rounded-lg border border-white/10 bg-white/[0.06] px-2.5 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {runs.map((row) => (
                <option key={row.run_id} value={row.run_id} className="bg-neutral-900">
                  {runLabel(row)} — {row.passed}/{row.question_count} (
                  {row.pass_rate.toFixed(0)}%)
                  {row.shared ? " · baseline" : ""}
                </option>
              ))}
            </select>

            <Button
              type="button"
              variant="outline"
              onClick={downloadPdf}
              disabled={!run || loadingRun}
              className="h-9 gap-2"
              title="Opens your browser's print dialog — choose 'Save as PDF'"
            >
              <Download className="size-4" aria-hidden />
              <span className="hidden sm:inline">Download PDF</span>
            </Button>
          </div>
        ) : null}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Body                                                              */}
      {/* ---------------------------------------------------------------- */}
      {tab === "method" ? (
        <MethodPanel method={method} calibration={calibration} loading={false} />
      ) : loadingRun ? (
        <div className="space-y-4">
          <Skeleton className="h-28 w-full rounded-2xl bg-white/10" />
          <Skeleton className="h-64 w-full rounded-2xl bg-white/10" />
        </div>
      ) : run ? (
        <>
          <RunMeta run={run} />
          <RunResults run={run} printing={printing} />
        </>
      ) : (
        <div className="glass rounded-2xl p-10 text-center">
          <FlaskConical
            className="mx-auto size-8 text-muted-foreground/50"
            aria-hidden
          />
          <p className="mt-3 text-sm font-medium">No evaluation has run yet</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
            Start one above to see how the system scores against the known-answer
            test set.
          </p>
        </div>
      )}
    </div>
  );
}

/** The provenance strip: what produced these numbers. */
function RunMeta({ run }: { run: EvalRun }) {
  const when = parseRunTimestamp(run.meta.timestamp);
  const items: [string, string][] = [
    ["Run", run.meta.label || run.meta.timestamp],
    ["When", when ? formatRelativeTime(when.toISOString()) : "—"],
    ["Questions", String(run.meta.question_count)],
    ["Pass mark", run.meta.threshold.toFixed(2)],
    ["Embedding model", run.meta.embedding_model],
  ];

  return (
    <section data-print="keep" className="glass flex flex-wrap gap-x-8 gap-y-3 rounded-2xl px-6 py-4">
      {items.map(([label, value]) => (
        <div key={label}>
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground/70">
            {label}
          </p>
          <p className="mt-0.5 truncate text-sm font-medium">{value}</p>
        </div>
      ))}
    </section>
  );
}
