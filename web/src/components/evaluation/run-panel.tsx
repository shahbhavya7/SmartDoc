"use client";

/**
 * Starting an evaluation: upload a test set, or run the shipped one.
 *
 * A run is expensive it asks the real system every question and pays for
 * embeddings so this panel is explicit about what is about to happen and how
 * long it takes, rather than presenting a bare button. The uploaded set is
 * validated server-side on upload, so an unusable file is rejected here with a
 * message naming the problem instead of failing minutes into a run.
 *
 * Progress is polled rather than streamed. The job outlives this component (and
 * the page): on mount the page asks whether a run is already in flight, so
 * reloading the tab mid-run reattaches instead of appearing to lose it.
 */

import { useCallback, useRef, useState } from "react";
import { FileJson, Loader2, Play, Upload, X } from "lucide-react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError, NetworkError, api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { EvalGoldSetOverview, EvalJob, EvalTestSetUpload } from "@/lib/types";

export function RunPanel({
  goldSet,
  job,
  onStarted,
}: {
  goldSet: EvalGoldSetOverview | null;
  job: EvalJob | null;
  onStarted: (job: EvalJob) => void;
}) {
  const { authorizedFetch } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);

  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [testSet, setTestSet] = useState<EvalTestSetUpload | null>(null);
  const [label, setLabel] = useState("");

  const running = job?.status === "queued" || job?.status === "running";

  const onPick = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      setUploading(true);
      try {
        const result = await authorizedFetch((token) =>
          api.uploadTestSet(token, file),
        );
        setTestSet(result);
        toast.success(`Loaded ${result.filename}`, {
          description: `${result.question_count} questions across ${result.categories.length} categories.`,
        });
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        toast.error("That test set could not be used", {
          description:
            caught instanceof ApiError || caught instanceof NetworkError
              ? caught.message
              : "Something went wrong.",
          duration: 10000,
        });
      } finally {
        setUploading(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [authorizedFetch],
  );

  const start = useCallback(async () => {
    setStarting(true);
    try {
      const started = await authorizedFetch((token) =>
        api.startEvalRun(token, {
          testSetId: testSet?.test_set_id ?? null,
          label: label.trim(),
        }),
      );
      onStarted(started);
      toast.success("Evaluation started", {
        description: testSet
          ? `Running ${testSet.question_count} uploaded questions.`
          : `Running the full ${goldSet?.total ?? ""} question gold set.`,
      });
    } catch (caught) {
      if (caught instanceof ApiError && caught.isUnauthenticated) return;
      toast.error("Could not start the evaluation", {
        description:
          caught instanceof ApiError || caught instanceof NetworkError
            ? caught.message
            : "Something went wrong.",
      });
    } finally {
      setStarting(false);
    }
  }, [authorizedFetch, goldSet, label, onStarted, testSet]);

  const questionCount = testSet?.question_count ?? goldSet?.total ?? 0;
  // ~1.5s per question against a warm API, rounded up to whole minutes.
  const estimate = Math.max(1, Math.ceil((questionCount * 1.5) / 60));

  return (
    <section className="glass rounded-2xl p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold">Run an evaluation</h2>
        <p className="text-xs text-muted-foreground">
          Questions are asked against your own documents
        </p>
      </div>

      {/* Two columns that share ROW boundaries rather than being two
          independent stacks. Each child declares its own lg: row and column, so
          the primary control in each column starts on the same line and the
          helper text under each ends on the same line. Side-by-side stacks only
          line up by accident, and here they didn't: the left column has three
          items to the right's two, so every edge across the gap was offset.

          Below lg the grid collapses to one column, where that row placement
          would interleave the two columns' children -- the name field landing
          between the gold-set card and the uploaded set it belongs with.
          `order-*` restores the reading order there and is dropped at lg. */}
      <div className="mt-4 grid grid-cols-1 gap-x-4 gap-y-2.5 lg:grid-cols-2 lg:grid-rows-[auto_auto_1fr]">
        {/* ------------------------------------------------------------ */}
        {/* Row 1: both column labels                                     */}
        {/* ------------------------------------------------------------ */}
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground order-1 lg:order-none lg:col-start-1 lg:row-start-1">
          Test set
        </p>
        <p className="hidden text-[11px] font-medium uppercase tracking-wide text-muted-foreground lg:col-start-2 lg:row-start-1 lg:block">
          Name this run (optional)
        </p>

        {/* ------------------------------------------------------------ */}
        {/* Row 2: the primary control in each column                     */}
        {/* ------------------------------------------------------------ */}
        <div className="order-2 lg:order-none lg:col-start-1 lg:row-start-2">
          <button
            type="button"
            onClick={() => setTestSet(null)}
            aria-pressed={!testSet}
            disabled={running}
            className={cn(
              "flex w-full items-start gap-3 rounded-xl p-3.5 text-left transition-colors",
              "outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60",
              !testSet ? "bg-white/10" : "glass-raised hover:bg-white/[0.08]",
            )}
          >
            <FileJson className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
            <span className="min-w-0">
              <span className="block text-sm font-medium">
                Built-in gold set
              </span>
              <span className="block text-xs text-muted-foreground">
                {goldSet?.total ?? ""} questions covering{" "}
                {goldSet?.categories.length ?? ""} question types.
              </span>
            </span>
          </button>
        </div>

        <div className="order-4 lg:order-none lg:col-start-2 lg:row-start-2">
          {/* Repeated on small screens, where the row-1 label is hidden and
              the columns stack into a single flow. */}
          <p className="mb-2.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground lg:hidden">
            Name this run (optional)
          </p>
          <Input
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="e.g. after fixing table retrieval"
            disabled={running}
            aria-label="Run label"
            // Matched to the gold-set card's height so the two columns' primary
            // controls occupy the same band instead of one floating mid-row.
            className="h-[68px]"
          />
        </div>

        {/* ------------------------------------------------------------ */}
        {/* Row 3: the secondary action in each column                    */}
        {/* ------------------------------------------------------------ */}
        <div className="order-3 lg:order-none lg:col-start-1 lg:row-start-3">
          {testSet ? (
            <div className="flex items-start gap-3 rounded-xl bg-white/10 p-3.5">
              <FileJson className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">
                  {testSet.filename}
                </span>
                <span className="block text-xs text-muted-foreground">
                  {testSet.question_count} questions ·{" "}
                  {testSet.categories.map((c) => c.name).join(", ")}
                </span>
              </span>
              <button
                type="button"
                onClick={() => setTestSet(null)}
                aria-label="Remove uploaded test set"
                className="rounded-md p-1 text-muted-foreground hover:bg-white/10 hover:text-foreground"
              >
                <X className="size-3.5" aria-hidden />
              </button>
            </div>
          ) : (
            <>
              <input
                ref={inputRef}
                type="file"
                accept=".json,.csv,application/json,text/csv"
                className="sr-only"
                onChange={(event) => onPick(event.target.files?.[0])}
              />
              <Button
                type="button"
                variant="outline"
                disabled={uploading || running}
                onClick={() => inputRef.current?.click()}
                className="w-full justify-start gap-2"
              >
                {uploading ? (
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                ) : (
                  <Upload className="size-4" aria-hidden />
                )}
                {uploading ? "Checking file…" : "Upload your own test set"}
              </Button>
            </>
          )}
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/75">
            {testSet ? (
              "Remove this to go back to the built-in gold set."
            ) : (
              <>
                A .json or .csv file where each row has a{" "}
                <code className="font-mono">question</code>, an{" "}
                <code className="font-mono">expected_answer</code>, and a{" "}
                <code className="font-mono">category</code>. It is checked before
                anything runs.
              </>
            )}
          </p>
        </div>

        <div className="order-5 lg:order-none lg:col-start-2 lg:row-start-3">
          <Button
            type="button"
            onClick={start}
            disabled={starting || running}
            className="w-full gap-2"
          >
            {starting || running ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : (
              <Play className="size-4" aria-hidden />
            )}
            {running
              ? "Evaluation in progress…"
              : `Run ${questionCount} question${questionCount === 1 ? "" : "s"}`}
          </Button>

          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/75">
            {running
              ? "You can leave this page the run continues on the server."
              : `Takes roughly ${estimate} minute${estimate === 1 ? "" : "s"}. Each question is a real request to the system.`}
          </p>
        </div>
      </div>

      {/* -------------------------------------------------------------- */}
      {/* Live progress                                                   */}
      {/* -------------------------------------------------------------- */}
      {job && job.status !== "done" ? (
        <div className="mt-5 rounded-xl bg-white/[0.05] p-4">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-medium">
              {job.status === "error" ? "Evaluation failed" : "Running…"}
            </p>
            <p className="text-xs tabular-nums text-muted-foreground">
              {job.total > 0 ? `${job.completed}/${job.total}` : null}
            </p>
          </div>
          {job.status !== "error" ? (
            <>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{
                    width: `${job.total ? (job.completed / job.total) * 100 : 5}%`,
                  }}
                />
              </div>
              <p className="mt-2 text-xs text-muted-foreground">{job.phase}</p>
            </>
          ) : (
            <p className="mt-1.5 text-xs text-rose-300">{job.error}</p>
          )}
        </div>
      ) : null}
    </section>
  );
}
