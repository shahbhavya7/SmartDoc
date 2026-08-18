"""Evaluation: load the latest saved report, or trigger a live run.

No scoring logic here. Both paths go through the existing evaluation pipeline --
the same `eval/eval_tool` runner and the same `eval/gold_set.json` the CLI uses --
via the API. This screen only displays what that pipeline produced.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from streamlit_demo import api


def _run_label(run: dict) -> str:
    label = (run.get("label") or "").strip()
    stamp = run.get("timestamp", run.get("run_id", ""))
    passed = f"{run.get('passed')}/{run.get('question_count')}"
    rate = run.get("pass_rate", 0.0)
    name = label or stamp
    return f"{name} -- {passed} ({rate:.0f}%)"


def _poll_active_job(token: str) -> None:
    """Follow a running evaluation to completion, updating a progress bar."""
    job_id = st.session_state.get("eval_job_id")
    if not job_id:
        return

    progress = st.progress(0.0, text="Starting")
    while True:
        try:
            job = api.get_eval_job(token, job_id)
        except api.ApiError as exc:
            st.error(str(exc))
            st.session_state.pop("eval_job_id", None)
            return

        status = job.get("status")
        total = job.get("total") or 0
        done = job.get("completed") or 0
        fraction = (done / total) if total else 0.0

        if status == "done":
            progress.progress(1.0, text="Complete")
            st.session_state.pop("eval_job_id", None)
            st.session_state["eval_selected"] = job.get("run_id")
            st.success("Evaluation finished.")
            return
        if status == "error":
            progress.empty()
            st.session_state.pop("eval_job_id", None)
            st.error(job.get("error") or "The evaluation failed.")
            return

        progress.progress(
            min(fraction, 0.99),
            text=f"{job.get('phase') or status} ({done}/{total or '?'})",
        )
        time.sleep(2)


def render(token: str) -> None:
    st.subheader("Evaluation")
    st.caption(
        "Runs the existing evaluation pipeline (eval/gold_set.json) against this "
        "live backend, or loads the most recent saved report. Scoring is done by "
        "the eval pipeline, not by this app."
    )

    # -- Trigger a run ----------------------------------------------------
    # Button and caption stacked rather than sat in two columns: a button and a
    # line of caption text have different natural heights, so side by side they
    # never share a baseline and the row reads as misaligned.
    if st.button(
        "Run evaluation",
        disabled="eval_job_id" in st.session_state,
        use_container_width=False,
    ):
        try:
            job = api.start_eval_run(token, label="Streamlit demo run")
            st.session_state["eval_job_id"] = job.get("job_id")
        except api.ApiError as exc:
            st.error(str(exc))
    st.caption(
        "A full run asks every gold-set question against your own documents "
        "and takes several minutes."
    )

    if "eval_job_id" in st.session_state:
        _poll_active_job(token)

    # -- Pick a run -------------------------------------------------------
    try:
        runs = api.list_eval_runs(token, limit=25).get("runs", [])
    except api.ApiError as exc:
        st.error(str(exc))
        return

    if not runs:
        st.info(
            "No evaluation results yet. Use Run evaluation above, or run "
            "python -m eval.eval_tool.run_eval from the command line."
        )
        return

    run_ids = [r["run_id"] for r in runs]
    labels = {r["run_id"]: _run_label(r) for r in runs}
    preselected = st.session_state.get("eval_selected")
    index = run_ids.index(preselected) if preselected in run_ids else 0

    selected = st.selectbox(
        "Report",
        run_ids,
        index=index,
        format_func=lambda rid: labels.get(rid, rid),
    )

    try:
        report = api.get_eval_run(token, selected)
    except api.ApiError as exc:
        st.error(str(exc))
        return

    summary = report.get("summary", {})
    meta = report.get("meta", {})

    # -- Overall ----------------------------------------------------------
    st.divider()
    mean_similarity = summary.get("mean_similarity")
    # Equal-width columns with a gap: the four values have very different
    # widths ("109/115" vs "0.64"), and default columns size to content, which
    # left the row visibly ragged.
    columns = st.columns(4, gap="large")
    for column, (label, value) in zip(
        columns,
        (
            ("Passed", f"{summary.get('passed', 0)}/{summary.get('total', 0)}"),
            ("Pass rate", f"{summary.get('pass_rate', 0.0):.1f}%"),
            (
                "Mean similarity",
                f"{mean_similarity:.3f}" if mean_similarity is not None else "-",
            ),
            ("Pass mark", f"{meta.get('threshold', 0):.2f}"),
        ),
    ):
        column.metric(label, value)

    st.caption(
        f"Run {meta.get('timestamp', selected)} | "
        f"{meta.get('question_count', summary.get('total', 0))} questions | "
        f"embeddings {meta.get('embedding_model', 'unknown')}"
    )

    # -- Per category -----------------------------------------------------
    st.markdown("**By category**")
    by_category = report.get("by_category", {})
    if by_category:
        frame = pd.DataFrame(
            [
                {
                    "Category": name,
                    "Passed": stats.get("passed", 0),
                    "Total": stats.get("total", 0),
                    "Pass rate (%)": round(stats.get("pass_rate", 0.0), 1),
                    "Mean similarity": stats.get("mean_similarity"),
                }
                for name, stats in by_category.items()
            ]
        # Weakest first: the report's job is to point at what to fix.
        ).sort_values("Pass rate (%)")
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            # Bounded so the table does not push the failures section off the
            # page; 24 categories otherwise render as one very long block.
            height=380,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Passed": st.column_config.NumberColumn("Passed", width="small"),
                "Total": st.column_config.NumberColumn("Total", width="small"),
                "Pass rate (%)": st.column_config.ProgressColumn(
                    "Pass rate", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Mean similarity": st.column_config.NumberColumn(
                    "Mean similarity", format="%.3f", width="small"
                ),
            },
        )

    # -- Failures ---------------------------------------------------------
    results = report.get("results", [])
    failures = [r for r in results if not r.get("passed")]
    st.markdown(f"**Questions that did not pass ({len(failures)})**")
    if failures:
        frame = pd.DataFrame(
            [
                {
                    "Question": r.get("question", "")[:90],
                    "Category": r.get("category", ""),
                    "Similarity": round(r.get("similarity", 0.0), 3),
                    "Reason": r.get("fail_reason", "")[:90],
                }
                for r in failures
            ]
        )
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Similarity": st.column_config.NumberColumn(
                    "Similarity", format="%.3f", width="small"
                ),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )
    else:
        st.success("Every question passed.")
