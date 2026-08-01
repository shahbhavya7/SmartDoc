"""Feature 2 -- recommendation planner (PLANNER_ENABLED).

For combination/workflow questions only: "design an onboarding workflow
combining the security and training requirements", "what should we do
end-to-end when a Tier 3 vendor fails its review".

The generator, handed a pile of ranked passages, tends to answer from the two or
three strongest and quietly ignore the rest. For a lookup that is correct
behaviour. For a question that asks you to BUILD something out of several
sections it is a failure mode: the workflow silently omits a stage because its
section ranked fourth.

So this inserts an explicit planning step between retrieval and generation:

    (a) enumerate  every distinct section present in the retrieved candidates
    (b) rank       those sections by the evidence already computed for them
    (c) plan       ask the model to lay out the workflow stages and assign the
                   ranked sections to stages -- structure only, no prose answer
    (d) answer     generate from the plan, with the same grounding rules

This is orchestration, not retrieval. It consumes ``RetrievedUnit`` objects
exactly as retrieval produced them and never re-queries, re-ranks, or re-embeds.
Metadata (source, page, chunk_index, section) rides through untouched, so
citations are unaffected.

Safety
------
The plan is a *scaffold*, never a source of facts. Its stage titles come from the
model, but every factual claim must still come from the retrieved passages, and
the answer prompt keeps the unchanged grounding rules -- including the exact
refusal string when the context cannot support an answer. If planning fails for
any reason the caller falls back to the normal path, so the feature can only add
structure, never remove an answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import openai

import backend.config as config
from backend.retrieval import RetrievedUnit
from backend.vectorstore import _shared_openai

logger = logging.getLogger("smartdoc.planner")

# Sections offered to the planner. Enough to cover a real workflow without
# turning the planning call into another context-window problem.
MAX_PLAN_SECTIONS = 24

# Characters of each section shown to the planner. It is choosing and ordering
# sections, not reading them for facts, so a short preview suffices.
SECTION_PREVIEW_CHARS = 220


@dataclass
class PlannedSection:
    """One retrieved section, as offered to the planner."""

    key: str
    source: str
    section: str
    page: int
    score: float
    preview: str
    units: list[RetrievedUnit] = field(default_factory=list)

    def label(self) -> str:
        heading = self.section or "(untitled section)"
        return f"{self.source} p{self.page} - {heading}"


@dataclass
class WorkflowStage:
    """One stage of the planned workflow."""

    title: str
    purpose: str = ""
    section_keys: list[str] = field(default_factory=list)


@dataclass
class WorkflowPlan:
    """The plan handed to the generator."""

    stages: list[WorkflowStage] = field(default_factory=list)
    sections: list[PlannedSection] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    ok: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stages": [
                {"title": s.title, "purpose": s.purpose, "sections": s.section_keys}
                for s in self.stages
            ],
            "sections_offered": len(self.sections),
            "unassigned": self.unassigned,
            "note": self.note,
        }


def enumerate_sections(units: list[RetrievedUnit]) -> list[PlannedSection]:
    """(a) Enumerate every distinct section present in the candidates.

    Units are grouped by ``(source, section)`` so a section split across several
    chunks is offered to the planner once, with all its units attached. Grouping
    matters: otherwise a long section occupies several planner slots and crowds
    out a short section that is a genuine workflow stage.
    """
    grouped: dict[tuple[str, str], PlannedSection] = {}
    for unit in units:
        key = (unit.source, unit.metadata.get("section", "") or "")
        existing = grouped.get(key)
        score = unit.rerank_score if unit.rerank_score is not None else unit.fused_score
        if existing is None:
            grouped[key] = PlannedSection(
                key=f"S{len(grouped) + 1}",
                source=unit.source,
                section=key[1],
                page=unit.page,
                score=score,
                preview=" ".join(unit.text.split())[:SECTION_PREVIEW_CHARS],
                units=[unit],
            )
        else:
            existing.units.append(unit)
            existing.score = max(existing.score, score)
            existing.page = min(existing.page, unit.page)
    return list(grouped.values())


def rank_sections(sections: list[PlannedSection]) -> list[PlannedSection]:
    """(b) Rank sections by the evidence retrieval already computed.

    No new scoring model: the reranker's judgement (falling back to the fusion
    score) is reused, then keys are reassigned so the planner sees S1 as the
    strongest. Introducing a second ranking signal here would mean two rankers
    that can disagree, for no measured benefit.
    """
    ordered = sorted(sections, key=lambda s: s.score, reverse=True)[:MAX_PLAN_SECTIONS]
    for index, section in enumerate(ordered, start=1):
        section.key = f"S{index}"
    return ordered


_PLANNER_PROMPT = """You plan the STRUCTURE of an answer. You do not write the \
answer and you do not state any facts.

You are given a question that asks for something to be designed, combined, or \
worked through end to end, plus a ranked list of document sections that were \
retrieved for it.

Produce an ordered workflow:
- 3 to 7 stages, in the order they should be carried out or presented.
- Each stage gets a short title and a one-line purpose.
- Assign every section that belongs to a stage by its key (S1, S2, ...). A \
section may be used by more than one stage; a section that fits nowhere may be \
left out.
- Base the stages on what the sections actually contain. Do NOT invent a stage \
for material that is not present -- a shorter, honest workflow is better than a \
complete-looking one with empty stages.

Reply with JSON only:
{"stages": [{"title": "...", "purpose": "...", "sections": ["S1", "S3"]}]}"""


def build_plan(question: str, units: list[RetrievedUnit]) -> WorkflowPlan:
    """(a)-(c): enumerate, rank, and plan. Never raises."""
    sections = rank_sections(enumerate_sections(units))
    plan = WorkflowPlan(sections=sections)

    if len(sections) < 2:
        plan.note = "Too few distinct sections to plan a workflow."
        return plan

    listing = "\n".join(
        f"{s.key}: {s.label()}\n    {s.preview}" for s in sections
    )
    try:
        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nRetrieved sections:\n{listing}",
                },
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Planner call failed; falling back to normal path: %s", exc)
        plan.note = f"Planner unavailable ({type(exc).__name__})."
        return plan

    valid_keys = {s.key for s in sections}
    stages: list[WorkflowStage] = []
    for raw in payload.get("stages") or []:
        title = str(raw.get("title", "")).strip()
        if not title:
            continue
        keys = [str(k).strip() for k in (raw.get("sections") or [])]
        stages.append(
            WorkflowStage(
                title=title,
                purpose=str(raw.get("purpose", "")).strip(),
                section_keys=[k for k in keys if k in valid_keys],
            )
        )

    if not stages:
        plan.note = "Planner returned no usable stages."
        return plan

    assigned = {k for stage in stages for k in stage.section_keys}
    plan.unassigned = [s.key for s in sections if s.key not in assigned]
    plan.stages = stages
    plan.ok = True
    return plan


def render_plan(plan: WorkflowPlan) -> str:
    """Render the plan as the scaffold shown to the generator."""
    lines: list[str] = []
    for index, stage in enumerate(plan.stages, start=1):
        refs = ", ".join(stage.section_keys) or "(no section assigned)"
        purpose = f" -- {stage.purpose}" if stage.purpose else ""
        lines.append(f"Stage {index}: {stage.title}{purpose}\n    Use: {refs}")
    if plan.unassigned:
        lines.append(
            "Sections not assigned to any stage (use only if genuinely relevant): "
            + ", ".join(plan.unassigned)
        )
    return "\n".join(lines)


def keyed_context(plan: WorkflowPlan, context_text: str) -> str:
    """Prefix the assembled context with the section key map.

    The generator needs to know which retrieved passage each ``S`` key refers
    to. The context itself is passed through unchanged -- this only prepends a
    lookup table, so the passages the model reads, and therefore the citations,
    are exactly what assembly produced.
    """
    key_map = "\n".join(f"  {s.key} = {s.label()}" for s in plan.sections)
    return f"Section keys:\n{key_map}\n\n{context_text}"


PLAN_INSTRUCTIONS = """A workflow plan has been prepared for this question from \
the retrieved sections. Follow it:

- Answer stage by stage, in the planned order, using the planned stage titles as \
headings.
- For each stage, use the sections assigned to it. State the concrete details \
those sections give -- deadlines, thresholds, approvers, values.
- If a planned stage has no supporting content in the context, say so for that \
stage rather than inventing content for it.
- The plan is a structure, not a source. Every fact must still come from the \
context passages. All grounding rules above continue to apply, including the \
exact refusal sentence when the context does not support an answer at all."""
