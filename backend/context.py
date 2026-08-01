"""Context assembly: deduplicate, merge, group, order, and budget the prompt.

Retrieval returns a relevance-ranked list. Handing that list to the model in
rank order is a mistake for three reasons:

1. **Repetition.** Chunk overlap means adjacent chunks share text verbatim. Two
   adjacent hits put the same sentences in the prompt twice, spending tokens and
   inviting the model to over-weight the duplicated claim.
2. **Incoherence.** Rank order interleaves documents, so a policy's section 3 can
   appear before its section 2 and after an unrelated manual.
3. **Position bias.** Models attend most reliably to the beginning and end of a
   long context. Placing the strongest evidence in the middle is the worst
   arrangement available.

So assembly deduplicates, optionally merges adjacent passages, groups by
document, and either interleaves by relevance (default) or preserves document
order (synthesis, procedures). Everything is capped by a token budget taken from
the query profile, and what was dropped is reported rather than hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.ingestion import count_tokens
from backend.retrieval import RetrievedUnit


@dataclass
class ContextBlock:
    """One labelled block of assembled context."""

    source: str
    doc_title: str
    section: str
    page: int
    page_end: int
    text: str
    units: list[RetrievedUnit] = field(default_factory=list)

    @property
    def page_label(self) -> str:
        """Human-readable page or page range."""
        if self.page_end and self.page_end > self.page:
            return f"pages {self.page}-{self.page_end}"
        return f"page {self.page}"


@dataclass
class AssembledContext:
    """The prompt-ready context plus what it cost and what it omitted."""

    text: str
    blocks: list[ContextBlock] = field(default_factory=list)
    tokens: int = 0
    units_used: list[RetrievedUnit] = field(default_factory=list)
    dropped_units: int = 0
    duplicates_removed: int = 0
    merges_performed: int = 0
    document_ordered: bool = False


_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Whitespace/case-normalised form used for duplicate detection."""
    return _WS_RE.sub(" ", text).strip().lower()


def _shingles(text: str, size: int = 8) -> set[str]:
    """Word shingles used to detect near-duplicate (overlapping) chunks."""
    words = _normalise(text).split()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _relevance(unit: RetrievedUnit) -> tuple[float, float]:
    """Sort key: reranker judgement first, fusion score as tie-break."""
    return (
        unit.rerank_score if unit.rerank_score is not None else 0.0,
        unit.fused_score,
    )


def deduplicate(
    units: list[RetrievedUnit], containment_threshold: float = 0.8
) -> tuple[list[RetrievedUnit], int]:
    """Drop units whose text is already substantially present.

    Near-duplicates are detected by shingle **containment** rather than
    similarity: chunk overlap produces a *subset* relationship (a short chunk
    fully inside a long one), which symmetric measures like Jaccard score as only
    moderately similar and therefore fail to catch.
    """
    kept: list[RetrievedUnit] = []
    seen_exact: set[str] = set()
    seen_shingles: list[set[str]] = []
    removed = 0

    for unit in units:
        norm = _normalise(unit.text)
        if not norm or norm in seen_exact:
            removed += 1
            continue

        shingles = _shingles(unit.text)
        duplicate = False
        for existing in seen_shingles:
            if not shingles:
                break
            if len(shingles & existing) / len(shingles) >= containment_threshold:
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue

        seen_exact.add(norm)
        seen_shingles.append(shingles)
        kept.append(unit)

    return kept, removed


def _merge_adjacent(units: list[RetrievedUnit]) -> tuple[list[RetrievedUnit], int]:
    """Merge contiguous passages from the same document into single units.

    Two chunks adjacent in the source arrive as two blocks, each with its own
    header, and the seam reads as a topic change that is not there. For an
    ordered procedure that is actively misleading: "Step 4" and "Step 5" become
    separate fragments.
    """
    by_source: dict[str, list[RetrievedUnit]] = {}
    for unit in units:
        by_source.setdefault(unit.source, []).append(unit)

    merged: list[RetrievedUnit] = []
    merges = 0

    for group in by_source.values():
        group.sort(key=lambda u: (int(u.metadata.get("chunk_index", 0) or 0), u.page))
        run: list[RetrievedUnit] = []

        def flush() -> None:
            nonlocal run, merges
            if not run:
                return
            if len(run) == 1:
                merged.append(run[0])
            else:
                merges += len(run) - 1
                best = max(run, key=_relevance)
                metadata = dict(best.metadata)
                metadata["page"] = min(u.page for u in run)
                metadata["page_end"] = max(
                    int(u.metadata.get("page_end", u.page) or u.page) for u in run
                )
                merged.append(
                    RetrievedUnit(
                        id=f"{run[0].id}+{len(run) - 1}",
                        text="\n\n".join(u.text.strip() for u in run),
                        metadata=metadata,
                        fused_score=best.fused_score,
                        rerank_score=best.rerank_score,
                        expanded_from="merged",
                        matched_text=best.matched_text or best.text,
                    )
                )
            run = []

        for unit in group:
            if not run:
                run = [unit]
                continue
            previous = run[-1]
            prev_index = int(previous.metadata.get("chunk_index", -99) or -99)
            this_index = int(unit.metadata.get("chunk_index", -99) or -99)
            contiguous = (
                0 < this_index - prev_index <= 2
                if prev_index >= 0 and this_index >= 0
                else abs(unit.page - previous.page) <= 1
            )
            if contiguous:
                run.append(unit)
            else:
                flush()
                run = [unit]
        flush()

    return merged, merges


def _group_by_document(units: list[RetrievedUnit]) -> list[ContextBlock]:
    """Group units by source, ordered by page within each group."""
    grouped: dict[str, list[RetrievedUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.source, []).append(unit)

    blocks: list[ContextBlock] = []
    for source, group in grouped.items():
        group.sort(key=lambda u: (u.page, int(u.metadata.get("chunk_index", 0) or 0)))
        first = group[0]
        sections = [
            u.metadata.get("section", "") for u in group if u.metadata.get("section")
        ]
        blocks.append(
            ContextBlock(
                source=source,
                doc_title=first.metadata.get("doc_title", ""),
                section=sections[0] if len(set(sections)) == 1 and sections else "",
                page=min(u.page for u in group),
                page_end=max(
                    int(u.metadata.get("page_end", u.page) or u.page) for u in group
                ),
                text="\n\n".join(u.text.strip() for u in group),
                units=group,
            )
        )
    return blocks


def _document_order_blocks(units: list[RetrievedUnit]) -> list[ContextBlock]:
    """Group units per document in reading order, one block per section run.

    Section titles are retained as block labels rather than stripped: for a
    document-wide answer the heading is part of the evidence, telling the model
    (and the reader checking a citation) which part of the policy a passage came
    from.
    """
    by_source: dict[str, list[RetrievedUnit]] = {}
    for unit in units:
        by_source.setdefault(unit.source, []).append(unit)

    blocks: list[ContextBlock] = []
    for source, group in by_source.items():
        group.sort(key=lambda u: (u.page, int(u.metadata.get("chunk_index", 0) or 0)))
        run: list[RetrievedUnit] = []
        current_section: str | None = None

        def flush() -> None:
            nonlocal run
            if not run:
                return
            first = run[0]
            blocks.append(
                ContextBlock(
                    source=source,
                    doc_title=first.metadata.get("doc_title", ""),
                    section=first.metadata.get("section", "") or "",
                    page=min(u.page for u in run),
                    page_end=max(
                        int(u.metadata.get("page_end", u.page) or u.page) for u in run
                    ),
                    text="\n\n".join(u.text.strip() for u in run),
                    units=list(run),
                )
            )
            run = []

        for unit in group:
            section = unit.metadata.get("section", "") or ""
            if current_section is None or section == current_section:
                run.append(unit)
                current_section = section
                continue
            flush()
            run = [unit]
            current_section = section
        flush()

    return blocks


def _interleave_for_position(blocks: list[ContextBlock]) -> list[ContextBlock]:
    """Arrange relevance-ordered blocks so the strongest sit at both ends.

    Given blocks ranked 1..n, emits 1, 3, 5, ... then ... 6, 4, 2 -- so rank 1
    opens the context and rank 2 closes it, pushing the weakest material into the
    middle where attention is least reliable.
    """
    if len(blocks) <= 2:
        return blocks
    front = blocks[0::2]
    back = blocks[1::2]
    return front + back[::-1]


def assemble(
    units: list[RetrievedUnit],
    max_tokens: int,
    group_by_document: bool = True,
    document_order: bool = False,
    merge_adjacent: bool = False,
    outline: list[dict] | None = None,
) -> AssembledContext:
    """Deduplicate, merge, group, order, and budget ``units`` into prompt context.

    Returns an ``AssembledContext`` whose ``units_used`` lists exactly the units
    that made it into the prompt -- which is what citations must be built from,
    so context and citations cannot diverge.
    """
    deduped, duplicates_removed = deduplicate(units)
    if not deduped:
        return AssembledContext(text="", duplicates_removed=duplicates_removed)

    merges = 0
    if merge_adjacent:
        deduped, merges = _merge_adjacent(deduped)

    # Apply the token budget PER UNIT, ranked by relevance across all documents,
    # before grouping. Budgeting whole document groups instead drops an entire
    # document at a time, which silently defeats exactly the queries that need
    # breadth: a multi-hop question can retrieve its bridging fact and then lose
    # it because it shared a group with a large sibling chunk.
    ranked = sorted(deduped, key=_relevance, reverse=True)
    selected: list[RetrievedUnit] = []
    used_tokens = 0
    dropped_units = 0
    for unit in ranked:
        unit_tokens = count_tokens(unit.text)
        if selected and used_tokens + unit_tokens > max_tokens:
            dropped_units += 1
            continue
        selected.append(unit)
        used_tokens += unit_tokens

    if document_order:
        # Reading order IS information for a document-wide answer, so the
        # relevance-interleaving used elsewhere is deliberately NOT applied:
        # scrambling a policy's sections to fight position bias destroys the
        # sequence the answer has to follow.
        ordered = _document_order_blocks(selected)
    else:
        blocks = (
            _group_by_document(selected)
            if group_by_document
            else [
                ContextBlock(
                    source=u.source,
                    doc_title=u.metadata.get("doc_title", ""),
                    section=u.metadata.get("section", ""),
                    page=u.page,
                    page_end=int(u.metadata.get("page_end", u.page) or u.page),
                    text=u.text,
                    units=[u],
                )
                for u in selected
            ]
        )
        blocks.sort(key=lambda b: max(_relevance(u) for u in b.units), reverse=True)
        ordered = _interleave_for_position(blocks)

    parts: list[str] = []
    if outline:
        # Giving the model the document's section outline tells it what the
        # document contains beyond the passages retrieved, so it can say what a
        # summary is missing instead of implying the excerpt is the whole thing.
        headings = "\n".join(
            f"  - {entry['section']} (page {entry['page']})" for entry in outline[:40]
        )
        titles = ", ".join(sorted({entry.get("source", "") for entry in outline}))
        parts.append(f"[Document outline: {titles}]\n{headings}")

    for index, block in enumerate(ordered, start=1):
        label = f"[{index}] {block.source}, {block.page_label}"
        if block.section:
            label += f", section: {block.section}"
        parts.append(f"{label}\n{block.text.strip()}")

    text = "\n\n---\n\n".join(parts)
    units_used = [u for block in ordered for u in block.units]

    return AssembledContext(
        text=text,
        blocks=ordered,
        tokens=count_tokens(text),
        units_used=units_used,
        dropped_units=dropped_units,
        duplicates_removed=duplicates_removed,
        merges_performed=merges,
        document_ordered=document_order,
    )
