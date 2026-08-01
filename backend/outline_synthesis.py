"""Feature 3 -- outline-driven synthesis (OUTLINE_SYNTHESIS_ENABLED).

For document-wide questions only: summarise this handbook, explain the entire
process, give me an overview of the policy.

The existing outline mode retrieves broadly and diversifies across sections,
which helps, but coverage is still whatever similarity happened to surface. It
measurably leaks: a summary of the security handbook came back with no mention
of encryption at all -- context completeness 0.25 -- because the encryption
section never won a slot and nothing checked that it was missing.

This replaces "retrieve then generate" with:

    headings  -> read the document's section outline (already in metadata)
    map       -> map each heading to its indexed chunks
    ensure    -> for every MAJOR section with no retrieved representation, pull
                 its best chunk in, so no major section is silently dropped
    generate  -> answer from the completed, document-ordered context

What makes this safe
--------------------
It only ever ADDS representation for sections the retriever already indexed from
the routed document; it never invents content and never reaches outside the
documents routing selected. Everything added carries its real
``{source, page, chunk_index, section}`` metadata, so citations behave exactly
as before. Retrieval, ranking and fusion are untouched -- this consumes the
outline and the chunk store, both of which already exist.

The token budget still applies downstream, so "ensure coverage" means "give
every major section a seat at the table", not "ignore the budget". Sections
added for coverage are marked so assembly can rank them sensibly and diagnostics
can show what coverage cost.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import backend.config as config
from backend.retrieval import RetrievedUnit
from backend.routing import document_chunks, document_outline

logger = logging.getLogger("smartdoc.outline")

# A heading shorter than this is treated as structural noise (a stray title
# line, a page artifact) rather than a section worth guaranteeing.
MIN_HEADING_CHARS = 4

# Sections whose combined chunks fall below this token count are treated as
# minor: covering them crowds out substance for no gain.
MIN_SECTION_TOKENS = 60

# Coverage additions are scored at the MEDIAN of the retrieved units' scores,
# not below the weakest.
#
# Scoring them below everything was the first attempt and it silently failed:
# coverage is added before context assembly, and assembly's token budget then
# discards the lowest-scored units first -- so the sections added for coverage
# were exactly the ones dropped. The coverage report said 5/5 sections covered
# while the answer mentioned none of them. For a document-wide question breadth
# IS the goal, so a section added for coverage should compete on equal terms
# with an average retrieved passage.
COVERAGE_SCORE = 2.0


@dataclass
class CoverageReport:
    """What outline coverage did, for diagnostics and measurement."""

    applied: bool = False
    outline_sections: int = 0
    major_sections: int = 0
    covered_before: int = 0
    added_sections: list[str] = field(default_factory=list)
    still_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "outline_sections": self.outline_sections,
            "major_sections": self.major_sections,
            "covered_before": self.covered_before,
            "added_sections": self.added_sections,
            "still_missing": self.still_missing,
        }


def _normalise_heading(text: str) -> str:
    """Normalise a heading for comparison: strip numbering, case, whitespace."""
    stripped = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text or "")
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _is_major(section: str, chunks: list[dict]) -> bool:
    """True if a section is substantial enough to guarantee coverage for."""
    if len(section.strip()) < MIN_HEADING_CHARS:
        return False
    tokens = sum(int(c["metadata"].get("token_count", 0) or 0) for c in chunks)
    return tokens >= MIN_SECTION_TOKENS


def ensure_section_coverage(
    units: list[RetrievedUnit],
    sources: list[str],
    collection_name: str | None = None,
    persist_directory=None,
) -> tuple[list[RetrievedUnit], CoverageReport]:
    """Add one representative chunk for each major section not yet represented.

    Args:
        units: what retrieval returned, unmodified.
        sources: the documents routing selected. Coverage never reaches outside
            them, so this cannot reintroduce material routing excluded.

    Returns:
        ``(units_plus_coverage, report)``. The input list is not mutated.
    """
    report = CoverageReport()
    if not sources:
        return units, report

    # Which of the routed documents' chunks are already present, by id?
    #
    # Coverage is judged on the section's OPENING chunk, not on "any chunk from
    # this section". A real policy section is mostly connective boilerplate
    # around one or two load-bearing sentences, and those sit at the top of the
    # section. Treating any retrieved chunk as coverage passed a summary of the
    # security handbook that mentioned neither encryption, passwords, nor
    # backups: every section was "represented" by a paragraph of procedural
    # filler. Section-level presence is the wrong granularity for a summary.
    present_ids = {u.id for u in units}

    # Group the routed documents' indexed chunks by section. This is a metadata
    # read of documents already selected -- no search, no embedding.
    chunks_by_section: dict[tuple[str, str], list[dict]] = {}
    for chunk in document_chunks(sources, collection_name, persist_directory):
        section = chunk["metadata"].get("section", "") or ""
        if not section:
            continue
        chunks_by_section.setdefault((chunk["metadata"]["source"], section), []).append(
            chunk
        )

    outline_entries: list[dict] = []
    for source in sources:
        outline_entries.extend(
            document_outline(source, collection_name, persist_directory)
        )
    report.outline_sections = len(outline_entries)

    # Score coverage units ABOVE everything retrieved, so the budget's greedy
    # relevance fill seats them first. Median scoring was the second attempt and
    # still failed: extending the budget just let other high-scoring units take
    # the new space too, and two sections were dropped again. For a
    # document-wide summary the section leads ARE the priority -- that is what
    # "no major section silently dropped" means -- so they outrank passages that
    # merely matched a vague query well.
    scored = [u.rerank_score for u in units if u.rerank_score is not None]
    coverage_score = (max(scored) if scored else COVERAGE_SCORE) + 0.5

    additions: list[RetrievedUnit] = []
    for entry in outline_entries:
        source, section = entry["source"], entry["section"]
        section_chunks = chunks_by_section.get((source, section)) or []
        if not _is_major(section, section_chunks):
            continue
        report.major_sections += 1

        if not section_chunks:
            report.still_missing.append(f"{source}: {section}")
            continue

        # The section's opening chunk is the representative: for a summary, the
        # start of a section states what it is about, whereas a middle chunk
        # often reads as a fragment of boilerplate.
        best = min(
            section_chunks, key=lambda c: int(c["metadata"].get("chunk_index", 0) or 0)
        )

        if best["id"] in present_ids:
            report.covered_before += 1
            continue
        additions.append(
            RetrievedUnit(
                id=best["id"],
                text=best["document"],
                metadata=best["metadata"],
                rerank_score=coverage_score,
                expanded_from="outline-coverage",
                matched_text=best["document"],
            )
        )
        report.added_sections.append(f"{source}: {section}")

    report.applied = True
    if additions:
        logger.info("Outline coverage added %d section(s)", len(additions))
    return units + additions, report


def sections_missing_from_context(
    units_used: list[RetrievedUnit],
    candidates: list[RetrievedUnit],
) -> list[RetrievedUnit]:
    """Return coverage units that assembly dropped before the model saw them.

    Coverage has to be verified on the ASSEMBLED context, not on the candidate
    list: the token budget runs after coverage is added, so a section can be
    "covered" in the unit list and absent from the prompt. Checking here closes
    that gap, and the caller re-assembles with the survivors pinned.
    """
    present = {
        _normalise_heading(u.metadata.get("section", ""))
        for u in units_used
        if u.metadata.get("section")
    }
    return [
        unit
        for unit in candidates
        if unit.expanded_from == "outline-coverage"
        and _normalise_heading(unit.metadata.get("section", "")) not in present
    ]
