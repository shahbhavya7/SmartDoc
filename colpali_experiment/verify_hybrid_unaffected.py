"""Isolation rule 2 confirmation: the hybrid pipeline must answer identically
to before, with the ColPali experiment installed alongside it.

Runs a handful of known-answer questions against the LIVE store (the same
one /ask uses) through the unmodified backend.rag.query path, scoped to
dev-user-0001 -- exactly as DECISIONS.md's own verification scripts do
(e.g. D8's "twenty (20) days" check). This script does not import anything
from colpali_experiment; it only proves the hybrid path was not disturbed by
this branch's additions.
"""

from __future__ import annotations

import sys

from backend.rag import query
from backend.user_scope import user_scope

CHECKS = [
    (
        "How many days of annual leave do Standard band employees get?",
        "employee_handbook.pdf",
        ["20", "twenty"],
    ),
    (
        "What is the deadline for the Anti-Bribery and Corruption training?",
        "onboarding_guide.pdf",
        ["20"],
    ),
    (
        "How often must the WidgetX unit be calibrated?",
        "widgetx_operations_manual.pdf",
        ["180"],
    ),
]


def main() -> int:
    failures = 0
    with user_scope("dev-user-0001"):
        for question, expected_source, expected_values in CHECKS:
            response = query(question)
            sources = {s.source for s in response.sources}
            answer = response.answer
            ok_source = expected_source in sources
            ok_value = any(v in answer for v in expected_values)
            status = "PASS" if (ok_source and ok_value) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"[{status}] {question}")
            print(f"    sources: {sorted(sources)}")
            print(f"    answer: {answer[:200]}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
