"""Verified repairs to PageIndex's extracted text.

PageIndex's parse of a PDF is very good but not perfect, and in a financial
document a single flipped word can invert the meaning of a headline metric. This
module applies a small, explicit, auditable set of corrections to node text
before it reaches the answer model.

Rules for adding an entry here:
  1. The error must be VERIFIED against the source PDF - quote what the original
     actually says in `verified`. Never "fix" something you have not read.
  2. The pattern must be narrow enough that it cannot fire on correct text.
  3. Corrections are logged, never silent. `apply()` returns what it changed.

This is a data-integrity layer, not a place to reword the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    pattern: str          # regex matched against node text
    replacement: str      # replacement (may use backreferences)
    reason: str           # why this is wrong
    verified: str         # what the ORIGINAL pdf says, and where


CORRECTIONS: list[Correction] = [
    Correction(
        pattern=r"(?i)\b(Revenues?)\s*\(including metals?\)",
        replacement=r"\1 (excluding metal)",
        reason=(
            "Extraction inverted the meaning of Umicore's headline KPI. "
            "'Revenues (excluding metal)' is turnover less the pass-through value "
            "of purchased metals; rendering it as '(including metal)' makes the "
            "EUR 3.56bn figure look like it should reconcile to the EUR 19.37bn "
            "turnover line, which is backwards."
        ),
        verified=(
            "Original PDF p18 reads 'Revenues (excluding metal) 1,657 1,772 "
            "3,461 3,562'. All 11 occurrences of this label in the source PDF "
            "say 'excluding'; none say 'including'."
        ),
    ),
]


def apply(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (corrected_text, [(matched_text, reason), ...])."""
    if not text:
        return text, []
    applied: list[tuple[str, str]] = []
    for correction in CORRECTIONS:
        matches = re.findall(correction.pattern, text)
        if not matches:
            continue
        found = re.search(correction.pattern, text)
        if found:
            applied.append((found.group(0), correction.reason))
        text = re.sub(correction.pattern, correction.replacement, text)
    return text, applied


def audit(roots: list[dict]) -> list[dict]:
    """Report every node a correction would fire on. Used by --audit tooling."""
    import retrieval  # local import to avoid a circular import at module load

    report: list[dict] = []
    for node, _, _ in retrieval.walk(roots):
        raw = retrieval.raw_node_text(node)
        _, applied = apply(raw)
        for matched, reason in applied:
            report.append(
                {
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                    "pages": retrieval.page_label(node),
                    "matched": matched,
                    "reason": reason,
                }
            )
    return report
