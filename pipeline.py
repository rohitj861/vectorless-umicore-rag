"""End-to-end vectorless RAG pipeline: tree -> node selection -> text -> answer."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

import generation
import retrieval
from pageindex_client import PageIndexClient

load_dotenv()


def load_roots(doc_id: str, client: PageIndexClient | None = None) -> list[retrieval.Node]:
    """Tree root nodes for a doc_id, from local cache when available."""
    client = client or PageIndexClient()
    payload = client.get_or_fetch_tree(doc_id)
    if payload.get("status") not in (None, "completed"):
        raise RuntimeError(
            f"Document {doc_id} is not ready yet (status={payload.get('status')})."
        )
    roots = retrieval.normalise_tree(payload)
    if not roots:
        raise RuntimeError(f"Empty tree for {doc_id}: {str(payload)[:300]}")
    return roots


def retrieve(
    query: str,
    roots: list[retrieval.Node],
    openai_client: Any | None = None,
    search_model: str | None = None,
    max_nodes: int = 8,
    char_budget: int | None = None,
) -> dict[str, Any]:
    """Steps 3+4: pick nodes by reasoning, then extract their text."""
    openai_client = openai_client or generation.get_openai_client()
    selection = retrieval.select_nodes(
        query,
        roots,
        openai_client,
        model=search_model or os.getenv("PAGEINDEX_SEARCH_MODEL"),
        max_nodes=max_nodes,
    )
    context, used = retrieval.build_context(
        selection["nodes"],
        char_budget=char_budget or generation.context_budget(),
    )
    selection["context"] = context
    selection["sources"] = used
    return selection


def ask(
    query: str,
    doc_id: str,
    doc_label: str = "annual report",
    max_nodes: int = 8,
    search_model: str | None = None,
    answer_model: str | None = None,
) -> dict[str, Any]:
    """Full pipeline for CLI use."""
    roots = load_roots(doc_id)
    openai_client = generation.get_openai_client()
    result = retrieve(
        query, roots, openai_client, search_model=search_model, max_nodes=max_nodes
    )
    result["answer"] = generation.answer(
        query,
        result["context"],
        client=openai_client,
        model=answer_model,
        doc_label=doc_label,
    )
    return result


# A few questions that deliberately span sections of an annual report.
SAMPLE_MULTIHOP_QUERIES = [
    "How did the Battery Materials segment perform in 2025, how did that flow "
    "through to group revenue and adjusted EBITDA, and what does management say "
    "about the outlook for it?",
    "Reconcile the group's reported net profit to adjusted EBITDA, and explain "
    "which one-off items or impairments drove the gap.",
    "What were the main capital expenditure commitments in 2025, how were they "
    "financed, and what did that do to net debt and the leverage ratio?",
    "Compare the CEO's stated strategic priorities with the risk factors "
    "disclosed, and identify which priorities carry the most disclosed risk.",
    "How does executive remuneration link to financial performance, and did the "
    "2025 results trigger the short-term incentive targets?",
]
