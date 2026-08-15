"""Tree inspection + reasoning-based (vectorless) node retrieval.

No embeddings, no vector store. The retrieval step is literally: show an LLM the
document's table of contents (node ids + titles + page ranges + summaries) and
ask it which nodes to open. For multi-hop questions it can pick several nodes
from unrelated parts of the report, which is exactly what a 220-page annual
report needs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Tree shape helpers — tolerant of the small field-name differences between
# PageIndex API versions.
# --------------------------------------------------------------------------

Node = dict[str, Any]


def normalise_tree(payload: Any) -> list[Node]:
    """Return the list of root nodes from whatever the API/cache handed us."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "tree", "structure", "nodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
    return []


def children(node: Node) -> list[Node]:
    for key in ("nodes", "children", "child_nodes"):
        value = node.get(key)
        if isinstance(value, list):
            return value
    return []


def raw_node_text(node: Node) -> str:
    """Node text exactly as PageIndex returned it, uncorrected."""
    for key in ("text", "content", "node_text", "raw_text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def node_text(node: Node) -> str:
    """Node text with verified extraction errors repaired (see corrections.py)."""
    import corrections  # local import keeps the module pair free of a cycle

    text, _ = corrections.apply(raw_node_text(node))
    return text


def node_summary(node: Node) -> str:
    for key in ("summary", "node_summary", "description"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def page_offset() -> int:
    """Pages to add so citations match the ORIGINAL published report.

    We index a trimmed PDF to fit the PageIndex Free Trial page cap, which
    shifts every page. Set PAGEINDEX_PAGE_OFFSET so answers cite the real
    report's page numbers instead of the trimmed file's.
    """
    try:
        return int(os.getenv("PAGEINDEX_PAGE_OFFSET", "0"))
    except (TypeError, ValueError):
        return 0


def page_range(node: Node) -> tuple[int | None, int | None]:
    start = node.get("start_index") or node.get("page_index") or node.get("start_page")
    end = node.get("end_index") or node.get("end_page") or start
    try:
        start = int(start) if start is not None else None
    except (TypeError, ValueError):
        start = None
    try:
        end = int(end) if end is not None else None
    except (TypeError, ValueError):
        end = None
    return start, end


def page_label(node: Node) -> str:
    """Page label in the ORIGINAL report's numbering (see page_offset)."""
    start, end = page_range(node)
    if start is None:
        return "p?"
    shift = page_offset()
    start += shift
    if end is not None:
        end += shift
    if end is None or end == start:
        return f"p{start}"
    return f"p{start}-{end}"


def walk(
    nodes: list[Node], depth: int = 0, path: tuple[str, ...] = ()
) -> Iterator[tuple[Node, int, tuple[str, ...]]]:
    """Depth-first walk yielding (node, depth, ancestor titles)."""
    for node in nodes:
        title = str(node.get("title", "")).strip()
        yield node, depth, path
        yield from walk(children(node), depth + 1, path + (title,))


def index_nodes(roots: list[Node]) -> dict[str, dict[str, Any]]:
    """node_id -> {node, depth, path} for O(1) lookup during extraction."""
    index: dict[str, dict[str, Any]] = {}
    for node, depth, path in walk(roots):
        node_id = str(node.get("node_id", "")).strip()
        if node_id:
            index[node_id] = {"node": node, "depth": depth, "path": path}
    return index


def tree_stats(roots: list[Node]) -> dict[str, Any]:
    total = 0
    max_depth = 0
    with_text = 0
    chars = 0
    max_page = 0
    for node, depth, _ in walk(roots):
        total += 1
        max_depth = max(max_depth, depth)
        text = node_text(node)
        if text:
            with_text += 1
            chars += len(text)
        _, end = page_range(node)
        if end:
            max_page = max(max_page, end)
    return {
        "nodes": total,
        "max_depth": max_depth + 1 if total else 0,
        "nodes_with_text": with_text,
        "total_text_chars": chars,
        "last_page": max_page,
    }


# --------------------------------------------------------------------------
# Outline rendering — the "prompt view" of the tree
# --------------------------------------------------------------------------


def size_label(node: Node) -> str:
    """Compact indicator of how much text a node actually holds.

    Some trees contain heading-only stub nodes whose real content sits in
    SIBLING nodes rather than children. Without this, the search model cannot
    tell 'F7 Segment information' (66 chars, empty heading) from 'Segment
    information 2025' (6.1k chars, the actual table) and picks the empty one.
    """
    own = len(node_text(node))
    subtree = own + sum(len(node_text(c)) for c, _, _ in walk(children(node)))
    if subtree < 200:
        return "EMPTY-heading only"
    if subtree < 1000:
        return f"{subtree} chars"
    return f"{subtree / 1000:.1f}k chars"


def render_outline(
    roots: list[Node],
    max_depth: int | None = None,
    include_summary: bool = True,
    summary_chars: int = 220,
    only_subtrees: set[str] | None = None,
    include_size: bool = True,
) -> str:
    """Indented text outline of the tree, one line per node.

    ``only_subtrees`` restricts output to the given node ids and everything
    beneath them — used for the drill-down pass.
    """
    lines: list[str] = []

    def emit(nodes: list[Node], depth: int, inside: bool) -> None:
        for node in nodes:
            node_id = str(node.get("node_id", "")).strip()
            here = inside or (only_subtrees is not None and node_id in only_subtrees)
            show = only_subtrees is None or here
            within_depth = max_depth is None or depth <= max_depth
            if show and within_depth:
                title = str(node.get("title", "(untitled)")).strip() or "(untitled)"
                meta = page_label(node)
                if include_size:
                    meta += f", {size_label(node)}"
                line = f"{'  ' * depth}- [{node_id}] {title} ({meta})"
                if include_summary:
                    summary = node_summary(node)
                    if summary:
                        line += f"\n{'  ' * depth}    summary: {summary[:summary_chars]}"
                lines.append(line)
            if within_depth or not show:
                emit(children(node), depth + 1, here)

    emit(roots, 0, only_subtrees is None)
    return "\n".join(lines)


def fit_outline(roots: list[Node], char_budget: int = 45000) -> tuple[str, dict[str, Any]]:
    """Render the largest outline that fits the budget.

    Tries: full tree with summaries -> full tree without summaries -> shallower
    and shallower trees. Returns (outline, info about what was dropped).
    """
    outline = render_outline(roots, include_summary=True)
    if len(outline) <= char_budget:
        return outline, {"summaries": True, "max_depth": None, "truncated": False}

    outline = render_outline(roots, include_summary=False)
    if len(outline) <= char_budget:
        return outline, {"summaries": False, "max_depth": None, "truncated": False}

    stats = tree_stats(roots)
    for depth in range(max(stats["max_depth"] - 1, 0), -1, -1):
        outline = render_outline(roots, max_depth=depth, include_summary=False)
        if len(outline) <= char_budget:
            return outline, {"summaries": False, "max_depth": depth, "truncated": True}

    return outline[:char_budget], {"summaries": False, "max_depth": 0, "truncated": True}


# --------------------------------------------------------------------------
# Reasoning-based node selection
# --------------------------------------------------------------------------

SEARCH_SYSTEM_PROMPT = """You are the retrieval step of a vectorless RAG system \
for financial documents (annual reports, 20-F/10-K style filings).

You are given the hierarchical table of contents of ONE document. Every node has \
an id, a title, a page range, a size, and sometimes a summary. You cannot see the \
node text — only the structure.

CRITICAL - read the size field. A node marked "EMPTY-heading only" contains no \
text; selecting it returns nothing to the answer model. This document's tree is \
partly flat, so a heading like "F7 Segment information" may be an empty stub \
while the real tables sit in SIBLING nodes listed just after it ("Segment \
information 2025", 6.1k chars). Always select the node that actually holds the \
text, never the empty heading above it. If you want a whole area, select each \
substantive sibling individually.

Your job: decide which nodes must be opened to answer the user's question.

Rules:
1. Many questions are MULTI-HOP: the answer is assembled from several sections \
that live far apart (e.g. a segment review + the consolidated income statement + \
a note to the accounts + the CEO letter). Return every node needed, not just the \
single best one.
2. Prefer specific leaf nodes over broad parent nodes. Pick a parent only when the \
answer is genuinely spread across all of its children.
3. In financial reports the number usually lives in the financial statements or \
the notes; the explanation usually lives in the management commentary or segment \
review. If the question asks for both a figure and a reason, select both kinds of \
node.
4. ALTERNATIVE PERFORMANCE MEASURES. A headline metric often exists in two or \
more versions that differ enormously: the IFRS line in the primary statements, \
and a non-IFRS / adjusted / underlying version that management actually steers \
on. The non-IFRS version is rarely in the primary statements - it lives in the \
SEGMENT INFORMATION note, an APM reconciliation, or the performance summary. \
Examples: revenue vs revenue excluding metals or excluding pass-through costs; \
EBIT vs adjusted EBIT vs adjusted EBITDA; net debt; ROCE; organic growth.
   So when the question asks for a headline figure - revenue, profit, margin, \
earnings, debt - you MUST select the segment information note and any \
performance-summary or reconciliation section IN ADDITION to the primary \
statement. Never return only the primary statement for such a question. Missing \
the alternative measure produces an answer that is technically true and \
materially misleading.
5. Also select sections holding comparatives, the definition of any measure \
involved, or restatements, when the question depends on them.
6. Prefer OVER-selecting to under-selecting. Extra context is cheap; a missing \
section cannot be recovered later because the answer model sees only what you \
choose. If a section plausibly contributes, include it.
7. Only return node ids that appear verbatim in the outline.

Reply with JSON only:
{"thinking": "<short reasoning about where the answer lives>",
 "nodes": [{"node_id": "...", "title": "...", "why": "<what this section contributes>", "confidence": 0.0-1.0}]}

Order `nodes` most useful first. Return between 1 and %(max_nodes)d nodes."""


def _chat_json(client: Any, model: str, system: str, user: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def select_nodes(
    query: str,
    roots: list[Node],
    openai_client: Any,
    model: str | None = None,
    max_nodes: int = 8,
    outline_char_budget: int | None = None,
) -> dict[str, Any]:
    """Reasoning-based tree search. Returns selected nodes + the model's reasoning.

    Two passes when the outline had to be truncated: pick promising branches from
    the shallow outline, then drill into just those subtrees at full depth.
    """
    model = model or os.getenv("PAGEINDEX_SEARCH_MODEL") or "gpt-4.1-mini"
    if outline_char_budget is None:
        # The full Umicore outline with summaries is ~79k chars (~20k tokens),
        # trivial for a modern context window and much better retrieval than the
        # title-only fallback. Lower it only for small-context models.
        try:
            outline_char_budget = int(os.getenv("PAGEINDEX_OUTLINE_CHAR_BUDGET", "120000"))
        except (TypeError, ValueError):
            outline_char_budget = 120000
    index = index_nodes(roots)

    outline, info = fit_outline(roots, outline_char_budget)
    system = SEARCH_SYSTEM_PROMPT % {"max_nodes": max_nodes}
    user = f"Document outline:\n{outline}\n\nQuestion: {query}"

    first = _chat_json(openai_client, model, system, user)
    passes = [{"stage": "outline", "info": info, "response": first}]
    selected = first.get("nodes") or []

    if info.get("truncated") and selected:
        branch_ids = {str(n.get("node_id")) for n in selected if n.get("node_id")}
        detail = render_outline(roots, include_summary=True, only_subtrees=branch_ids)
        if detail.strip():
            detail = detail[:outline_char_budget]
            drill_user = (
                f"Relevant branches of the document, now shown in full detail:\n{detail}"
                f"\n\nQuestion: {query}\n\n"
                "Refine the selection to the most specific nodes that answer the question."
            )
            second = _chat_json(openai_client, model, system, drill_user)
            passes.append({"stage": "drill_down", "response": second})
            if second.get("nodes"):
                selected = second["nodes"]

    # Keep only ids that really exist, de-duplicate, preserve model ordering.
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for item in selected:
        node_id = str(item.get("node_id", "")).strip()
        if not node_id or node_id in seen:
            continue
        entry = index.get(node_id)
        if entry is None:
            dropped.append(node_id)
            continue
        seen.add(node_id)
        node = entry["node"]
        resolved.append(
            {
                "node_id": node_id,
                "title": str(node.get("title", "")).strip() or item.get("title", ""),
                "why": item.get("why", ""),
                "confidence": item.get("confidence"),
                "pages": page_label(node),
                "path": entry["path"],
                "depth": entry["depth"],
                "node": node,
            }
        )
        if len(resolved) >= max_nodes:
            break

    return {
        "query": query,
        "model": model,
        "thinking": passes[-1]["response"].get("thinking", ""),
        "nodes": resolved,
        "hallucinated_ids": dropped,
        "outline_info": info,
        "passes": passes,
    }


# --------------------------------------------------------------------------
# Extraction — turn the selected node ids into text for the generator
# --------------------------------------------------------------------------


def collect_text(node: Node, include_children: bool = True, max_chars: int = 20000) -> str:
    """Text of a node; for parent nodes, concatenate the subtree."""
    parts: list[str] = []
    total = 0

    def add(current: Node, depth: int) -> None:
        nonlocal total
        if total >= max_chars:
            return
        text = node_text(current)
        if text:
            if depth > 0:
                title = str(current.get("title", "")).strip()
                if title:
                    parts.append(f"\n### {title} ({page_label(current)})")
                    total += len(title) + 12
            chunk = text[: max_chars - total]
            parts.append(chunk)
            total += len(chunk)
        if include_children:
            for child in children(current):
                add(child, depth + 1)

    add(node, 0)
    return "\n".join(parts).strip()


def build_context(
    selected: list[dict[str, Any]],
    char_budget: int = 90000,
) -> tuple[str, list[dict[str, Any]]]:
    """Assemble the retrieved sections into one prompt context.

    Budget is split evenly across the selected nodes so one huge section cannot
    starve the others (important for multi-hop answers).
    """
    if not selected:
        return "", []

    per_node = max(char_budget // len(selected), 2000)
    blocks: list[str] = []
    used: list[dict[str, Any]] = []

    for item in selected:
        node = item["node"]
        text = collect_text(node, include_children=True, max_chars=per_node)
        if not text:
            continue
        breadcrumb = " > ".join([p for p in item.get("path", ()) if p] + [item["title"]])
        blocks.append(
            f"[{item['node_id']}] {breadcrumb} ({item['pages']})\n"
            f"{'-' * 70}\n{text}"
        )
        used.append(
            {
                "node_id": item["node_id"],
                "title": item["title"],
                "pages": item["pages"],
                "breadcrumb": breadcrumb,
                "chars": len(text),
                "why": item.get("why", ""),
                "confidence": item.get("confidence"),
                "text": text,
            }
        )

    return "\n\n".join(blocks), used
