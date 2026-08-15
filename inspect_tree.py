"""Step 3 — inspect the tree PageIndex built.

    python inspect_tree.py                       # newest cached doc
    python inspect_tree.py --depth 2
    python inspect_tree.py --node 0006 --text    # dump one section's text
    python inspect_tree.py --grep "battery"
"""

from __future__ import annotations

import argparse
import sys

import retrieval
from pageindex_client import PageIndexClient, PageIndexError


def newest_cached_doc_id(client: PageIndexClient) -> str | None:
    trees = sorted(
        client.cache_dir.glob("*.tree.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if trees:
        return trees[0].name.removesuffix(".tree.json")
    registry = client.registry()
    return next(iter(registry.values()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a PageIndex document tree.")
    parser.add_argument("--doc-id")
    parser.add_argument("--depth", type=int, default=1, help="Max depth to print (0 = roots).")
    parser.add_argument("--summaries", action="store_true", help="Include node summaries.")
    parser.add_argument("--node", help="Print details for one node id.")
    parser.add_argument("--text", action="store_true", help="With --node, dump full text.")
    parser.add_argument("--grep", help="List nodes whose title matches (case-insensitive).")
    args = parser.parse_args()

    try:
        client = PageIndexClient()
    except PageIndexError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    doc_id = args.doc_id or newest_cached_doc_id(client)
    if not doc_id:
        print("! No doc_id given and nothing cached. Run ingest.py first.", file=sys.stderr)
        return 1

    payload = client.get_or_fetch_tree(doc_id)
    roots = retrieval.normalise_tree(payload)
    if not roots:
        print(f"! No tree for {doc_id} (status={payload.get('status')}).", file=sys.stderr)
        return 1

    stats = retrieval.tree_stats(roots)
    print(f"doc_id: {doc_id}")
    print(
        f"nodes: {stats['nodes']} | depth: {stats['max_depth']} | "
        f"with text: {stats['nodes_with_text']} | chars: {stats['total_text_chars']:,} | "
        f"pages: {stats['last_page']}\n"
    )

    if args.node:
        index = retrieval.index_nodes(roots)
        entry = index.get(args.node)
        if not entry:
            print(f"! No node {args.node}", file=sys.stderr)
            return 1
        node = entry["node"]
        print(" > ".join(entry["path"] + (str(node.get("title", "")),)))
        print(f"pages: {retrieval.page_label(node)}")
        summary = retrieval.node_summary(node)
        if summary:
            print(f"\nsummary:\n{summary}")
        body = retrieval.collect_text(node, max_chars=1_000_000 if args.text else 1500)
        print(f"\ntext ({len(body):,} chars):\n{body}")
        return 0

    if args.grep:
        needle = args.grep.lower()
        for node, depth, path in retrieval.walk(roots):
            title = str(node.get("title", ""))
            if needle in title.lower():
                crumb = " > ".join(p for p in path if p)
                print(
                    f"[{node.get('node_id')}] {title} ({retrieval.page_label(node)})"
                    + (f"\n    under: {crumb}" if crumb else "")
                )
        return 0

    print(
        retrieval.render_outline(
            roots, max_depth=args.depth, include_summary=args.summaries
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
