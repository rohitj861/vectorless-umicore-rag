"""Steps 4-6 — tree search, node text extraction, answer generation.

    python ask.py "How did Battery Materials affect group revenue in 2025?"
    python ask.py --sample 0
    python ask.py --list-samples
"""

from __future__ import annotations

import argparse
import sys

import pipeline
from inspect_tree import newest_cached_doc_id
from pageindex_client import PageIndexClient, PageIndexError


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question, vectorlessly.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--doc-id")
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--sample", type=int, help="Use one of the built-in multi-hop queries.")
    parser.add_argument("--list-samples", action="store_true")
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved text.")
    args = parser.parse_args()

    if args.list_samples:
        for i, q in enumerate(pipeline.SAMPLE_MULTIHOP_QUERIES):
            print(f"[{i}] {q}\n")
        return 0

    query = args.query
    if args.sample is not None:
        query = pipeline.SAMPLE_MULTIHOP_QUERIES[args.sample]
    if not query:
        parser.error("give a query, --sample N, or --list-samples")

    try:
        client = PageIndexClient()
    except PageIndexError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    doc_id = args.doc_id or newest_cached_doc_id(client)
    if not doc_id:
        print("! Nothing indexed yet. Run: python ingest.py", file=sys.stderr)
        return 1

    print(f"doc_id : {doc_id}")
    print(f"query  : {query}\n")

    result = pipeline.ask(query, doc_id, max_nodes=args.max_nodes)

    print("--- tree search reasoning ---")
    print(result["thinking"] or "(none returned)")
    print(f"\n--- selected {len(result['sources'])} node(s) ---")
    for src in result["sources"]:
        conf = src.get("confidence")
        conf_s = f" conf={conf}" if conf is not None else ""
        print(f"[{src['node_id']}] {src['breadcrumb']} ({src['pages']}) "
              f"{src['chars']:,} chars{conf_s}")
        if src.get("why"):
            print(f"    why: {src['why']}")
    if result["hallucinated_ids"]:
        print(f"(ignored non-existent ids: {result['hallucinated_ids']})")

    if args.show_context:
        print(f"\n--- context ({len(result['context']):,} chars) ---")
        print(result["context"])

    print("\n--- answer ---")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
