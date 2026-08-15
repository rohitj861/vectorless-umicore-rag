"""Step 1+2 — submit the PDF to PageIndex, get the doc_id, wait for the tree.

    python ingest.py                                  # uses PAGEINDEX_DEFAULT_PDF
    python ingest.py "Umicore Annual Report 2025.pdf"
    python ingest.py --doc-id pi-abc123               # just fetch an existing tree
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

import retrieval
from pageindex_client import PageIndexClient, PageIndexError

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a PDF to PageIndex.")
    parser.add_argument("pdf", nargs="?", default=os.getenv("PAGEINDEX_DEFAULT_PDF"))
    parser.add_argument("--doc-id", help="Skip upload; fetch the tree for this doc_id.")
    parser.add_argument("--poll", type=int, default=10, help="Seconds between polls.")
    parser.add_argument("--timeout", type=int, default=2400, help="Give up after N seconds.")
    parser.add_argument("--no-summary", action="store_true", help="Skip node summaries.")
    parser.add_argument("--force", action="store_true", help="Re-upload even if cached.")
    args = parser.parse_args()

    try:
        client = PageIndexClient()
    except PageIndexError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    doc_id = args.doc_id
    if not doc_id:
        if not args.pdf:
            print("! No PDF given and PAGEINDEX_DEFAULT_PDF is unset.", file=sys.stderr)
            return 1
        cached = client.doc_id_for(args.pdf)
        if cached and not args.force:
            doc_id = cached
            print(f"= Already submitted: {args.pdf}\n  doc_id: {doc_id}")
        else:
            print(f"> Uploading {args.pdf} ...")
            doc_id = client.submit_document(args.pdf)
            print(f"+ doc_id: {doc_id}")

    print("> Waiting for the tree to be built (this can take a few minutes) ...")

    def progress(status: str, elapsed: float) -> None:
        # flush so the status is visible when stdout is redirected to a log
        print(f"  [{elapsed:6.0f}s] status={status}", flush=True)

    payload = client.wait_for_tree(
        doc_id,
        poll_seconds=args.poll,
        timeout_seconds=args.timeout,
        node_summary=not args.no_summary,
        on_progress=progress,
    )

    roots = retrieval.normalise_tree(payload)
    stats = retrieval.tree_stats(roots)
    print("\n=== Done ===")
    print(f"doc_id     : {doc_id}")
    print(f"tree cached: {client.tree_path(doc_id)}")
    print(
        f"tree       : {stats['nodes']} nodes, depth {stats['max_depth']}, "
        f"{stats['nodes_with_text']} with text, ~{stats['total_text_chars']:,} chars, "
        f"last page {stats['last_page']}"
    )
    print("\nNext: python inspect_tree.py --doc-id " + doc_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
