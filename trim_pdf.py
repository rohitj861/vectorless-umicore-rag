"""Produce a <=200-page copy of the annual report so it fits the PageIndex Free Trial.

The Free Trial allows 200 active pages; the Umicore 2025 report is 220. This drops
20 pages that carry no analytical content for a financial-QA system - covers, the
static contents page, marketing spreads, the ESRS cross-reference appendices and
the glossary. Every financial statement, note, segment review, governance and
remuneration section is kept intact.

    python trim_pdf.py            # writes 'Umicore Annual Report 2025 (trimmed).pdf'
    python trim_pdf.py --list     # show what would be dropped, write nothing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

load_dotenv()

# 1-based, inclusive page ranges to drop, with the reason each is safe to lose.
DROP_RANGES: list[tuple[int, int, str]] = [
    (1, 2, "front cover / title spread - no content"),
    (3, 3, "static contents page - PageIndex builds its own tree"),
    (5, 6, "'Discover Umicore' marketing spread - no financial data"),
    (200, 204, "ESRS appendices: datapoint lists and disclosure index (cross-references only)"),
    (211, 213, "limited assurance report on the sustainability statements"),
    (214, 218, "glossary - definitions, not reported figures"),
    (219, 220, "additional information / back cover"),
]

KEPT_HIGHLIGHTS = [
    "p14-27   Performance, segment reviews, financial review",
    "p29-60   Corporate governance, remuneration report, principal risks",
    "p62-140  Financial statements + all notes F1-F43 + auditor's report",
    "p141-199 Sustainability statements (ESRS E1-E5, S1-S2, G1)",
]


def dropped_pages() -> set[int]:
    pages: set[int] = set()
    for start, end, _ in DROP_RANGES:
        pages.update(range(start, end + 1))
    return pages


def dropped_pages_setting() -> str:
    """The PAGEINDEX_DROPPED_PAGES value matching DROP_RANGES.

    Citations are only correct if this exact list reaches the app, so it is
    derived from DROP_RANGES rather than written out a second time by hand.
    """
    return ",".join(
        f"{start}" if start == end else f"{start}-{end}"
        for start, end, _ in sorted(DROP_RANGES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim the PDF under the page cap.")
    parser.add_argument("pdf", nargs="?", default=os.getenv("PAGEINDEX_DEFAULT_PDF"))
    parser.add_argument("--out", help="Output path (default: '<name> (trimmed).pdf')")
    parser.add_argument("--limit", type=int, default=200, help="Page cap to fit under.")
    parser.add_argument("--list", action="store_true", help="Show the plan, write nothing.")
    args = parser.parse_args()

    src = Path(args.pdf)
    if not src.exists():
        print(f"! Not found: {src}")
        return 1

    reader = PdfReader(str(src))
    total = len(reader.pages)
    if total <= args.limit:
        print(f"! {src.name} already has {total} pages, at or under the "
              f"{args.limit}-page cap - there is nothing to trim.")
        print("  Trimming it again would drop a SECOND set of pages and silently")
        print("  corrupt every page citation. Point this at the ORIGINAL report:")
        print(f'    python trim_pdf.py "Umicore Annual Report 2025.pdf"')
        print("  (PAGEINDEX_DEFAULT_PDF points at the trimmed copy, so the bare")
        print("   command picks the wrong file.)")
        return 1

    drop = dropped_pages()
    keep = [p for p in range(1, total + 1) if p not in drop]

    print(f"source : {src.name}  ({total} pages)")
    print(f"target : <= {args.limit} pages\n")
    print("dropping:")
    for start, end, why in DROP_RANGES:
        span = f"p{start}" if start == end else f"p{start}-{end}"
        count = end - start + 1
        print(f"  {span:<12} {count:>2} page(s)  {why}")
    print(f"\n  total dropped: {len(drop)}  ->  result: {len(keep)} pages")
    print("\nkeeping (unchanged):")
    for line in KEPT_HIGHLIGHTS:
        print(f"  {line}")

    if len(keep) > args.limit:
        print(f"\n! Still {len(keep)} pages, over the {args.limit} cap.")
        return 1

    if args.list:
        print(f"\nsetting: PAGEINDEX_DROPPED_PAGES={dropped_pages_setting()}")
        print("(--list given, nothing written)")
        return 0

    out = Path(args.out) if args.out else src.with_name(f"{src.stem} (trimmed).pdf")
    writer = PdfWriter()
    for page_no in keep:
        writer.add_page(reader.pages[page_no - 1])
    with out.open("wb") as handle:
        writer.write(handle)

    size_mb = out.stat().st_size / 1_048_576
    print(f"\nwrote: {out.name}  ({len(keep)} pages, {size_mb:.1f} MB)")
    print(
        "\nPut this in .env so answers cite the ORIGINAL report's pages:\n"
        f"  PAGEINDEX_DROPPED_PAGES={dropped_pages_setting()}"
    )
    print(f"\nNext:  python ingest.py \"{out.name}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
