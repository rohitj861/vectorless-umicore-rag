"""Verify that .env is being read and that both API keys actually work.

    python check_setup.py

Run this right after pasting your keys. It makes one tiny call per service,
so it costs essentially nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")


def mask(value: str) -> str:
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:7]}...{value[-4:]}  ({len(value)} chars)"


def looks_wrong(value: str) -> str | None:
    """Catch the mistakes people actually make when pasting keys."""
    if value != value.strip():
        return "has leading/trailing whitespace - delete the spaces"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return "is wrapped in quotes - remove them, .env needs bare values"
    if value.startswith("<") or value.endswith(">"):
        return "still has angle brackets around it - paste the key only"
    if "your-" in value.lower() or "here" in value.lower():
        return "is still the placeholder text, not a real key"
    if " " in value:
        return "contains a space - the key was probably pasted incompletely"
    return None


def check_file() -> bool:
    print("1. Looking for the .env file")
    print(f"   expected at: {ENV_PATH}")
    if not ENV_PATH.exists():
        print("   [X] NOT FOUND.")
        stray = list(ENV_PATH.parent.glob(".env*"))
        if stray:
            print(f"       Found instead: {[p.name for p in stray]}")
            print("       Windows may have saved it as '.env.txt'. Rename it to '.env'.")
        return False
    print(f"   [OK] found ({ENV_PATH.stat().st_size} bytes)")
    return True


def check_keys() -> bool:
    print("\n2. Reading the keys")
    ok = True
    for name, hint in (
        ("PAGEINDEX_API_KEY", "https://dash.pageindex.ai -> API Keys"),
        ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    ):
        raw = os.getenv(name, "")
        if not raw:
            print(f"   [X] {name} is empty. Get one at {hint}")
            ok = False
            continue
        problem = looks_wrong(raw)
        if problem:
            print(f"   [X] {name} {problem}")
            ok = False
            continue
        print(f"   [OK] {name} = {mask(raw)}")
    return ok


def check_openai() -> bool:
    print("\n3. Testing the OpenAI key (live)")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001 - surface whatever the SDK says
        print(f"   [X] failed: {type(exc).__name__}: {str(exc)[:200]}")
        print("       401 -> wrong/revoked key. 429 -> no credit on the account.")
        return False
    print(f"   [OK] key works ({len(models)} models visible)")

    for var, default in (
        ("PAGEINDEX_SEARCH_MODEL", "gpt-4.1-mini"),
        ("PAGEINDEX_ANSWER_MODEL", "gpt-4.1"),
    ):
        wanted = os.getenv(var, default)
        mark = "[OK]" if wanted in models else "[!] "
        note = "" if wanted in models else "  <- not available to this account"
        print(f"   {mark} {var} = {wanted}{note}")
    return True


def check_pageindex() -> bool:
    print("\n4. Testing the PageIndex key (live)")
    import requests

    base = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai").rstrip("/")
    try:
        response = requests.get(
            f"{base}/doc/setup-check-not-a-real-doc/",
            headers={"api_key": os.getenv("PAGEINDEX_API_KEY", "")},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   [X] could not reach {base}: {type(exc).__name__}: {exc}")
        return False

    if response.status_code in (401, 403):
        print(f"   [X] key rejected (HTTP {response.status_code}): {response.text[:200]}")
        return False
    # 404/400 on a made-up doc_id means auth passed and it just looked it up.
    print(f"   [OK] key accepted (HTTP {response.status_code} on a dummy doc_id, as expected)")
    return True


def main() -> int:
    print("=" * 70)
    print("Setup check - vectorless RAG")
    print("=" * 70)

    file_ok = check_file()
    load_dotenv(ENV_PATH, override=True)
    keys_ok = check_keys()

    if not (file_ok and keys_ok):
        print("\nFix the above, then run this again.")
        return 1

    openai_ok = check_openai()
    pageindex_ok = check_pageindex()

    print("\n" + "=" * 70)
    if openai_ok and pageindex_ok:
        print("ALL GOOD. Next:  python ingest.py     (or:  streamlit run app.py)")
        return 0
    print("Something is still wrong - see the [X] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
