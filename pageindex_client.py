"""Thin REST client for the PageIndex document API.

PageIndex turns a PDF into a *hierarchical tree* of sections (a machine-readable
table of contents where every node carries its own text and page range). That
tree is what makes retrieval "vectorless": instead of embedding chunks, an LLM
reads the outline and reasons about which sections hold the answer.

Endpoints used
    POST /doc/                 multipart upload  -> {"doc_id": "..."}
    GET  /doc/{doc_id}/        status + result   -> {"status": "...", "result": [...]}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.pageindex.ai"
REGISTRY_FILENAME = "documents.json"


class PageIndexError(RuntimeError):
    """Raised when the PageIndex API returns something we cannot use."""


class PageIndexClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("PAGEINDEX_API_KEY", "")
        if not self.api_key:
            raise PageIndexError(
                "PAGEINDEX_API_KEY is not set. Put it in your .env file."
            )
        self.base_url = (base_url or os.getenv("PAGEINDEX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir or os.getenv("PAGEINDEX_CACHE_DIR") or "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    # -- low level ---------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        return {"api_key": self.api_key}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _json_or_raise(response: requests.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            body = response.text[:500]
            if response.status_code == 403 and "LimitReached" in body:
                raise PageIndexError(
                    "PageIndex 403 LimitReached - your account quota blocked this upload.\n"
                    "  Indexing costs 1 credit per page, and each plan also caps how many\n"
                    "  pages you may keep indexed at once (Free Trial: 200 active pages).\n"
                    "  A 220-page report exceeds the Free Trial cap by 20 pages.\n"
                    "  Options: upgrade at https://dash.pageindex.ai (Subscription), or\n"
                    "  shrink the PDF under the cap with:  python trim_pdf.py"
                )
            if response.status_code in (401, 403):
                raise PageIndexError(
                    f"PageIndex {response.status_code} - key rejected or quota exceeded: {body}"
                )
            raise PageIndexError(f"PageIndex API {response.status_code}: {body}")
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise PageIndexError(f"Non-JSON response: {response.text[:500]}") from exc

    # -- step 1: submit ----------------------------------------------------
    def submit_document(self, pdf_path: str | os.PathLike[str]) -> str:
        """Upload a PDF and return its ``doc_id``."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        with pdf_path.open("rb") as handle:
            response = requests.post(
                self._url("/doc/"),
                headers=self._headers,
                files={"file": (pdf_path.name, handle, "application/pdf")},
                timeout=self.timeout,
            )
        payload = self._json_or_raise(response)
        doc_id = payload.get("doc_id")
        if not doc_id:
            raise PageIndexError(f"No doc_id in response: {payload}")

        self._remember(pdf_path.name, doc_id)
        return doc_id

    # -- step 2: status / tree --------------------------------------------
    def get_doc(
        self,
        doc_id: str,
        result_type: str = "tree",
        node_summary: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"type": result_type}
        if result_type == "tree":
            # The API spells this `summary`; the SDK calls it `node_summary`.
            params["summary"] = "true" if node_summary else "false"
        response = requests.get(
            self._url(f"/doc/{doc_id}/"),
            headers=self._headers,
            params=params,
            timeout=self.timeout,
        )
        return self._json_or_raise(response)

    def wait_for_tree(
        self,
        doc_id: str,
        poll_seconds: int = 10,
        timeout_seconds: int = 1800,
        node_summary: bool = True,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """Poll until processing completes, then return the full payload."""
        started = time.monotonic()
        while True:
            payload = self.get_doc(doc_id, "tree", node_summary=node_summary)
            status = str(payload.get("status", "unknown"))
            elapsed = time.monotonic() - started
            if on_progress:
                on_progress(status, elapsed)
            if status == "completed":
                self.save_tree(doc_id, payload)
                return payload
            if status in {"failed", "error"}:
                raise PageIndexError(f"Processing failed for {doc_id}: {payload}")
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"{doc_id} still '{status}' after {timeout_seconds}s. "
                    "Re-run later — the doc_id stays valid."
                )
            time.sleep(poll_seconds)

    # -- local cache -------------------------------------------------------
    def tree_path(self, doc_id: str) -> Path:
        return self.cache_dir / f"{doc_id}.tree.json"

    def save_tree(self, doc_id: str, payload: dict[str, Any]) -> Path:
        path = self.tree_path(doc_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_tree(self, doc_id: str) -> dict[str, Any] | None:
        path = self.tree_path(doc_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_or_fetch_tree(self, doc_id: str, node_summary: bool = True) -> dict[str, Any]:
        cached = self.load_tree(doc_id)
        if cached and cached.get("result"):
            return cached
        payload = self.get_doc(doc_id, "tree", node_summary=node_summary)
        if payload.get("status") == "completed":
            self.save_tree(doc_id, payload)
        return payload

    # -- doc_id registry ---------------------------------------------------
    @property
    def registry_path(self) -> Path:
        return self.cache_dir / REGISTRY_FILENAME

    def registry(self) -> dict[str, str]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _remember(self, filename: str, doc_id: str) -> None:
        registry = self.registry()
        registry[filename] = doc_id
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def doc_id_for(self, filename: str) -> str | None:
        return self.registry().get(Path(filename).name)
