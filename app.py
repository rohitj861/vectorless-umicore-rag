"""Streamlit UI for vectorless RAG over the Umicore annual report.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import generation
import pipeline
import retrieval
from pageindex_client import PageIndexClient, PageIndexError

load_dotenv()


def bridge_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ.

    Locally, config comes from `.env` via python-dotenv. On Streamlit Community
    Cloud there is no `.env` — secrets are pasted as TOML in the deploy dialog
    and surface as `st.secrets`. Every module here reads `os.getenv`, so we
    bridge once at startup and the rest of the codebase is unchanged.
    `.env` wins when both exist, so local runs behave exactly as before.
    """
    keys = (
        "PAGEINDEX_API_KEY",
        "PAGEINDEX_BASE_URL",
        "OPENAI_API_KEY",
        "PAGEINDEX_SEARCH_MODEL",
        "PAGEINDEX_ANSWER_MODEL",
        "PAGEINDEX_DEFAULT_PDF",
        "PAGEINDEX_DOC_ID",
        "PAGEINDEX_PAGE_OFFSET",
        "PAGEINDEX_OUTLINE_CHAR_BUDGET",
        "PAGEINDEX_CONTEXT_CHAR_BUDGET",
        "PAGEINDEX_ALLOW_UPLOAD",
        "APP_PASSWORD",
    )
    for key in keys:
        if os.getenv(key):
            continue
        try:
            value = st.secrets[key]
        except Exception:
            continue
        if value not in (None, ""):
            os.environ[key] = str(value)


bridge_secrets()

st.set_page_config(page_title="Vectorless RAG — Umicore", page_icon="📑", layout="wide")

DEFAULT_PDF = os.getenv("PAGEINDEX_DEFAULT_PDF", "Umicore Annual Report 2025.pdf")


def truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# On a shared deployment, uploading is what costs money and page quota.
ALLOW_UPLOAD = truthy("PAGEINDEX_ALLOW_UPLOAD", True)


def password_gate() -> bool:
    """Optional shared-password gate. No APP_PASSWORD set => no gate."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("_auth_ok"):
        return True
    st.title("📑 Vectorless RAG — Umicore")
    st.caption("This deployment is password protected.")
    entered = st.text_input("Password", type="password")
    if entered:
        # constant-time compare so the check does not leak length by timing
        import hmac

        if hmac.compare_digest(entered, expected):
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
    return False


password_gate()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def get_client() -> PageIndexClient | None:
    try:
        return PageIndexClient()
    except PageIndexError as exc:
        st.error(str(exc))
        return None


@st.cache_data(show_spinner=False)
def load_tree_cached(doc_id: str, _mtime: float) -> dict:
    """Tree payload, re-read whenever the cached file changes."""
    return PageIndexClient().get_or_fetch_tree(doc_id)


def load_roots(doc_id: str) -> list[dict]:
    client = PageIndexClient()
    path = client.tree_path(doc_id)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    payload = load_tree_cached(doc_id, mtime)
    if payload.get("status") not in (None, "completed"):
        st.warning(f"Document {doc_id} is still `{payload.get('status')}`.")
        return []
    return retrieval.normalise_tree(payload)


def known_docs(client: PageIndexClient) -> dict[str, str]:
    """label -> doc_id, from the registry plus any orphan cached trees."""
    docs = {f"{name}  ({doc_id})": doc_id for name, doc_id in client.registry().items()}
    known_ids = set(docs.values())
    for path in client.cache_dir.glob("*.tree.json"):
        doc_id = path.name.removesuffix(".tree.json")
        if doc_id not in known_ids:
            docs[f"(cached tree)  {doc_id}"] = doc_id
    return docs


def render_tree_node(node: dict, depth: int = 0) -> None:
    title = str(node.get("title", "(untitled)")).strip() or "(untitled)"
    node_id = node.get("node_id", "?")
    kids = retrieval.children(node)
    label = f"`{node_id}`  {title}  ·  {retrieval.page_label(node)}"
    if kids:
        label += f"  ·  {len(kids)} sub"
    with st.expander(label, expanded=False):
        summary = retrieval.node_summary(node)
        if summary:
            st.caption(summary)
        text = retrieval.node_text(node)
        if text:
            st.text_area(
                f"text · {len(text):,} chars",
                text[:20000],
                height=200,
                key=f"txt-{node_id}-{depth}",
                disabled=True,
            )
        for child in kids:
            render_tree_node(child, depth + 1)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

st.sidebar.title("📑 Vectorless RAG")
st.sidebar.caption("PageIndex tree search — no embeddings, no vector DB.")

pi_ok = bool(os.getenv("PAGEINDEX_API_KEY"))
oa_ok = bool(os.getenv("OPENAI_API_KEY"))
st.sidebar.markdown(
    f"- PageIndex key: {'✅' if pi_ok else '❌ missing'}\n"
    f"- OpenAI key: {'✅' if oa_ok else '❌ missing'}"
)
if not (pi_ok and oa_ok):
    st.sidebar.warning("Fill in `.env`, then restart the app.")

search_model = st.sidebar.text_input(
    "Tree-search model", os.getenv("PAGEINDEX_SEARCH_MODEL", "gpt-4.1-mini")
)
answer_model = st.sidebar.text_input(
    "Answer model", os.getenv("PAGEINDEX_ANSWER_MODEL", "gpt-4.1")
)
max_nodes = st.sidebar.slider("Max nodes to retrieve", 1, 15, 8)
char_budget = st.sidebar.slider(
    "Context char budget", 10_000, 200_000, generation.context_budget(), step=10_000
)

client = get_client() if pi_ok else None

selected_doc_id = None
if client:
    docs = known_docs(client)
    if docs:
        label = st.sidebar.selectbox("Document", list(docs.keys()))
        selected_doc_id = docs[label]
        st.sidebar.code(selected_doc_id, language=None)
    else:
        st.sidebar.info("No document indexed yet — use the **Index** tab.")

manual_id = st.sidebar.text_input("…or paste a doc_id", "")
if manual_id.strip():
    selected_doc_id = manual_id.strip()

st.session_state["doc_id"] = selected_doc_id


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------

tab_index, tab_tree, tab_ask = st.tabs(
    ["1 · Index", "2 · Inspect tree", "3 · Ask"]
)

# ---- 1. index ------------------------------------------------------------
with tab_index:
    st.header("Submit a PDF to PageIndex")
    st.write(
        "PageIndex parses the PDF into a hierarchical tree of sections. "
        "The `doc_id` it returns is the handle for every later step."
    )

    # NOTE: never st.stop() inside a tab - it halts the whole script run and
    # blanks the other tabs. Gate with a plain conditional instead.
    if not ALLOW_UPLOAD:
        st.info(
            "Indexing is disabled on this deployment (`PAGEINDEX_ALLOW_UPLOAD=false`). "
            "Each upload consumes PageIndex credits and counts against the account's "
            "active-page cap, so it is switched off for shared instances. "
            "The document below is already indexed and ready to query."
        )
        if selected_doc_id:
            st.code(selected_doc_id, language=None)

    local_pdfs = sorted(str(p.name) for p in Path(".").glob("*.pdf"))
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("From this folder")
        if not ALLOW_UPLOAD:
            st.caption("Disabled on this deployment.")
        if local_pdfs:
            choice = st.selectbox(
                "PDF",
                local_pdfs,
                index=local_pdfs.index(DEFAULT_PDF) if DEFAULT_PDF in local_pdfs else 0,
            )
            force = st.checkbox("Re-upload even if already submitted", value=False)
            if st.button(
                "Submit to PageIndex",
                type="primary",
                disabled=not client or not ALLOW_UPLOAD,
            ):
                assert client is not None
                existing = client.doc_id_for(choice)
                if existing and not force:
                    st.info(f"Already submitted. doc_id: `{existing}`")
                    doc_id = existing
                else:
                    with st.spinner("Uploading…"):
                        doc_id = client.submit_document(choice)
                    st.success(f"Submitted. doc_id: `{doc_id}`")
                st.session_state["pending_doc_id"] = doc_id
        else:
            st.info("No PDF found in the working directory.")

    with col_b:
        st.subheader("Or upload one")
        if not ALLOW_UPLOAD:
            st.caption("Disabled on this deployment.")
        upload = st.file_uploader("PDF", type=["pdf"], disabled=not ALLOW_UPLOAD)
        if upload is not None and st.button(
            "Upload & submit", disabled=not client or not ALLOW_UPLOAD
        ):
            assert client is not None
            tmp = client.cache_dir / upload.name
            tmp.write_bytes(upload.getbuffer())
            with st.spinner("Uploading…"):
                doc_id = client.submit_document(tmp)
            st.success(f"Submitted. doc_id: `{doc_id}`")
            st.session_state["pending_doc_id"] = doc_id

    pending = st.session_state.get("pending_doc_id") or selected_doc_id
    if pending and client:
        st.divider()
        st.subheader("Processing status")
        st.code(pending, language=None)
        if st.button("Check / wait for the tree"):
            status_box = st.empty()
            bar = st.progress(0.0)
            start = time.monotonic()
            try:
                while True:
                    payload = client.get_doc(pending, "tree", node_summary=True)
                    status = payload.get("status", "unknown")
                    elapsed = time.monotonic() - start
                    status_box.write(f"status: **{status}** · {elapsed:.0f}s elapsed")
                    bar.progress(min(elapsed / 600, 0.99))
                    if status == "completed":
                        client.save_tree(pending, payload)
                        bar.progress(1.0)
                        load_tree_cached.clear()
                        roots = retrieval.normalise_tree(payload)
                        stats = retrieval.tree_stats(roots)
                        st.success(
                            f"Tree ready — {stats['nodes']} nodes, depth "
                            f"{stats['max_depth']}, {stats['total_text_chars']:,} chars."
                        )
                        break
                    if status in {"failed", "error"}:
                        st.error(f"Processing failed: {payload}")
                        break
                    if elapsed > 1800:
                        st.warning("Still processing. Come back and check again later.")
                        break
                    time.sleep(8)
            except PageIndexError as exc:
                st.error(str(exc))

# ---- 2. tree -------------------------------------------------------------
with tab_tree:
    st.header("The document tree")
    doc_id = st.session_state.get("doc_id")
    if not doc_id:
        st.info("Pick or index a document first.")
    else:
        roots = load_roots(doc_id)
        if roots:
            stats = retrieval.tree_stats(roots)
            cols = st.columns(5)
            cols[0].metric("Nodes", stats["nodes"])
            cols[1].metric("Depth", stats["max_depth"])
            cols[2].metric("With text", stats["nodes_with_text"])
            cols[3].metric("Text chars", f"{stats['total_text_chars']:,}")
            cols[4].metric("Last page", stats["last_page"])

            view = st.radio(
                "View", ["Expandable", "Outline", "Raw JSON"], horizontal=True
            )
            needle = st.text_input("Filter by title contains", "")

            if view == "Expandable":
                shown = 0
                for node, depth, path in retrieval.walk(roots):
                    if depth > 0:
                        continue
                    render_tree_node(node)
                    shown += 1
                if needle:
                    st.divider()
                    st.subheader(f"Title matches for “{needle}”")
                    for node, _, path in retrieval.walk(roots):
                        title = str(node.get("title", ""))
                        if needle.lower() in title.lower():
                            crumb = " > ".join(p for p in path if p)
                            st.markdown(
                                f"`{node.get('node_id')}` **{title}** · "
                                f"{retrieval.page_label(node)}"
                                + (f"  \n<small>{crumb}</small>" if crumb else ""),
                                unsafe_allow_html=True,
                            )
            elif view == "Outline":
                depth = st.slider("Depth", 0, max(stats["max_depth"] - 1, 0), 2)
                with_summaries = st.checkbox("Show node summaries", value=False)
                st.code(
                    retrieval.render_outline(
                        roots, max_depth=depth, include_summary=with_summaries
                    ),
                    language=None,
                )
            else:
                st.json(roots, expanded=False)

# ---- 3. ask --------------------------------------------------------------
with tab_ask:
    st.header("Ask a question")
    doc_id = st.session_state.get("doc_id")
    if not doc_id:
        st.info("Pick or index a document first.")
    elif not oa_ok:
        st.warning("Set `OPENAI_API_KEY` in `.env` to run retrieval and generation.")
    else:
        st.caption(
            "Retrieval = an LLM reads the tree and picks node ids. "
            "Generation = a second call reads only those sections."
        )
        sample = st.selectbox(
            "Multi-section starter questions",
            ["—"] + pipeline.SAMPLE_MULTIHOP_QUERIES,
        )
        default_q = "" if sample == "—" else sample
        query = st.text_area("Question", value=default_q, height=110)

        if st.button("Run", type="primary", disabled=not query.strip()):
            roots = load_roots(doc_id)
            if not roots:
                st.stop()
            openai_client = generation.get_openai_client()

            with st.status("Reasoning over the tree…", expanded=False) as status:
                result = pipeline.retrieve(
                    query,
                    roots,
                    openai_client,
                    search_model=search_model,
                    max_nodes=max_nodes,
                    char_budget=char_budget,
                )
                status.update(
                    label=f"Selected {len(result['sources'])} section(s)", state="complete"
                )

            if result["thinking"]:
                with st.expander("Tree-search reasoning", expanded=True):
                    st.write(result["thinking"])

            st.subheader("Retrieved sections")
            if not result["sources"]:
                st.error("No sections selected — try rephrasing the question.")
                st.stop()
            for src in result["sources"]:
                conf = src.get("confidence")
                header = (
                    f"`{src['node_id']}` · {src['breadcrumb']} · {src['pages']} · "
                    f"{src['chars']:,} chars"
                    + (f" · conf {conf}" if conf is not None else "")
                )
                with st.expander(header, expanded=False):
                    if src.get("why"):
                        st.caption(f"Why selected: {src['why']}")
                    st.text(src["text"][:20000])
            if result["hallucinated_ids"]:
                st.caption(f"Ignored ids not in the tree: {result['hallucinated_ids']}")
            st.caption(f"Context assembled: {len(result['context']):,} chars")

            st.subheader("Answer")
            st.write_stream(
                generation.answer_stream(
                    query,
                    result["context"],
                    client=openai_client,
                    model=answer_model,
                    doc_label=DEFAULT_PDF,
                )
            )

            st.download_button(
                "Download retrieval trace (JSON)",
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "query": query,
                        "search_model": search_model,
                        "answer_model": answer_model,
                        "thinking": result["thinking"],
                        "sources": [
                            {k: v for k, v in s.items() if k != "text"}
                            for s in result["sources"]
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="retrieval_trace.json",
                mime="application/json",
            )
