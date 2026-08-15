"""Answer generation over the sections PageIndex retrieval handed us."""

from __future__ import annotations

import os
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ANSWER_SYSTEM_PROMPT = """You are a financial analyst answering questions about a \
single company report. You are given the verbatim text of the specific sections \
that a document-structure search selected, each headed by its node id, section \
path and page range.

Rules:
- Answer ONLY from the provided sections. If they do not contain the answer, say \
exactly what is missing and which section would likely hold it. Never guess a number.
- Every figure you state must carry its unit, currency and period (e.g. \
"€1,234 million, FY2025"). Keep the report's own scale — do not rescale.
- Distinguish reported (IFRS) figures from adjusted / alternative performance \
measures, and say which one you used.
- If the sections contain MORE THAN ONE measure that answers the question (e.g. \
IFRS turnover and a revenue-excluding-metals or adjusted figure), report ALL of \
them, state each one's label exactly as the report labels it, and explain in one \
line why they differ and which one management steers on. Giving only the IFRS \
line when an alternative measure is also present is a wrong answer, because the \
two can differ by multiples and imply opposite trends.
- When the question spans several sections, answer it as one connected narrative \
and make the link between the sections explicit.
- Cite as you go using the section title and page, like (Segment review — Battery \
Materials, p. 42). Do not invent page numbers.
- Show any arithmetic you perform (e.g. "1,240 − 1,100 = 140, i.e. +12.7%").
- Be concise and factual. Use short paragraphs or bullets. No preamble."""


def get_openai_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")
    return OpenAI(api_key=key)


def _messages(query: str, context: str, doc_label: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document: {doc_label}\n\n"
                f"Retrieved sections:\n{'=' * 70}\n{context}\n{'=' * 70}\n\n"
                f"Question: {query}"
            ),
        },
    ]


def answer(
    query: str,
    context: str,
    client: OpenAI | None = None,
    model: str | None = None,
    doc_label: str = "annual report",
    temperature: float = 0.1,
) -> str:
    client = client or get_openai_client()
    model = model or os.getenv("PAGEINDEX_ANSWER_MODEL") or "gpt-4.1"
    if not context.strip():
        return "No section text was retrieved, so there is nothing to answer from."
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=_messages(query, context, doc_label),
    )
    return response.choices[0].message.content or ""


def answer_stream(
    query: str,
    context: str,
    client: OpenAI | None = None,
    model: str | None = None,
    doc_label: str = "annual report",
    temperature: float = 0.1,
) -> Iterator[str]:
    """Token stream, for Streamlit's ``st.write_stream``."""
    client = client or get_openai_client()
    model = model or os.getenv("PAGEINDEX_ANSWER_MODEL") or "gpt-4.1"
    if not context.strip():
        yield "No section text was retrieved, so there is nothing to answer from."
        return
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        stream=True,
        messages=_messages(query, context, doc_label),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def context_budget() -> int:
    raw: Any = os.getenv("PAGEINDEX_CONTEXT_CHAR_BUDGET", "90000")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 90000
