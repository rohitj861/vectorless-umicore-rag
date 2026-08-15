# Vectorless RAG over the Umicore Annual Report 2025

No embeddings. No vector database. No chunking.

[PageIndex](https://pageindex.ai) parses the 220-page PDF into a **hierarchical
tree of sections** — a machine-readable table of contents where every node
carries its own title, page range, summary and text. Retrieval is then a
*reasoning* step: an LLM reads that outline and decides which `node_id`s to
open, the way an analyst flips to the right pages. Only those sections are sent
to the answer model.

This matters for a financial document, where the answer to one question is
usually split across the management commentary, the income statement and a note
to the accounts.

```
PDF ──POST /doc/──▶ doc_id ──GET /doc/{id}?type=tree──▶ section tree
                                                            │
                    query ──▶ tree search (LLM reads outline, returns node_ids)
                                                            │
                              extract node text ──▶ answer model ──▶ cited answer
```

## Setup

```powershell
pip install -r requirements.txt
```

Requires **Python 3.11+** and **Streamlit 1.31+** (`st.write_stream`). See the warning under [Run](#run) if you have more than one Python installed.

Then open `.env` and paste your two keys:

```ini
PAGEINDEX_API_KEY=...      # https://dash.pageindex.ai  -> API Keys
OPENAI_API_KEY=...         # https://platform.openai.com/api-keys
```

`.env` is git-ignored; `.env.example` documents every setting.

Verify both keys before doing anything else:

```powershell
python check_setup.py
```

It masks the keys, catches the usual paste mistakes (quotes, stray spaces,
placeholder text left in), makes one tiny live call per service, and tells you
whether your chosen models are available on the account.

### The 200-page cap

The PageIndex Free Trial allows **200 active pages** and 200 credits (indexing
costs 1 credit per page). The Umicore 2025 report is **220 pages**, so a direct
upload fails with `403 LimitReached`.

`trim_pdf.py` builds a 200-page copy by dropping 20 pages that carry no
analytical content — covers, the static contents page, a marketing spread, the
ESRS cross-reference appendices and the glossary:

```powershell
python trim_pdf.py --list   # show exactly what would be dropped
python trim_pdf.py          # write 'Umicore Annual Report 2025 (trimmed).pdf'
```

Everything analytical survives: all segment reviews, the full financial
statements with notes F1–F43, the auditor's report, governance, the remuneration
report, principal risks and the ESRS sustainability statements.

**Page numbers shift — and are corrected automatically.** Trimming drops 5 pages
before original p7, so everything after it sits 5 pages earlier in the indexed
file. `PAGEINDEX_PAGE_OFFSET=5` adds that back, so **every page the app cites is
the page in your published 220-page PDF**, not the trimmed one. Verified against
the original's bookmarks: F18 Impairment is indexed at p97 and cited as p102,
which is where it actually sits.

Set `PAGEINDEX_PAGE_OFFSET=0` if you ever index the full untrimmed report
(Standard plan: $30/mo, 10,000 active pages), which removes both the trimming
and the offset.

### Model choice

Two models, set independently in `.env`:

| Variable | Job | Recommendation |
| --- | --- | --- |
| `PAGEINDEX_SEARCH_MODEL` | Reads the outline, picks node ids | **`gpt-4.1-mini`** — the multi-hop selection is the quality bottleneck of the whole system, and it is a short prompt, so this is cheap. `gpt-4o-mini` works but misses secondary sections more often. |
| `PAGEINDEX_ANSWER_MODEL` | Writes the answer from the extracted text | **`gpt-4.1`** — it handles long section text and gets the arithmetic right. Drop to `gpt-4o-mini` if cost matters more than numeric precision. |

Both are overridable live in the Streamlit sidebar, so you can A/B them without
editing `.env`.

### All settings

Every setting lives in `.env` (see `.env.example`). Defaults in brackets.

| Variable | Purpose |
| --- | --- |
| `PAGEINDEX_API_KEY` | PageIndex key. Required. |
| `OPENAI_API_KEY` | OpenAI key. Required. |
| `PAGEINDEX_BASE_URL` | PageIndex API root [`https://api.pageindex.ai`]. |
| `PAGEINDEX_SEARCH_MODEL` | Model that picks node ids [`gpt-4.1-mini`]. |
| `PAGEINDEX_ANSWER_MODEL` | Model that writes the answer [`gpt-4.1`]. |
| `PAGEINDEX_DOC_ID` | Pre-selected document for the app. |
| `PAGEINDEX_DEFAULT_PDF` | PDF the CLI scripts use when none is given. |
| `PAGEINDEX_CACHE_DIR` | Where trees and the `doc_id` registry live [`cache`]. |
| `PAGEINDEX_PAGE_OFFSET` | Added to every cited page so citations match the published report [`0`; **`5` here**]. |
| `PAGEINDEX_OUTLINE_CHAR_BUDGET` | Max outline chars shown to the search model [`120000`]. The full Umicore outline with summaries is ~79k, so all node summaries stay visible; lower it for small-context models. |
| `PAGEINDEX_CONTEXT_CHAR_BUDGET` | Max node text sent to the answer model [`90000`], split evenly across selected nodes. |
| `PAGEINDEX_ALLOW_UPLOAD` | `false` disables indexing in the UI [`true`]. |
| `APP_PASSWORD` | Optional password gate. Unset = no gate. |

## Run

### The app (this is the main deliverable)

Double-click **`run_app.bat`**, or:

```powershell
python -m streamlit run app.py
```

Then open <http://localhost:8501>.

> ### ⚠️ Do NOT use `streamlit run app.py`
>
> This machine has two Python installations, and the bare `streamlit` command
> resolves to the wrong one:
>
> | Command | Python | Streamlit | Works? |
> | --- | --- | --- | --- |
> | `streamlit run app.py` | 3.10 | 1.26.0 | ❌ no `st.write_stream` |
> | `python -m streamlit run app.py` | 3.14 | 1.58.0 | ✅ |
>
> On 1.26 retrieval runs (spending API credits) and then generation dies with
> `AttributeError: module 'streamlit' has no attribute 'write_stream'`.
> `app.py` now checks the version at startup and says so in plain language
> rather than failing cryptically. Minimum required: **Streamlit 1.31**.

Pick a **Section** in the sidebar — the app opens on *Ask*:

1. **Index** — submit a PDF, get the `doc_id`, watch processing until the tree
   is ready. The tree is cached in `cache/<doc_id>.tree.json`, so this is a
   one-time cost per document. Set `PAGEINDEX_ALLOW_UPLOAD=false` to disable it.
2. **Inspect tree** — node/depth/page statistics, an expandable tree with each
   node's summary and raw text, a flat outline view at any depth, and raw JSON.
3. **Ask** — type a question (or pick a built-in multi-section starter) and hit
   **Run**. You get the answer first, then the tree-search reasoning, exactly
   which nodes were selected and why, and the text pulled from each. The
   retrieval trace is downloadable as JSON.

Two Streamlit behaviours the UI works around, both of which made it look broken:
`st.tabs` resets to the first tab on every rerun (so the answer rendered into a
tab you could no longer see) — replaced with a sidebar radio held in
`session_state`. And `st.text_area` only commits its value on Ctrl+Enter (so the
Run button stayed disabled on a first click) — the question and button now live
in an `st.form`, which commits everything on submit.

### Or from the command line

```powershell
python ingest.py                      # submit the default PDF, wait for the tree
python inspect_tree.py --depth 2      # print the outline
python inspect_tree.py --grep battery # find sections by title
python inspect_tree.py --node 0012 --text   # dump one section's text
python ask.py --list-samples
python ask.py --sample 0              # run a multi-hop question end to end
python ask.py "What drove the 2025 impairment?" --show-context
```

## Deployment

**This project runs locally only.** The server binds to `127.0.0.1`, so nothing
on your network or the internet can reach it. That is deliberate: every query
spends your OpenAI credits, and an open URL means strangers spending them.

Cloud deployment was prepared and then abandoned. The scaffolding is still in
the repo and is harmless — it is inert locally — so it is documented here in
case you want it later.

<details>
<summary>Optional: deploying to Streamlit Community Cloud</summary>

`bridge_secrets()` in `app.py` copies `st.secrets` into `os.environ` at startup,
so the same code reads `.env` locally and Cloud secrets when deployed. It checks
for a `secrets.toml` first — touching `st.secrets` when none exists makes
Streamlit render an error into the page.

Steps: push to GitHub → [share.streamlit.io](https://share.streamlit.io) →
**Create app** → repo, branch `main`, main file `app.py` → **Advanced settings →
Secrets**, paste `.streamlit/secrets.toml.example` with real keys → Deploy.

**Community Cloud only deploys _public_ apps from GitHub.** Private apps route to
a paid Snowflake trial. Streamlit's own docs still describe a free private-app
viewer allowlist; that is out of date. A private repo keeps your *code* private,
but the *app* is reachable by anyone with the URL.

Two controls for a shared instance:

| Secret | Effect |
| --- | --- |
| `PAGEINDEX_ALLOW_UPLOAD = "false"` | Disables the Index section's submit and upload actions — protects your PageIndex credits and the 200-page cap. |
| `APP_PASSWORD` | Password gate before the app renders. A deterrent, not real authentication. |

</details>

## How retrieval works

`retrieval.select_nodes()` renders the tree as an indented outline
(`- [node_id] Title (p102, 3.3k chars)` plus summaries) and asks the search model which
nodes are needed. The system prompt is tuned for financial documents: it is told
that questions are often multi-hop, that the *figure* lives in the statements or
notes while the *explanation* lives in the management commentary, and that it
should return both.

Three rules in that prompt were added after watching it fail on real questions:

- **Alternative performance measures.** Asked "what was the revenue in 2025?",
  the first version returned only IFRS turnover (€19.37bn) and never selected the
  segment note holding *revenues excluding metals* (€3.56bn) — Umicore's actual
  steering metric. The two differ by 5x and imply opposite trends (+30% vs
  +2.9%). The prompt now requires selecting the segment note and any APM
  reconciliation alongside the primary statement for any headline-figure
  question, and the answer prompt requires reporting every measure present.
- **Stub nodes.** This tree is partly flat: "F7 Segment information" is a
  66-character empty heading whose real tables are *siblings*, not children.
  The search model picked the empty heading and got nothing. Every outline line
  now carries a size (`6.1k chars`, or `EMPTY-heading only`), so the model can
  see which nodes hold text.
- **Over-selection.** The answer model sees only what search chooses, so a
  missed section is unrecoverable. The prompt now says to prefer over-selecting.

Guards that keep it honest on a 220-page report:

- **Outline budget** — `fit_outline()` tries the full tree with summaries, then
  without, then progressively shallower, until it fits the character budget.
- **Two-pass drill-down** — if the outline had to be truncated, a second pass
  re-expands only the branches the model picked, at full depth, so it can land
  on a specific note rather than "Financial statements".
- **Hallucination filter** — returned ids are checked against the real tree;
  invented ones are dropped and reported, never silently ignored.
- **Even context split** — the character budget is divided across the selected
  nodes so one 40-page section cannot crowd out the note that completes a
  multi-hop answer.

## Files

| File | Role |
| --- | --- |
| `pageindex_client.py` | REST client: submit, poll, fetch tree, local cache + `doc_id` registry |
| `retrieval.py` | Tree walking/stats/outline, reasoning-based node selection, text extraction |
| `generation.py` | Answer prompt + streaming/non-streaming generation |
| `pipeline.py` | Glue: `load_roots` → `retrieve` → `ask`, plus sample multi-hop queries |
| `ingest.py` / `inspect_tree.py` / `ask.py` | CLI for each stage |
| `corrections.py` | Verified repairs to extraction errors in node text (see below) |
| `check_setup.py` | Validates `.env` and live-tests both API keys |
| `trim_pdf.py` | Builds a 200-page copy to fit the Free Trial page cap |
| `app.py` | Streamlit UI (sidebar sections, form-based Ask, version guard) |
| `run_app.bat` | Launches the app with the correct Python — use this |
| `cache/` | `documents.json` (filename → doc_id) and `<doc_id>.tree.json` |

## Extraction corrections

PageIndex's parse is good but not perfect, and in a financial document one
flipped word can invert a metric's meaning. In this report the Group key figures
table (`0010`) came back as **"Revenues (including metal)"** where the source PDF
says **"excluding metal"** — the opposite. The answer model faithfully repeated
the wrong label.

`corrections.py` holds a small, explicit list of repairs applied to node text
before it reaches the model. Every entry records why it is wrong and what the
original PDF actually says, verified by reading the source. Corrections are
logged rather than silent, and `retrieval.raw_node_text()` always returns the
untouched original so you can audit what changed:

```powershell
python -c "import corrections,retrieval;from pageindex_client import PageIndexClient as C;print(corrections.audit(retrieval.normalise_tree(C().get_or_fetch_tree('YOUR_DOC_ID'))))"
```

For this document exactly one correction fires, on one node. The other 10
occurrences of that label across the report extracted correctly — it is an
isolated defect, not a systemic one. **Rule for adding entries: verify against
the source PDF first, and never use this to reword the document.**

## Notes

- PageIndex processing of a 220-page PDF takes several minutes. The `doc_id`
  stays valid — if you close the app mid-processing, reopen it and hit
  *Check / wait for the tree*.
- Trees are cached locally, so tree inspection and repeat questions cost no
  PageIndex calls; each question costs exactly two OpenAI calls (search +
  answer).
- The answer prompt forbids answering from outside the retrieved sections. If a
  question can't be answered, the model is told to say which section would
  likely hold it — that gap is usually a retrieval miss you can fix by raising
  *Max nodes to retrieve*.
