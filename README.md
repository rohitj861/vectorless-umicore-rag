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

**Page numbers shift.** Content after original p6 sits 6 pages earlier in the
trimmed file, so a citation reading "p. 56" corresponds to p. 62 of the
published report. Raising the cap (Standard plan: $30/mo, 10,000 active pages)
removes both the trimming and the offset.

### Model choice

Two models, set independently in `.env`:

| Variable | Job | Recommendation |
| --- | --- | --- |
| `PAGEINDEX_SEARCH_MODEL` | Reads the outline, picks node ids | **`gpt-4.1-mini`** — the multi-hop selection is the quality bottleneck of the whole system, and it is a short prompt, so this is cheap. `gpt-4o-mini` works but misses secondary sections more often. |
| `PAGEINDEX_ANSWER_MODEL` | Writes the answer from the extracted text | **`gpt-4.1`** — it handles long section text and gets the arithmetic right. Drop to `gpt-4o-mini` if cost matters more than numeric precision. |

Both are overridable live in the Streamlit sidebar, so you can A/B them without
editing `.env`.

## Run

### The app (this is the main deliverable)

```powershell
streamlit run app.py
```

Three tabs, matching the three stages:

1. **Index** — submit `Umicore Annual Report 2025.pdf`, get the `doc_id`, watch
   the processing status until the tree is ready. The tree is cached in
   `cache/<doc_id>.tree.json`, so this is a one-time cost per document.
2. **Inspect tree** — node/depth/page statistics, an expandable tree with each
   node's summary and raw text, a flat outline view at any depth, and the raw
   JSON.
3. **Ask** — type a question (or pick one of the built-in multi-section
   starters). You see the tree-search reasoning, exactly which nodes were
   selected and why, the text pulled from each, then the streamed answer. The
   retrieval trace is downloadable as JSON.

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

## Deploying to Streamlit Community Cloud

The app reads config from `.env` locally and from `st.secrets` on Cloud —
`bridge_secrets()` in `app.py` copies Cloud secrets into `os.environ` at
startup, so every module keeps using plain `os.getenv` and nothing else changes.

What ships in the repo: the code, and `cache/<doc_id>.tree.json` (~1.1 MB) so the
deployed app queries the already-indexed document without re-parsing anything.
What never ships: `.env`, `.streamlit/secrets.toml`, and the PDFs (unused at
runtime — the tree is enough).

1. Push the repo to GitHub (private repo → private app by default).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** →
   pick the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste the contents of
   `.streamlit/secrets.toml.example` with your real keys filled in.
4. Deploy. First boot installs `requirements.txt` and takes a couple of minutes.
5. **Set the viewer allowlist**: app → **Share** → "Only specific people can view
   this app" → add the email addresses that should have access.

### Cost and exposure on a shared deployment

Every visitor who can open the app spends *your* OpenAI credits, and every
upload spends *your* PageIndex credits and page allowance. Two controls:

| Secret | Effect |
| --- | --- |
| `PAGEINDEX_ALLOW_UPLOAD = "false"` | Disables the Index tab's submit and upload actions. Set this on any shared instance — it protects the 200-page cap. |
| `APP_PASSWORD` | Optional shared-password gate before the app renders. A deterrent, not real auth; prefer the viewer allowlist. |

Streamlit Community Cloud allows **one private app at a time**. If you already
have one, either make it public or delete it before deploying this.

## How retrieval works

`retrieval.select_nodes()` renders the tree as an indented outline
(`- [node_id] Title (p30-36)` plus summaries) and asks the search model which
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
| `app.py` | Streamlit UI |
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
