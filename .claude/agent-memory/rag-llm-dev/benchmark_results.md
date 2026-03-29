---
name: End-to-End Benchmark Results (2026-03-29 — real data)
description: Live benchmark results against OpenThaiGPT API and Pinecone — pass rates, latency, real-data retrieval quality, confirmed design patterns
type: project
---

Second end-to-end benchmark run on 2026-03-29. All 9 tests passed (9/9), now including two
real-data tests (H and I) that exercise actual ingestion and retrieval against Pinecone.

**Why:** The first run used an empty TEST_USER_ID namespace, so retrieval always returned 0
docs. Tests H and I close that gap by (a) ingesting a controlled benchmark document and
verifying every RAG sub-feature, and (b) probing the real user namespace.

**How to apply:** Use these numbers as regression targets. Re-run benchmark.py after any
vector_store.py or generator.py change.

## Pinecone index state (as of 2026-03-29)

| Namespace | Vectors | Content |
|---|---|---|
| `default` | 2509 | Legacy vectors from old schema (no `content` key, garbled Thai — unusable by current pipeline) |
| `google_102886497280770017710` | 6 | test_doc.pdf (new schema: `content` + `doc_name` keys, correct) |
| `__benchmark_test_user__` | 0 | Benchmark-only; cleaned up after each Test H run |

The `default` namespace was ingested by a prior version of the app that stored content in a
`text` key (not `content`) and omitted `doc_name`. The current pipeline reads `content` from
Pinecone metadata, so those 2509 legacy vectors are effectively invisible to `retrieve_unified()`.

## Latency baseline (with real Pinecone retrieval)

| Phase | Typical |
|---|---|
| Embedding (cold, first call) | ~1300–1400ms |
| Embedding (warm, cached) | ~370–500ms |
| Retrieval (Pinecone query, real docs) | ~275–2500ms |
| LLM (OpenThaiGPT, short) | ~5600–11700ms |
| LLM (OpenThaiGPT, long) | ~10000–16000ms |
| Streaming first token | ~137–155ms |
| End-to-end (typical chat) | ~9000–15000ms |
| Test H full ingest+retrieve cycle | ~18800ms |

## Real data retrieval results (Test H)

Ingested a 5-section English/Thai RAG knowledge document into the benchmark namespace:
- 5 child chunks created (adaptive chunk_size=800, 1 parent page)
- All 3 content-relevant queries returned hits (precision 3/3)
- Parent-child expansion fired correctly: 3/3 retrieved docs had `chunk_type=parent_expanded`
- Context length control: 3022 chars returned, well under the 8000 char cap
- Hybrid BM25 and vector-only modes both returned results without error
- Embedding cache populated: 18 entries after the full test suite

## Live namespace results (Test I)

Queried `google_102886497280770017710` (6 vectors from test_doc.pdf):
- Precision 4/4 — all content-relevant queries returned keyword hits
- Score distribution: min=0.761, max=0.872, avg=0.811 — all above the 0.30 threshold
- Parent-child expansion: 1/1 sample doc was `parent_expanded` (SQLite lookup working)

## Confirmed design patterns

- `generate_answer()` intentionally returns `<think>...</think>` tags EMBEDDED in the response
  string. `app.py`'s `parse_think_content()` splits them out for the UI.
- `retrieve_unified()` returns `[]` for empty/whitespace queries (guard at top of function).
- `is_small_talk()` ignores queries >80 chars — long research queries always route to retrieval.
- Streaming (`generate_answer_stream`) falls back gracefully when the API returns non-SSE content.
- Windows console cp874 encoding: `document_loader.py` prints `✓` (U+2713) which crashes on the
  Thai Windows codepage. Fixed in benchmark.py by forcing UTF-8 on sys.stdout/stderr at startup.

## Bugs found in this run

1. **Windows cp874 UnicodeEncodeError** — `document_loader.py` uses `✓` in print statements;
   the Thai Windows console (cp874) cannot encode it. Fix: `benchmark.py` now forces
   `sys.stdout = io.TextIOWrapper(..., encoding='utf-8')` before any test runs.
   This is a display-only issue (the app runs under Streamlit which handles encoding correctly).

## Benchmark script location

`C:\Users\USER\Desktop\old workspace\wijaiwaiv2.1\benchmark.py`

Added tests H (ingest+retrieve real doc) and I (live namespace probe).
Test H self-cleans: vectors are deleted from Pinecone after the test.
