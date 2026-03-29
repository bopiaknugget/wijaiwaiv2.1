---
name: Performance Improvements (fix_emedding.md)
description: All 12 latency/quality improvements from fix_emedding.md — what was implemented, where, and design decisions
type: project
---

Implemented 2026-03-29 based on fix_emedding.md. Changes across vector_store.py, generator.py, app.py.

## vector_store.py changes

1. **Embedding cache** — `_EMBEDDING_CACHE` dict keyed by SHA-256 hash. Max 4096 entries, FIFO eviction. Query embeddings prefixed with "q:" to avoid collision with passage cache. Covers both `_embed_texts` and `_embed_query`.

2. **Parallel embedding** — `_embed_texts` uses `ThreadPoolExecutor(max_workers=4)` to embed multiple batches of 96 concurrently. Single-batch case skips thread overhead. Only uncached texts hit the API.

3. **Metadata filtering** — `retrieve_unified` now accepts `doc_name` parameter in addition to `source_type`. Multiple filters are stacked with `{"$and": [...]}` for Pinecone.

4. **Re-rank cutoff** — `_PRE_FILTER_MULTIPLIER=6`, `_MAX_PRE_FILTER=30`. Fetches `min(k*6, 30)` candidates from Pinecone, then hybrid re-ranks to final k.

5. **Hybrid search (BM25)** — `_bm25_scores()` pure-Python BM25. Fused with vector scores: `0.7 * vector + 0.3 * bm25`. Results re-sorted by fused score. Controlled by `hybrid=True` param (default on).

6. **Timeout + fallback** — Pinecone query wrapped in `ThreadPoolExecutor(max_workers=1)`, `future.result(timeout=10)`. Returns `[]` on timeout. `_PINECONE_QUERY_TIMEOUT=10` seconds.

7. **Result deduplication** — Improved from first-200-chars to normalised (lowercase + collapsed whitespace) first-300-chars fingerprint. Catches formatting variants of same passage.

8. **Context length control** — Budget-based truncation. `_MAX_CONTEXT_CHARS=8000` (~2000 tokens). Each doc consumed from budget; last doc truncated to remaining budget.

9. **Score threshold** — `_MIN_SCORE_THRESHOLD=0.30` filters out low-relevance Pinecone matches before hybrid scoring.

10. **Cache utilities** — `get_embedding_cache_stats()` and `clear_embedding_cache()` added.

## generator.py changes

11. **Query routing** — `is_small_talk(query)` regex-based detector. Matches Thai greetings, meta-questions, closings. Returns True only for short queries (<80 chars) matching patterns. Exported for use by app.py.

12. **Streaming** — `_call_api_stream()` uses `requests` with `stream=True`, parses SSE `data: {...}` lines. Falls back to yielding full response if server doesn't return `text/event-stream`. `generate_answer_stream()` is the public API for plain chat mode streaming.

## app.py changes

- Import `generate_answer_stream`, `is_small_talk` from generator
- Query routing: if `not is_research and is_small_talk(actual_query)` → skip `retrieve_unified`
- Streaming path: plain chat uses `st.write_stream(generate_answer_stream(...))` for low perceived latency
- Research mode still uses blocking `generate_answer()` (structured JSON not streamable)
- Token count for streamed responses approximated as `len(text) // 4`

## What was NOT implemented

- Async end-to-end (item 10): Streamlit is synchronous; async would require significant architecture change
- Connection reuse (item 11): Already handled by `@st.cache_resource` on Pinecone client and index

**Why:** Streaming required keeping research mode blocking because JSON must be complete before parsing. The `st.write_stream()` call renders the chat message directly so it does NOT go through the existing `display_assistant_message()` path — this is intentional to avoid double-rendering.
