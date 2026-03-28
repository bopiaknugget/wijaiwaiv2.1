# CLAUDE.md

## Rules
    - always use claude --dangerously-skip-permissions to skip dangerously permissions requests.
## Remark 
    - User prefer to use Thai language to command and chat with claude.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

**Research Workbench** — an AI-powered research platform combining RAG (Retrieval-Augmented Generation) with a specialized text editor for academic research. Supports bilingual Thai/English content. Uses OpenThaiGPT as the primary LLM and local HuggingFace embeddings.

## Commands

```bash
# Run the Streamlit web UI (primary interface)
streamlit run app.py

# CLI: ingest a PDF into the vector store
python main.py --ingest path/to/document.pdf

# CLI: query the vector store and generate an answer
python main.py --query "your question here"

# CLI with custom DB path and top-k
python main.py --ingest doc.pdf --db ./my_db
python main.py --query "question" --db ./my_db --k 5

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Create `.env` in the project root with:
```
OPENTHAI_API_KEY=your_key_here
```

The API key is loaded by `generator.py`, `reviewer.py`, and `web_scraper.py` using `python-dotenv`. No quotes or spaces around the value.

## Architecture

Two entry points exist for the same underlying pipeline:

- **`app.py`** — Streamlit web UI with three panels: Sidebar (Docs + Notes + Web Pages), Research Workbench (center editor, saves files to `./user_data/`), and Assistant (right chat panel with Q&A, Research Mode, and Advisor Review). State is managed with `st.session_state`. Note input fields use the "Value Proxy pattern" (separate `*_val` keys + `st.rerun()`) to avoid Streamlit widget key conflicts on save.

- **`main.py`** — CLI entry point with `--ingest` / `--query` modes.

### Data flow (query path)

In **app.py** (web UI):
```
User question
  → Intent classification: "chat" | "edit" | "research"
  → retrieve_unified() [vector_store.py]
      → MMR search on unified collection (metadata-filtered by source_type)
      → Parent-child expansion: child match → fetch parent content from SQLite
      → Deduplication by content
  → generate_answer(query, retrieved_docs, chat_history) [generator.py]
      → optional: re-phrase query via API if chat_history exists
      → _call_api() → POST to OpenThaiGPT API
  → parse_think_content(answer) [app.py]
      → splits <think>...</think> from the answer text
      → think shown in collapsible expander; answer shown normally

  If intent == "edit":
      → AI writes/modifies editor content directly
  If intent == "research":
      → Deep research with structured output sent to editor (higher token budget: 12,000)
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, `<think>` tag parsing/display, Research Workbench editor, 3-panel layout |
| `generator.py` | OpenThaiGPT API calls, chat-history-based query re-phrasing, agentic intent classification (chat/edit/research), editor manipulation |
| `vector_store.py` | HuggingFace embeddings (local, no API key), unified ChromaDB collection, parent-child retrieval, MMR search |
| `document_loader.py` | PDF/TXT/DOCX loading, rich metadata extraction, parent-child chunking, adaptive chunk sizing, summary embedding |
| `database.py` | SQLite CRUD for research notes, document metadata, parent chunks, and web page metadata (`./Database/research_notes.db`) |
| `reviewer.py` | Research advisor review system — strict thesis advisor persona, reviews 5-chapter thesis structure, returns color-coded feedback |
| `web_scraper.py` | Web content extraction (trafilatura + BeautifulSoup fallback), AI-generated summaries and titles, content chunking into ChromaDB |
| `main.py` | CLI wrapper around the same pipeline modules |
| `rag_pipeline.py` | **Legacy Phase 1 reference only — do not use or extend** |

### Database schema

**SQLite** (`./Database/research_notes.db`) has four tables:

| Table | Purpose | Key Columns |
|---|---|---|
| `research_notes` | User's research notes | id, title, content, timestamp |
| `documents` | Uploaded document metadata | id, filename, file_type, chunk_count, db_path, timestamp |
| `parent_chunks` | Parent content for parent-child RAG | id (TEXT PK), content, source_file, page_number, section, timestamp |
| `web_pages` | Scraped web page metadata | id, url, title, summary, chunk_count, timestamp |

**ChromaDB** (`./Database/unified_chroma_db/`) — single `unified` collection. Content types separated by `source_type` metadata: `"document"`, `"note"`, `"web"`.

### Key implementation details

- **Embeddings** use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local HuggingFace, ~400MB, supports Thai). Cached with `@st.cache_resource`.
- **ChromaDB** uses a single unified collection for documents, notes, and web content, separated by `source_type` metadata. `get_or_create_vector_store()` loads from disk if path exists, creates new only if documents are provided.
- **Advanced RAG** techniques:
  - **Rich Metadata**: Each chunk carries `paper_title`, `authors`, `year`, `section`, `source_type`, `created_at`.
  - **Parent-Child Chunking**: Small child chunks are stored in ChromaDB for precise vector search; parent content (full pages) is stored in SQLite. On retrieval, child matches are expanded to parent content for richer LLM context.
  - **Summary Embedding**: Extractive summaries of each page are also embedded in ChromaDB for broad semantic matching.
  - **Adaptive Chunk Sizing**: Chunk size auto-adjusts based on total content length (<2k→300, 2k-10k→800, >10k→1200 chars).
- **MMR Retrieval**: `lambda_mult=0.6` (60% relevance, 40% diversity), `fetch_k=k*4`.
- **`<think>` tag handling**: `parse_think_content()` in `app.py` uses regex to extract `<think>...</think>` blocks. `generator.py` strips these tags from chat history before sending to the API for context re-phrasing.
- **Token cost** is calculated at `$0.4 / 1M tokens`, displayed in THB (1 USD = 35 THB). Tracked in `st.session_state.total_tokens` and `total_cost_thb`.
- **Research notes** are saved to SQLite and embedded into the unified vector store with `source_type='note'`, making them searchable alongside document content.
- **Web content** is scraped, summarized by AI, chunked, and embedded with `source_type='web'`. Metadata stored in SQLite `web_pages` table.
- **Advisor review** uses a strict thesis advisor persona (20+ years expertise) that reviews chapter structure and returns color-coded feedback: 🔴 [ต้องแก้ไข] (must fix), 🟢 [ดีแล้ว] (well done), 🟡 [คำแนะนำ] (recommendations).

### API details

All LLM calls go to OpenThaiGPT:
- **Endpoint**: `POST http://thaillm.or.th/api/openthaigpt/v1/chat/completions`
- **Auth header**: `apikey: {OPENTHAI_API_KEY}`
- **Model**: `/model`
- **Default max_tokens**: 2048 (chat), 3000 (normal), 12000 (research mode)
- **Temperature**: 0.3

### Important paths

| Path | Purpose |
|---|---|
| `./Database/research_notes.db` | SQLite database (auto-created) |
| `./Database/unified_chroma_db/` | ChromaDB persistent storage (auto-created) |
| `./user_data/` | Saved research documents from editor (auto-created) |
| `./md/` | Design documents and task notes |
| `./.env` | API key (git-ignored) |

### Common pitfalls

- **Do not** modify `rag_pipeline.py` — it is legacy code kept for reference only.
- **Streamlit widget keys**: When adding new input fields, use unique keys and the Value Proxy pattern if the field value needs to be programmatically reset.
- **ChromaDB collection**: There is only ONE collection (`unified`). Do not create separate collections — use `source_type` metadata to filter.
- **Embedding model**: The first run downloads ~400MB. Subsequent runs use the cached model.
- **Database directory**: All persistent data lives under `./Database/`. The directory and tables are auto-created on first use.
- **Chat history**: Capped at last 6 messages in `st.session_state` to control context length.
