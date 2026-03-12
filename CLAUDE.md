# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

Create `wijaiwaiv2/.env` with:
```
OPENTHAI_API_KEY=your_key_here
```

The API key is loaded by `generator.py` using `python-dotenv`. No quotes or spaces around the value.

## Architecture

Two entry points exist for the same underlying pipeline:

- **`app.py`** — Streamlit web UI with three tabs: Assistant (chat), Research Notes (SQLite-backed), Usage (token/cost stats). State is managed with `st.session_state`. Note input fields use the "Value Proxy pattern" (separate `*_val` keys + `st.rerun()`) to avoid Streamlit widget key conflicts on save.

- **`main.py`** — CLI entry point with `--ingest` / `--query` modes.

### Data flow (query path)

```
User question
  → retrieve_similar_documents() [vector_store.py, ChromaDB similarity search]
  → generate_answer(query, retrieved_docs, chat_history) [generator.py]
      → optional: re-phrase query via API if chat_history exists
      → _call_api() → POST to OpenThaiGPT API
  → parse_think_content(answer) [app.py]
      → splits <think>...</think> from the answer text
      → think shown in collapsible expander; answer shown normally
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, `<think>` tag parsing/display |
| `generator.py` | OpenThaiGPT API calls, chat-history-based query re-phrasing |
| `vector_store.py` | HuggingFace embeddings (local, no API key), ChromaDB lifecycle |
| `document_loader.py` | PDF loading via PyPDFLoader, text chunking |
| `database.py` | SQLite CRUD for research notes (`research_notes.db`) |
| `main.py` | CLI wrapper around the same pipeline modules |

### Key implementation details

- **Embeddings** use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local HuggingFace, ~400MB, supports Thai). Cached with `@st.cache_resource`.
- **ChromaDB** auto-persists to `./temp_chroma_db` (PDFs) or `./notes_chroma_db` (notes). `get_or_create_vector_store()` loads from disk if path exists, creates new only if documents are provided — avoids re-embedding costs.
- **`<think>` tag handling**: `parse_think_content()` in `app.py` uses regex to extract `<think>...</think>` blocks. `generator.py` strips these tags from chat history before sending to the API for context re-phrasing.
- **Token cost** is calculated at `$0.4 / 1M tokens`, displayed in THB (1 USD = 35 THB). The Usage tab shows cumulative session totals.
- **Research notes** are both saved to SQLite and embedded into the vector store, making them searchable alongside PDF content.
