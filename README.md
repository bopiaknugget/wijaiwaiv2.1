# 🔬 Research Workbench — AI-Powered RAG with Research Editor

ระบบค้นคว้าวิจัยอัจฉริยะที่รวม RAG (Retrieval-Augmented Generation) เข้ากับ editor สำหรับเขียนงานวิจัย รองรับภาษาไทยและอังกฤษ ใช้ OpenThaiGPT เป็น LLM หลัก

## 🎯 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                         │
│                                                                      │
│  ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐  │
│  │   Sidebar    │  │  🔬 Research Workbench │  │  🤖 Assistant     │  │
│  │  📄 Documents│  │   Title + Editor      │  │   Chat + RAG Q&A │  │
│  │  📝 Notes    │  │   Save / Load / Export│  │   Research Mode  │  │
│  └──────┬───────┘  └───────────┬───────────┘  └────────┬──────────┘  │
│         │                      │                       │              │
└─────────┼──────────────────────┼───────────────────────┼──────────────┘
          │                      │                       │
  ┌───────▼──────────┐   ┌──────▼───────┐   ┌──────────▼──────────┐
  │ document_loader  │   │  your_work/  │   │    generator.py     │
  │ PDF/TXT/DOCX     │   │  (local fs)  │   │  OpenThaiGPT API    │
  │ Parent-Child     │   └──────────────┘   │  Query re-phrasing  │
  │ Chunking         │                      └──────────┬──────────┘
  └───────┬──────────┘                                 │
          │                                            │
  ┌───────▼──────────────────────────────────┐         │
  │           vector_store.py                │         │
  │  HuggingFace Embeddings (local, no key)  │◄────────┘
  │  Unified ChromaDB Collection             │
  │  MMR + Parent-Child Retrieval            │
  └───────┬──────────────────────────────────┘
          │
  ┌───────▼──────────┐    ┌──────────────────┐
  │  ChromaDB        │    │  database.py     │
  │  (unified_       │    │  SQLite          │
  │   chroma_db/)    │    │  Notes + Parents │
  └──────────────────┘    └──────────────────┘
```

## 📋 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web UI** | Streamlit |
| **LLM** | OpenThaiGPT API (รองรับภาษาไทย) |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (local, no API key) |
| **Vector DB** | ChromaDB (local, unified collection) |
| **Relational DB** | SQLite (`research_notes.db`) |
| **Document Parsing** | PyPDF, docx2txt |
| **Framework** | LangChain Core + Community |

## 📦 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with your OpenThaiGPT API key
echo "OPENTHAI_API_KEY=your_key_here" > .env
```

## 🚀 Quick Start

### Web UI (Primary)

```bash
streamlit run app.py
```

### CLI

```bash
# Ingest a document
python main.py --ingest path/to/document.pdf

# Query the vector store
python main.py --query "your question here"

# With custom DB path and top-k
python main.py --ingest doc.pdf --db ./my_db
python main.py --query "question" --db ./my_db --k 5
```

## 🏗️ Module Breakdown

### `app.py` — Streamlit UI
3-panel layout:
- **Sidebar** — Document upload (PDF/TXT/DOCX/DOC) + Research Notes (SQLite-backed)
- **Research Workbench** (center) — Title + text editor with Save/Load/Export/Import, Undo/Redo
- **Assistant** (right) — Chat Q&A with RAG retrieval, Research Mode for detailed answers

### `generator.py` — LLM Integration
- OpenThaiGPT API calls (`POST http://thaillm.or.th/api/openthaigpt/v1/chat/completions`)
- Chat-history-based query re-phrasing
- Agentic editor manipulation (selection edit)
- `<think>` tag stripping from chat history context

### `vector_store.py` — Embeddings & Retrieval
- Local HuggingFace embeddings (~400MB, supports Thai, no API key required)
- Single unified ChromaDB collection separated by `source_type` metadata
- MMR-based retrieval with parent-child expansion
- `@st.cache_resource` on embeddings to avoid repeated model loads

### `document_loader.py` — Document Processing
- PDF, TXT, DOCX, DOC loading with rich metadata extraction
- Parent-Child Chunking: small children for precise search, large parents for context
- Summary embedding for broad semantic matching
- Adaptive chunk sizing based on content length (<2k→300, 2k-10k→800, >10k→1200 chars)

### `database.py` — SQLite Storage
- CRUD for research notes
- Document metadata storage
- Parent chunk storage for parent-child retrieval
- Database: `research_notes.db`

### `main.py` — CLI Entry Point
- `--ingest` mode: Load document → Chunk → Embed → Store
- `--query` mode: Load store → MMR search → Retrieve

### `rag_pipeline.py`
**Legacy Phase 1 reference only — do not use or extend.**

## 📁 Project Structure

```
wijaiwaiv2.1/
├── app.py                  # Streamlit Web UI (3-panel layout)
├── generator.py            # OpenThaiGPT API integration
├── vector_store.py         # HuggingFace embeddings + ChromaDB
├── document_loader.py      # PDF/TXT/DOCX loading & chunking
├── database.py             # SQLite CRUD (notes, parents, metadata)
├── main.py                 # CLI entry point
├── rag_pipeline.py         # Legacy Phase 1 (unused)
├── requirements.txt        # Python dependencies
├── .env                    # OPENTHAI_API_KEY (not committed)
├── .gitignore
├── CLAUDE.md               # Claude Code guidance
├── README.md               # This file
├── research_notes.db       # SQLite database (auto-created)
├── unified_chroma_db/      # ChromaDB vector store (auto-created)
├── your_work/              # Saved research files from editor
├── data/                   # Sample/test data
└── md/                     # Design documents
```

## 🔬 Key Features

### Advanced RAG Pipeline
- **Rich Metadata** — Each chunk carries `paper_title`, `authors`, `year`, `section`, `source_type`, `created_at`
- **Parent-Child Chunking** — Small child chunks in ChromaDB for precise vector search; parent content (full pages) in SQLite for richer LLM context
- **Summary Embedding** — Extractive summaries embedded alongside chunks for broad semantic matching
- **MMR Retrieval** — Maximal Marginal Relevance for diverse, high-quality results

### Research Workbench Editor
- Title + content text editor with auto-save
- File operations: Save, Save As, Load, Export, Import
- Edit operations: Undo, Redo, Clear
- Research Mode: AI writes detailed answers directly into the editor

### Assistant Chat
- RAG-powered Q&A with context from uploaded documents and notes
- `<think>` tag parsing — reasoning shown in collapsible expander
- Research Mode toggle for comprehensive answers
- Token usage tracking with cost estimation (THB)

### Research Notes
- SQLite-backed notes with full CRUD
- Notes are embedded into the unified vector store (`source_type='note'`)
- Searchable alongside uploaded documents

## ⚙️ Environment Setup

Create `.env` in the project root:
```
OPENTHAI_API_KEY=your_key_here
```

The API key is loaded by `generator.py` using `python-dotenv`. No quotes or spaces around the value.

## 📊 Cost

| Component | Cost |
|-----------|------|
| Embeddings (HuggingFace local) | Free |
| ChromaDB (local) | Free |
| SQLite (local) | Free |
| OpenThaiGPT API | Per-token usage |

Token cost is displayed in the UI in THB ($0.4/1M tokens, 1 USD = 35 THB).

## 📄 License

Provided as-is for educational and research purposes.
