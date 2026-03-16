# Research Workbench — AI-Powered RAG with Research Editor

ระบบค้นคว้าวิจัยอัจฉริยะที่รวม RAG (Retrieval-Augmented Generation) เข้ากับ editor สำหรับเขียนงานวิจัย รองรับภาษาไทยและอังกฤษ ใช้ OpenThaiGPT เป็น LLM หลัก

An intelligent research platform combining RAG with a specialized text editor for academic research. Supports bilingual Thai/English content using OpenThaiGPT as the primary LLM.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                         │
│                                                                      │
│  ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐  │
│  │   Sidebar    │  │  Research Workbench    │  │  Assistant        │  │
│  │  Documents   │  │   Title + Editor      │  │   Chat + RAG Q&A │  │
│  │  Notes       │  │   Save / Load / Export│  │   Research Mode  │  │
│  │  Web Pages   │  │                       │  │   Advisor Review │  │
│  └──────┬───────┘  └───────────┬───────────┘  └────────┬──────────┘  │
│         │                      │                       │              │
└─────────┼──────────────────────┼───────────────────────┼──────────────┘
          │                      │                       │
  ┌───────▼──────────┐   ┌──────▼───────┐   ┌──────────▼──────────┐
  │ document_loader  │   │  user_data/  │   │    generator.py     │
  │ PDF/TXT/DOCX     │   │  (local fs)  │   │  OpenThaiGPT API    │
  │ Parent-Child     │   └──────────────┘   │  Query re-phrasing  │
  │ Chunking         │                      │  Intent classifier  │
  └───────┬──────────┘                      └──────────┬──────────┘
          │                                            │
  ┌───────▼──────────────────────────────────┐         │
  │           vector_store.py                │         │
  │  HuggingFace Embeddings (local, no key)  │◄────────┘
  │  Unified ChromaDB Collection             │
  │  MMR + Parent-Child Retrieval            │
  └───────┬──────────────────────────────────┘
          │
  ┌───────▼──────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  ChromaDB        │    │  database.py     │    │  reviewer.py     │
  │  (Database/      │    │  SQLite          │    │  Thesis Advisor  │
  │   unified_       │    │  Notes + Parents │    │  Review System   │
  │   chroma_db/)    │    │  + Web Pages     │    │                  │
  └──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web UI** | Streamlit (3-panel layout) |
| **LLM** | OpenThaiGPT API (supports Thai natively) |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (local, no API key, ~400MB) |
| **Vector DB** | ChromaDB (local, single unified collection) |
| **Relational DB** | SQLite (`Database/research_notes.db`) |
| **Document Parsing** | PyPDF, docx2txt |
| **Web Scraping** | Trafilatura + BeautifulSoup4 (fallback) |
| **Framework** | LangChain Core + Community |

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your OpenThaiGPT API key
echo "OPENTHAI_API_KEY=your_key_here" > .env
```

> **Note**: The `.env` file must contain just `OPENTHAI_API_KEY=your_key_here` — no quotes, no spaces around the `=`.

> **Note**: First run will download the embedding model (~400MB). Subsequent runs use the cached version.

## Quick Start

### Web UI (Primary)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with three panels:

| Panel | Features |
|-------|----------|
| **Sidebar** (left) | Upload documents (PDF/TXT/DOCX/DOC), manage research notes, manage web pages |
| **Research Workbench** (center) | Text editor with title, Save/Load/Export/Import, Undo/Redo |
| **Assistant** (right) | Chat Q&A with RAG, Research Mode, Advisor Review |

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

## Features

### RAG-Powered Q&A
Ask questions about your uploaded documents. The system retrieves relevant context using MMR (Maximal Marginal Relevance) search and generates answers with OpenThaiGPT. Supports follow-up questions with automatic query re-phrasing based on chat history.

### Research Mode
Toggle Research Mode for comprehensive, structured answers. The AI writes detailed research output (introduction, analysis, findings, summary, references) directly into the editor with a higher token budget (12,000 tokens).

### Agentic Editor
The assistant classifies user intent as:
- **Chat** — answers in the chat panel
- **Edit** — writes or modifies content in the editor
- **Research** — deep research with structured output to editor

Selection-based editing: highlight text in the editor and ask the AI to modify just that selection.

### Research Notes
Create and manage notes that are embedded into the same vector store as documents. Notes become searchable alongside uploaded PDFs — great for annotations, summaries, and cross-referencing.

### Web Content Integration
Paste a URL to scrape and integrate web content:
- Automatic content extraction (trafilatura with BeautifulSoup fallback)
- AI-generated summaries and titles
- Content is chunked and embedded into the unified vector store
- Metadata stored in SQLite for management

### Advisor Review
Click the advisor button to get your research reviewed by a strict thesis advisor persona (simulating 20+ years of expertise). Reviews cover the standard 5-chapter thesis structure:
- Chapter 1: Introduction, objectives, hypotheses, scope
- Chapter 2: Literature review, theoretical framework
- Chapter 3: Research methodology
- Chapter 4: Results presentation
- Chapter 5: Summary, discussion, recommendations

Feedback is color-coded:
- **Red** [ต้องแก้ไข] — Must fix
- **Green** [ดีแล้ว] — Well done
- **Yellow** [คำแนะนำ] — Recommendations

### Advanced RAG Pipeline
- **Rich Metadata** — Each chunk carries `paper_title`, `authors`, `year`, `section`, `source_type`, `created_at`
- **Parent-Child Chunking** — Small child chunks in ChromaDB for precise vector search; parent content (full pages) in SQLite for richer LLM context
- **Summary Embedding** — Extractive summaries embedded alongside chunks for broad semantic matching
- **Adaptive Chunk Sizing** — Auto-adjusts based on content length (<2k→300, 2k-10k→800, >10k→1200 chars)
- **MMR Retrieval** — Maximal Marginal Relevance (lambda=0.6) for diverse, high-quality results

## Module Breakdown

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, `<think>` tag parsing, Research Workbench editor, 3-panel layout |
| `generator.py` | OpenThaiGPT API calls, query re-phrasing, intent classification (chat/edit/research), editor manipulation |
| `vector_store.py` | HuggingFace embeddings (local), unified ChromaDB collection, parent-child retrieval, MMR search |
| `document_loader.py` | PDF/TXT/DOCX loading, metadata extraction, parent-child chunking, adaptive chunk sizing, summary embedding |
| `database.py` | SQLite CRUD for notes, documents, parent chunks, and web pages |
| `reviewer.py` | Research advisor review system with color-coded thesis feedback |
| `web_scraper.py` | Web content extraction, AI summarization, content chunking into ChromaDB |
| `main.py` | CLI entry point (`--ingest` / `--query` modes) |
| `rag_pipeline.py` | Legacy Phase 1 (unused — do not modify) |

## Project Structure

```
wijaiwaiv2.1/
├── app.py                  # Streamlit Web UI (3-panel layout)
├── generator.py            # OpenThaiGPT API integration + intent classifier
├── vector_store.py         # HuggingFace embeddings + ChromaDB
├── document_loader.py      # PDF/TXT/DOCX loading & chunking
├── database.py             # SQLite CRUD (notes, parents, documents, web pages)
├── reviewer.py             # Thesis advisor review system
├── web_scraper.py          # Web content scraping & summarization
├── main.py                 # CLI entry point
├── rag_pipeline.py         # Legacy Phase 1 (unused)
├── requirements.txt        # Python dependencies
├── .env                    # OPENTHAI_API_KEY (not committed)
├── .gitignore
├── CLAUDE.md               # Claude Code guidance
├── README.md               # This file
├── Database/               # Auto-created on first run
│   ├── research_notes.db   # SQLite database
│   └── unified_chroma_db/  # ChromaDB vector store
├── user_data/              # Saved research files from editor
└── md/                     # Design documents & task notes
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENTHAI_API_KEY` | Yes | OpenThaiGPT API key for LLM calls |

## Cost

| Component | Cost |
|-----------|------|
| Embeddings (HuggingFace local) | Free |
| ChromaDB (local) | Free |
| SQLite (local) | Free |
| OpenThaiGPT API | Per-token ($0.4/1M tokens, displayed in THB at 1 USD = 35 THB) |

## License

Provided as-is for educational and research purposes.
