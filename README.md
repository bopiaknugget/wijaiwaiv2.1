# Research Workbench — AI-Powered RAG with Research Editor

An intelligent research platform combining RAG (Retrieval-Augmented Generation) with a specialized text editor for academic research. Supports bilingual Thai/English content using an LLM model as the primary backend, Pinecone cloud vector store, and Google OAuth for multi-user isolation.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                         │
│                    Google OAuth 2.0 Login Screen                     │
│                                                                      │
│  ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐  │
│  │   Sidebar    │  │  Research Workbench    │  │  Assistant        │  │
│  │  Documents   │  │   Title + Editor      │  │   Chat + RAG Q&A │  │
│  │  Notes       │  │   Save / Load / Export│  │   Research Mode  │  │
│  │  Web Pages   │  │   (SQLite-backed)     │  │   Advisor Review │  │
│  └──────┬───────┘  └───────────┬───────────┘  └────────┬──────────┘  │
│         │                      │                       │              │
└─────────┼──────────────────────┼───────────────────────┼──────────────┘
          │                      │                       │
  ┌───────▼──────────┐   ┌──────▼───────┐   ┌──────────▼──────────┐
  │ document_loader  │   │  database.py │   │    generator.py     │
  │ PDF/TXT/DOCX     │   │  SQLite      │   │  LLM model API      │
  │ Parent-Child     │   │  editor docs │   │  Streaming output   │
  │ Chunking         │   │  token usage │   │  Intent detection   │
  └───────┬──────────┘   └──────────────┘   └──────────┬──────────┘
          │                                            │
  ┌───────▼──────────────────────────────────┐         │
  │           vector_store.py                │         │
  │  Pinecone (cloud, per-user namespaces)   │◄────────┘
  │  multilingual-e5-large (Pinecone Infer.) │
  │  Hybrid BM25+Vector Retrieval            │
  │  Parent-Child Expansion                  │
  └───────┬──────────────────────────────────┘
          │
  ┌───────▼──────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  Pinecone Index  │    │  auth.py         │    │  reviewer.py     │
  │  (wijaiwai)      │    │  Google OAuth    │    │  Thesis Advisor  │
  │  Per-user        │    │  2.0 Login       │    │  Review System   │
  │  namespaces      │    │                  │    │  Chunked review  │
  └──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web UI** | Streamlit (3-panel layout) |
| **Authentication** | Google OAuth 2.0 |
| **LLM** | LLM model API (Thai/English support) |
| **Embeddings** | Pinecone Inference API — `multilingual-e5-large` (1024-dim, supports Thai) |
| **Vector DB** | Pinecone (cloud, per-user namespace isolation) |
| **Relational DB** | SQLite (`Database/research_notes.db`) |
| **Document Parsing** | PyPDF, docx2txt |
| **Web Scraping** | Trafilatura + BeautifulSoup4 (fallback) |
| **Search** | Hybrid: BM25 (0.3) + vector (0.7) fusion |

## Installation

### Prerequisites

- Python 3.8+
- Pinecone account with an index named `wijaiwai` (1024-dim, cosine metric)
- Google Cloud project with OAuth 2.0 credentials
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

# 3. Create .env file
```

Create `.env` in the project root:
```
LLM_API_KEY=your_llm_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=wijaiwai
PINECONE_HOST=your_pinecone_host_url
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8501/oauth2callback
```

> **Note**: No quotes or spaces around `=` values.

## Quick Start

### Web UI (Primary)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You will be prompted to log in with Google. After login, the three-panel interface is available:

| Panel | Features |
|-------|----------|
| **Sidebar** (left) | Upload documents (PDF/TXT/DOCX/DOC), manage research notes, manage web pages |
| **Research Workbench** (center) | Text editor with title, Save/Load/Export/Import, Undo/Redo — files stored per user in SQLite |
| **Assistant** (right) | Chat Q&A with RAG, Research Mode, Advisor Review |

### CLI

```bash
# Ingest a document
python main.py --ingest path/to/document.pdf

# Query the vector store
python main.py --query "your question here"
```

## Features

### Google OAuth Login
All users authenticate via Google before accessing the platform. Each user's documents, notes, web pages, editor files, and vector data are fully isolated by their Google user ID.

### RAG-Powered Q&A
Ask questions about your uploaded documents. The system retrieves relevant context using hybrid search (BM25 + vector fusion) scoped to your Pinecone namespace and generates answers with the LLM model. Supports follow-up questions with automatic query re-phrasing based on chat history. Streaming output for progressive display.

### Research Mode
Toggle Research Mode for comprehensive, structured answers (>=1,000 words). The LLM model writes detailed research output (introduction, literature review, analysis, findings, summary, references) directly into the editor with a higher token budget (12,000 tokens).

### Agentic Editor
The assistant classifies user intent locally before any API call:
- **Small talk** — skipped retrieval, lightweight response
- **Edit** — writes or modifies content in the editor (selection-based editing supported)
- **Chat** — answers in the chat panel with retrieved context
- **Research** — deep research with structured output to editor, streaming

### Research Notes
Create and manage notes that are embedded into Pinecone alongside documents. Notes become searchable with `source_type='note'` — great for annotations, summaries, and cross-referencing.

### Web Content Integration
Paste a URL to scrape and integrate web content:
- Automatic content extraction (trafilatura with BeautifulSoup fallback)
- AI-generated summaries and titles
- Content is chunked and embedded into Pinecone (user namespace)
- Metadata stored in SQLite for management

### Advisor Review
Click the advisor button to get your research reviewed by a strict thesis advisor persona (simulating 20+ years of expertise). Automatically handles large documents via chunked processing. Reviews cover the standard 5-chapter thesis structure:
- Chapter 1: Introduction, objectives, hypotheses, scope
- Chapter 2: Literature review, theoretical framework
- Chapter 3: Research methodology
- Chapter 4: Results presentation
- Chapter 5: Summary, discussion, recommendations

Feedback is color-coded:
- **Red** [Must fix] — Critical issues that require correction
- **Green** [Well done] — Strong sections that meet expectations
- **Yellow** [Recommendations] — Suggestions for improvement

### Advanced RAG Pipeline
- **Hybrid Search** — BM25 (0.3) + vector (0.7) fusion for improved retrieval quality
- **Per-User Isolation** — All Pinecone queries scoped to user's namespace
- **Rich Metadata** — Each chunk carries `paper_title`, `authors`, `year`, `section`, `source_type`, `doc_id`, `parent_id`, `created_at`
- **Parent-Child Chunking** — Small child chunks in Pinecone for precise search; parent content (full pages) in SQLite for richer LLM context
- **Summary Embedding** — Extractive summaries embedded alongside chunks for broad semantic matching
- **Adaptive Chunk Sizing** — Auto-adjusts based on content length (<2k→300, 2k-10k→800, >10k→1200 chars)
- **Embedding Cache** — SHA-256 hashing avoids redundant Pinecone Inference API calls

### Token Tracking
Input/output tokens are tracked per user per function in SQLite (`token_usage` table). Cost displayed in THB at `$0.4 / 1M tokens` (1 USD = 35 THB).

## Module Breakdown

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, Google OAuth splash, session state, `<think>` tag parsing, Research Workbench editor, 3-panel layout |
| `auth.py` | Google OAuth 2.0 — auth URL generation, callback handling, user info fetch |
| `generator.py` | LLM model API calls, streaming output, query re-phrasing, local intent detection, editor manipulation |
| `vector_store.py` | Pinecone cloud vector DB, hybrid retrieval, per-user namespaces, parent-child expansion |
| `document_loader.py` | PDF/TXT/DOCX loading, metadata extraction, parent-child chunking, adaptive chunk sizing, summary embedding |
| `database.py` | SQLite CRUD for notes, documents, parent chunks, web pages, editor documents, token usage, OAuth users |
| `reviewer.py` | Research advisor review system with chunked processing and color-coded thesis feedback |
| `web_scraper.py` | Web content extraction, AI summarization, content chunking into Pinecone |
| `query_router.py` | Query classification and routing |
| `main.py` | CLI entry point (`--ingest` / `--query` modes) |
| `rag_pipeline.py` | Legacy Phase 1 (unused — do not modify) |

## Project Structure

```
wijaiwaiv2.1/
├── app.py                  # Streamlit Web UI (3-panel layout, OAuth login)
├── auth.py                 # Google OAuth 2.0 integration
├── generator.py            # LLM model API + intent classifier + streaming
├── vector_store.py         # Pinecone vector DB + hybrid retrieval
├── document_loader.py      # PDF/TXT/DOCX loading & chunking
├── database.py             # SQLite CRUD (7 tables, per-user data)
├── reviewer.py             # Thesis advisor review system
├── web_scraper.py          # Web content scraping & summarization
├── query_router.py         # Query classification and routing
├── main.py                 # CLI entry point
├── rag_pipeline.py         # Legacy Phase 1 (unused)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
├── CLAUDE.md               # Claude Code guidance
├── README.md               # This file
├── Database/               # Auto-created on first run
│   └── research_notes.db   # SQLite database (7 tables)
└── md/                     # Design documents & task notes
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | Yes | LLM model API key for language generation |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector store |
| `PINECONE_INDEX_NAME` | Yes | Pinecone index name (default: `wijaiwai`) |
| `PINECONE_HOST` | Yes | Pinecone index host URL |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | Yes | Google OAuth 2.0 client secret |
| `GOOGLE_REDIRECT_URI` | Yes | OAuth callback URL (default: `http://localhost:8501/oauth2callback`) |

## Cost

| Component | Cost |
|-----------|------|
| Pinecone (Starter tier) | Free up to limits |
| Pinecone Inference API (embeddings) | Pay-per-use |
| SQLite (local) | Free |
| LLM model API | Varies by provider |
| Google OAuth | Free |

## License

Provided as-is for educational and research purposes.
