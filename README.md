# 🚀 RAG Pipeline - Production-Ready Architecture

A modular, enterprise-grade RAG (Retrieval-Augmented Generation) pipeline for building semantic search and document retrieval systems. This Phase 2 refactor provides a clean separation of concerns with CLI support.

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (CLI Entry)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  --ingest mode: Load PDF → Chunk → Embed → Store  │   │
│  │  --query mode:  Load store → Search → Retrieve    │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────┬────────────────────────────────┬────────────┘
               │                                │
       ┌───────▼──────────┐          ┌────────▼────────┐
       │ document_loader  │          │ vector_store    │
       ├──────────────────┤          ├─────────────────┤
       │ load_pdf         │          │ initialize_     │
       │ chunk_documents  │          │   embeddings    │
       └──────────────────┘          │ get_or_create   │
                                     │ retrieve_docs   │
                                     └─────────────────┘
                                            │
                                     ┌──────▼──────┐
                                     │  ChromaDB   │
                                     │  (Local)    │
                                     └─────────────┘
```

## 📋 Tech Stack

- **Python 3.8+**
- **LangChain** - Orchestration framework
- **Google Gemini API** - Text embeddings (text-embedding-004)
- **ChromaDB** - Local vector database with auto-persistence
- **PyPDF** - PDF document loading
- **python-dotenv** - Environment configuration

## 🎯 Module Breakdown

### 1. `document_loader.py`
Handles PDF loading and intelligent text chunking.

**Functions:**
- `load_pdf_document(pdf_path)` - Loads PDF pages as Document objects
- `chunk_documents(documents, chunk_size, chunk_overlap)` - Splits with RecursiveCharacterTextSplitter

**Features:**
- Error handling for missing files
- Preserves document metadata (source, page numbers)
- Configurable chunk size and overlap

### 2. `vector_store.py`
Manages embeddings and the vector database lifecycle.

**Functions:**
- `initialize_embeddings()` - Initializes GoogleGenerativeAIEmbeddings
- `get_or_create_vector_store(db_path, chunked_documents, embeddings)` - **CRITICAL**: Smart DB management
  - ✅ If DB exists: Load from disk (fast, cheap)
  - ✅ If DB missing + documents provided: Create new DB
  - ✅ If DB missing + no documents: Raise error
- `retrieve_similar_documents(vector_store, query, k)` - Semantic search
- `print_retrieval_results(query, results)` - Pretty-print results

**Key Improvements:**
- ✅ **FIXED**: Removed manual `.persist()` (Chroma auto-persists)
- ✅ **FIXED**: Removed duplicate embeddings (load existing DB)
- ✅ **FIXED**: Uses `langchain_community.vectorstores import Chroma`

### 3. `main.py`
CLI entry point with argparse-based routing.

**Modes:**
- `--ingest PDF_PATH` - Load and store PDF
- `--query QUERY_TEXT` - Retrieve documents

**Optional Parameters:**
- `--db DB_PATH` - Custom ChromaDB path (default: `./chroma_db`)
- `--k K` - Top-k results (default: 3)

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Get your Google API key from: https://ai.google.dev/

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

## 🚀 Quick Start

### Ingest a PDF

```bash
python main.py --ingest hesse_philosophy_analysis.pdf
```

**Output:**
```
================================================================================
🚀 RAG PIPELINE - INGEST MODE
================================================================================

[1/4] Loading environment configuration...
✓ Environment loaded successfully

[2/4] Loading and chunking PDF document...
✓ Successfully loaded PDF: hesse_philosophy_analysis.pdf
✓ Total pages loaded: 42
✓ Document chunking completed
✓ Total chunks created: 156
✓ Chunk size: 1000 chars | Overlap: 200 chars

[3/4] Initializing embeddings model...
✓ GoogleGenerativeAIEmbeddings initialized (text-embedding-004)

[4/4] Creating/updating vector store...
✓ Creating new ChromaDB at: C:\path\to\chroma_db
✓ Embedding 156 chunks with Google Gemini...
✓ ChromaDB created and persisted successfully
✓ Database location: C:\path\to\chroma_db
✓ Total chunks embedded: 156

✅ RAG Pipeline - Ingest completed successfully!
📌 Vector store is ready at: C:\path\to\chroma_db
```

### Query the Vector Store

```bash
python main.py --query "What is the nature of consciousness?"
```

**Output:**
```
================================================================================
🔍 RAG PIPELINE - QUERY MODE
================================================================================

[1/3] Loading environment configuration...
✓ Environment loaded successfully

[2/3] Initializing embeddings model...
✓ GoogleGenerativeAIEmbeddings initialized (text-embedding-004)

[3/3] Loading vector store and retrieving documents...
✓ Existing ChromaDB found at: C:\path\to\chroma_db
✓ Loading vector store from disk (skipping embedding step)...
✓ ChromaDB loaded successfully
✓ Documents in store: 156

⏳ Searching for documents similar to: 'What is the nature of consciousness?'

================================================================================
📋 RETRIEVAL RESULTS
================================================================================
Query: 'What is the nature of consciousness?'
Retrieved: 3 document(s)

--- Document 1 ---
Content:
[Relevant passage from your PDF...]

Metadata:
  source: hesse_philosophy_analysis.pdf
  page: 15

[Additional results...]

✅ Query completed successfully!
```

## 🔧 Usage Examples

### With Custom Database Path

```bash
# Ingest to custom location
python main.py --ingest paper.pdf --db ./my_documents_db

# Query from custom location
python main.py --query "your question" --db ./my_documents_db
```

### Retrieve More Results

```bash
python main.py --query "your question" --k 5
```

### Show Help

```bash
python main.py --help
```

## 📁 Project Structure

```
WijaiWaiV2/
├── main.py                          # CLI entry point
├── document_loader.py               # PDF loading & chunking
├── vector_store.py                  # Embeddings & ChromaDB
├── requirements.txt                 # Python dependencies
├── .env                             # API key (create from .env.example)
├── .env.example                     # Template
├── README.md                        # This file
├── hesse_philosophy_analysis.pdf    # Your sample PDF
└── chroma_db/                       # Auto-created vector store
    ├── chroma-collections.parquet
    ├── chroma-embeddings.parquet
    └── [other shard files]
```

## 🔄 Workflow Examples

### Initial Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add GOOGLE_API_KEY to .env

# 3. Place your PDF in the working directory

# 4. Ingest the PDF
python main.py --ingest hesse_philosophy_analysis.pdf

# 5. Start querying
python main.py --query "What are key philosophical concepts?"
```

### Multiple Documents

```bash
# Ingest first document
python main.py --ingest document1.pdf --db ./shared_db

# Ingest second document to same DB
python main.py --ingest document2.pdf --db ./shared_db

# Query combined database
python main.py --query "your question" --db ./shared_db
```

## 🐛 Critical Fixes Implemented

### ✅ Fix 1: Import Deprecation
**Before (❌ deprecated):**
```python
from langchain.document_loaders import PyPDFLoader
```

**After (✓ correct):**
```python
from langchain_community.document_loaders import PyPDFLoader
```

### ✅ Fix 2: Manual Persist Removed
**Before (❌ deprecated):**
```python
vector_store = Chroma.from_documents(...)
vector_store.persist()  # ← Deprecated, error-prone
```

**After (✓ correct):**
```python
vector_store = Chroma.from_documents(
    persist_directory=db_path,  # ← Auto-persists in Chroma v0.4+
    ...
)
# No manual .persist() call needed
```

### ✅ Fix 3: Smart Database Lifecycle
**Before (❌ expensive recreation every run):**
```python
# Always creates new embeddings, even if DB exists!
vector_store = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    persist_directory=db_path,
)
# Result: Wastes API credits on duplicate embeddings
```

**After (✓ efficient reuse):**
```python
def get_or_create_vector_store(db_path, chunked_documents=None, embeddings=None):
    if os.path.exists(db_path):
        # Load existing DB (no embedding cost!)
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
        )
    else:
        # Only create if new
        vector_store = Chroma.from_documents(
            documents=chunked_documents,
            embedding=embeddings,
            persist_directory=db_path,
        )
    return vector_store
```

**Impact:** Saves ~99% of embedding API calls after initial ingestion.

## ⚙️ Configuration

### Adjust Chunking Parameters

Edit `main.py`, line ~128:

```python
chunked_documents = chunk_documents(
    documents,
    chunk_size=2000,      # Larger chunks
    chunk_overlap=400     # More overlap
)
```

### Change Embedding Model

*(Future support)* In `vector_store.py`:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-005"  # When available
)
```

## 🔍 Troubleshooting

### Error: "GOOGLE_API_KEY not found"

✅ Solution:
- Create `.env` file in working directory
- Add: `GOOGLE_API_KEY=your_key_from_ai.google.dev`
- No spaces around `=`

### Error: "PDF file not found"

✅ Solution:
```bash
# Use correct path
python main.py --ingest ./documents/paper.pdf

# Or use absolute path
python main.py --ingest "C:\Users\USER\Documents\paper.pdf"
```

### Error: "ChromaDB not found at ./chroma_db"

✅ Solution:
- Run ingest mode first: `python main.py --ingest document.pdf`
- Or use existing DB path: `python main.py --query "..." --db ./existing_db`

### "Module not found" errors

✅ Solution:
```bash
pip install -r requirements.txt
python --version  # Ensure 3.8+
```

### Slow first run

✅ Expected:
- First ingest: Embedding takes 1-10 min (API calls)
- Subsequent queries: <1 sec (loaded from disk)

## 📊 API Cost Estimate

| Operation | Cost | Notes |
|-----------|------|-------|
| Ingest 100 chunks | ~$0.001 | One-time, uses text-embedding-004 |
| Query (existing DB) | ~$0.00005 | One embedding per query |
| Store 1000 chunks | Free | ChromaDB is local, no storage fees |

**With the database reuse fix:** Only ingest once, then query cheaply!

## 🚀 Next Steps (Phase 3)

- 📝 **LLM Chat Integration** - Add Gemini Pro for response generation
- 🔗 **Multi-document Support** - Batch ingestion for research corpora
- 🎯 **Reranking** - Cross-encoders for better relevance
- 💾 **Persistent Chat History** - Save conversations
- 🔐 **Authentication** - Multi-user support

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Google Generative AI API](https://ai.google.dev/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [PyPDF Documentation](https://github.com/py-pdf/pypdf)

## 📄 License

Provided as-is for educational and research purposes.

---

**Enterprise-ready. Cost-efficient. Modular. 🎓**
