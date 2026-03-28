"""
Vector Store Module — Pinecone with Multi-Tenant Namespaces
Handles embeddings initialization and Pinecone vector database management.

Key features:
- Pinecone cloud vector DB with per-user namespace isolation (Google User ID)
- Embedding model: intfloat/multilingual-e5-large (dim=768, supports Thai)
- Rich metadata on every chunk (paper_title, authors, year, section, etc.)
- Parent-Child Chunking: search small children, retrieve large parent context
- Summary Embedding: AI-generated summaries stored alongside chunks
- @st.cache_resource on embeddings and Pinecone client to avoid repeated loads
"""

import os
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import database

# Conditional Streamlit import — allows CLI usage (main.py) without Streamlit
try:
    import streamlit as st
    _HAS_STREAMLIT = hasattr(st, "cache_resource")
except ImportError:
    _HAS_STREAMLIT = False

# Load environment variables
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wijaiwai")


# ── Embedding Model ──────────────────────────────────────────────────────────

def _load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the multilingual embedding model.
    Uses intfloat/multilingual-e5-large (768 dimensions, supports Thai).

    Returns:
        SentenceTransformer: Initialized embedding model
    """
    try:
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        print("OK  Embedding model loaded (intfloat/multilingual-e5-large, dim=768)")
        return model
    except Exception as e:
        raise ValueError(f"Error initializing embedding model: {str(e)}")


# Apply Streamlit caching if available, otherwise use plain function
if _HAS_STREAMLIT:
    get_embedding_model = st.cache_resource(_load_embedding_model)
else:
    get_embedding_model = _load_embedding_model


def _embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the loaded model.
    For E5 models, prepend 'query: ' or 'passage: ' prefix for best results.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of 768 floats)
    """
    # E5 models expect 'passage: ' prefix for documents being indexed
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def _embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """
    Embed a single query text.
    For E5 models, use 'query: ' prefix for queries.

    Args:
        model: Loaded SentenceTransformer model
        query: Query string

    Returns:
        Embedding vector (list of 768 floats)
    """
    prefixed = f"query: {query}"
    embedding = model.encode([prefixed], show_progress_bar=False, normalize_embeddings=True)
    return embedding[0].tolist()


# ── Pinecone Client ──────────────────────────────────────────────────────────

def _load_pinecone_index():
    """
    Initialize Pinecone client and connect to the 'wijaiwai' index.
    Uses @st.cache_resource so the connection is created only once.

    Returns:
        pinecone.Index: Connected Pinecone index object
    """
    from pinecone import Pinecone

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found in .env. "
            "Please set PINECONE_API_KEY=your_key_here"
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"OK  Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
    return index


# Apply Streamlit caching if available
if _HAS_STREAMLIT:
    get_pinecone_index = st.cache_resource(_load_pinecone_index)
else:
    get_pinecone_index = _load_pinecone_index


# ── Upsert (Ingest) ─────────────────────────────────────────────────────────

def upsert_documents(chunks: list, user_id: str,
                     embedding_model: SentenceTransformer = None) -> int:
    """
    Embed chunks and upsert them into Pinecone under the user's namespace.

    Each chunk can be a LangChain Document object or a dict with
    'page_content' and 'metadata' keys.

    Args:
        chunks: List of Document objects or dicts with page_content/metadata
        user_id: Google user ID used as Pinecone namespace
        embedding_model: Optional pre-loaded model (loads from cache if None)

    Returns:
        int: Number of vectors upserted
    """
    if not chunks:
        return 0

    if embedding_model is None:
        embedding_model = get_embedding_model()

    index = get_pinecone_index()

    # Extract texts and metadata from chunks
    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        # Support both LangChain Document objects and plain dicts
        if hasattr(chunk, 'page_content'):
            text = chunk.page_content
            meta = dict(chunk.metadata) if chunk.metadata else {}
        else:
            text = chunk.get('page_content', chunk.get('content', ''))
            meta = dict(chunk.get('metadata', {}))

        # Store content in metadata for retrieval
        meta['content'] = text[:39000]  # Pinecone metadata limit ~40KB

        # Ensure source_type exists
        if 'source_type' not in meta:
            meta['source_type'] = 'document'

        # Generate unique vector ID
        vec_id = f"{meta.get('doc_name', 'doc')}_{uuid.uuid4().hex[:12]}"

        # Clean metadata: Pinecone only supports str, int, float, bool, list[str]
        clean_meta = {}
        for k, v in meta.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = [str(item) for item in v]
            else:
                clean_meta[k] = str(v)

        texts.append(text)
        metadatas.append(clean_meta)
        ids.append(vec_id)

    # Embed all texts
    embeddings = _embed_texts(embedding_model, texts)

    # Upsert in batches of 100 (Pinecone recommended batch size)
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(ids), batch_size):
        batch_vectors = []
        for j in range(i, min(i + batch_size, len(ids))):
            batch_vectors.append({
                "id": ids[j],
                "values": embeddings[j],
                "metadata": metadatas[j],
            })
        index.upsert(vectors=batch_vectors, namespace=user_id)
        total_upserted += len(batch_vectors)

    print(f"OK  Upserted {total_upserted} vectors to namespace '{user_id}'")
    return total_upserted


# ── Ingest with Advanced RAG ────────────────────────────────────────────────

def ingest_documents(child_chunks: list, parent_records: list,
                     user_id: str, summary_docs: list = None,
                     embedding_model: SentenceTransformer = None):
    """
    Ingest child chunks (and optional summaries) into Pinecone,
    and save parent chunks to SQLite.

    Args:
        child_chunks: List of child Document objects with parent_id in metadata
        parent_records: List of dicts from create_parent_child_chunks()
        user_id: Google user ID for namespace isolation
        summary_docs: Optional list of summary Document objects
        embedding_model: Optional pre-loaded model
    """
    # Save parent chunks to SQLite (batch for performance)
    database.save_parent_chunks_batch(parent_records)
    print(f"OK  Saved {len(parent_records)} parent chunks to SQLite")

    # Combine child chunks and summary docs for single upsert
    all_chunks = list(child_chunks) if child_chunks else []
    if summary_docs:
        all_chunks.extend(summary_docs)

    if all_chunks:
        upsert_documents(all_chunks, user_id, embedding_model)


def ingest_note(note_id: int, title: str, content: str,
                user_id: str, embedding_model: SentenceTransformer = None):
    """
    Ingest a research note into Pinecone with source_type='note'.

    Args:
        note_id: SQLite note ID
        title: Note title
        content: Note content text
        user_id: Google user ID for namespace
        embedding_model: Optional pre-loaded model
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from document_loader import get_adaptive_chunk_params

    total_chars = len(content)
    chunk_size, chunk_overlap = get_adaptive_chunk_params(total_chars)

    doc = Document(
        page_content=content,
        metadata={
            'title': title,
            'source_type': 'note',
            'note_id': note_id,
            'chunk_type': 'child',
            'doc_name': f'note_{note_id}',
            'created_at': date.today().isoformat(),
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    upsert_documents(chunks, user_id, embedding_model)
    print(f"OK  Ingested note '{title}' ({len(chunks)} chunks, size={chunk_size})")


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_unified(query: str, user_id: str, k: int = 3,
                     source_type: str = None,
                     expand_parents: bool = True,
                     embedding_model: SentenceTransformer = None) -> list:
    """
    Unified retrieval from Pinecone with optional metadata filtering,
    parent-child expansion, and deduplication.

    Args:
        query: Natural language query
        user_id: Google user ID (Pinecone namespace)
        k: Number of results to retrieve
        source_type: Optional filter -- "document", "note", "web_page", or None (all)
        expand_parents: If True, replace child chunks with their parent content
        embedding_model: Optional pre-loaded model

    Returns:
        list: Retrieved documents as simple objects with .page_content and .metadata
    """
    from langchain_core.documents import Document

    if not query or not user_id:
        return []

    if embedding_model is None:
        embedding_model = get_embedding_model()

    try:
        index = get_pinecone_index()
    except Exception as e:
        print(f"Warning: Could not connect to Pinecone: {e}")
        return []

    # Embed the query
    query_embedding = _embed_query(embedding_model, query)

    # Build metadata filter
    filter_dict = None
    if source_type:
        filter_dict = {"source_type": {"$eq": source_type}}

    # Query Pinecone
    try:
        results = index.query(
            vector=query_embedding,
            top_k=k * 2,  # Fetch extra for deduplication
            namespace=user_id,
            filter=filter_dict,
            include_metadata=True,
        )
    except Exception as e:
        print(f"Warning: Pinecone query error: {e}")
        return []

    if not results or not results.get("matches"):
        return []

    # Convert Pinecone results to Document objects
    docs = []
    for match in results["matches"]:
        meta = dict(match.get("metadata", {}))
        content = meta.pop("content", "")
        docs.append(Document(
            page_content=content,
            metadata=meta,
        ))

    # Parent-Child expansion: swap child content with parent content
    if expand_parents and docs:
        parent_ids = list({
            doc.metadata.get('parent_id')
            for doc in docs
            if doc.metadata.get('parent_id')
        })
        if parent_ids:
            parents = database.get_parent_chunks_batch(parent_ids)
            expanded = []
            seen_parents = set()
            for doc in docs:
                pid = doc.metadata.get('parent_id')
                if pid and pid in parents and pid not in seen_parents:
                    seen_parents.add(pid)
                    parent_data = parents[pid]
                    expanded.append(Document(
                        page_content=parent_data['content'],
                        metadata={
                            **doc.metadata,
                            'chunk_type': 'parent_expanded',
                            'parent_page': parent_data.get('page_number'),
                            'parent_section': parent_data.get('section'),
                        }
                    ))
                elif not pid:
                    # Document without parent (e.g., notes, summaries)
                    expanded.append(doc)
            docs = expanded if expanded else docs

    # Deduplicate by first 200 chars
    seen: set = set()
    unique: list = []
    for doc in docs:
        sig = doc.page_content[:200]
        if sig not in seen:
            seen.add(sig)
            unique.append(doc)

    return unique[:k]


# ── Delete ───────────────────────────────────────────────────────────────────

def delete_document(doc_name: str, user_id: str) -> None:
    """
    Delete all vectors for a document from Pinecone by listing and filtering.

    Since Pinecone serverless doesn't support delete-by-metadata-filter directly,
    we list vectors in the namespace and delete by ID prefix matching.

    Args:
        doc_name: The doc_name metadata value to match
        user_id: Google user ID (Pinecone namespace)
    """
    try:
        index = get_pinecone_index()
        # Use list + delete approach for serverless
        # First try delete with metadata filter (works on pod-based indexes)
        # For serverless, we use the prefix-based approach
        # Since our IDs start with doc_name, we can use prefix delete
        safe_prefix = doc_name.replace(" ", "_")[:50]

        # Try listing vectors with the prefix
        try:
            listed = index.list(namespace=user_id, prefix=safe_prefix)
            if listed and hasattr(listed, 'vectors'):
                vec_ids = [v for v in listed.vectors]
                if vec_ids:
                    index.delete(ids=vec_ids, namespace=user_id)
                    print(f"OK  Deleted {len(vec_ids)} vectors for '{doc_name}'")
                    return
        except Exception:
            pass

        # Fallback: delete all vectors in namespace matching doc_name
        # Query with a dummy vector to find matching IDs
        try:
            model = get_embedding_model()
            dummy_vec = _embed_query(model, doc_name)
            results = index.query(
                vector=dummy_vec,
                top_k=1000,
                namespace=user_id,
                filter={"doc_name": {"$eq": doc_name}},
                include_metadata=False,
            )
            if results and results.get("matches"):
                ids_to_delete = [m["id"] for m in results["matches"]]
                if ids_to_delete:
                    # Delete in batches of 1000
                    for i in range(0, len(ids_to_delete), 1000):
                        batch = ids_to_delete[i:i + 1000]
                        index.delete(ids=batch, namespace=user_id)
                    print(f"OK  Deleted {len(ids_to_delete)} vectors for '{doc_name}'")
        except Exception as e:
            print(f"Warning: Could not delete vectors for '{doc_name}': {e}")

    except Exception as e:
        print(f"Warning: Delete operation failed: {e}")


def delete_by_metadata(filter_key: str, filter_value, user_id: str) -> None:
    """
    Delete vectors from Pinecone matching a metadata filter.

    Args:
        filter_key: Metadata key to filter on (e.g., 'note_id', 'web_page_id')
        filter_value: Value to match
        user_id: Google user ID (Pinecone namespace)
    """
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        # Use a dummy query to find matching vectors
        dummy_vec = _embed_query(model, str(filter_value))
        results = index.query(
            vector=dummy_vec,
            top_k=1000,
            namespace=user_id,
            filter={filter_key: {"$eq": filter_value}},
            include_metadata=False,
        )
        if results and results.get("matches"):
            ids_to_delete = [m["id"] for m in results["matches"]]
            if ids_to_delete:
                for i in range(0, len(ids_to_delete), 1000):
                    batch = ids_to_delete[i:i + 1000]
                    index.delete(ids=batch, namespace=user_id)
                print(f"OK  Deleted {len(ids_to_delete)} vectors matching {filter_key}={filter_value}")
    except Exception as e:
        print(f"Warning: delete_by_metadata failed: {e}")


# ── Legacy compatibility aliases ─────────────────────────────────────────────

def initialize_embeddings():
    """Legacy alias for get_embedding_model(). Returns the model."""
    return get_embedding_model()


def print_retrieval_results(query: str, results: list):
    """Pretty-print retrieval results to stdout."""
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    print(f"Query: '{query}'")
    print(f"Retrieved: {len(results)} document(s)\n")

    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content:\n{doc.page_content[:500]}")
        if len(doc.page_content) > 500:
            print("...")
        if doc.metadata:
            print("Metadata:")
            for key, value in doc.metadata.items():
                if key != 'content':  # Skip the full content field
                    print(f"  {key}: {value}")

    print("\n" + "=" * 80 + "\n")
