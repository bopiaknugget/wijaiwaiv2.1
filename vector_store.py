"""
Vector Store Module — Unified Collection with Advanced RAG
Handles embeddings initialization and ChromaDB management.

Key features (optimized):
- Single unified collection per project, separated by metadata (source_type)
- Rich metadata on every chunk (paper_title, authors, year, section, etc.)
- Parent-Child Chunking: search small children, retrieve large parent context
- Summary Embedding: AI-generated summaries stored alongside chunks
- Maximal Marginal Relevance (MMR) for diverse, high-quality retrieval
- @st.cache_resource on embeddings to avoid repeated 400 MB model loads
"""

import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import database


# ── Default unified collection name ──────────────────────────────────────────
UNIFIED_COLLECTION = "unified"
UNIFIED_DB_PATH = "./Database/unified_chroma_db"


@st.cache_resource
def initialize_embeddings():
    """
    Initialize a local multilingual embedding model using HuggingFace.
    Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (~400 MB).
    Supports Thai and many other languages, requires no API key.

    The @st.cache_resource decorator ensures the model is loaded only once
    per Streamlit session — subsequent calls return the cached instance.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✓ HuggingFaceEmbeddings initialized (paraphrase-multilingual-MiniLM-L12-v2)")
        return embeddings
    except Exception as e:
        raise ValueError(f"❌ Error initializing HuggingFace embeddings: {str(e)}")


def get_or_create_vector_store(db_path=UNIFIED_DB_PATH, chunked_documents=None,
                                embeddings=None, collection_name=UNIFIED_COLLECTION):
    """
    Intelligent vector store lifecycle manager for a unified collection.

    Behaviour:
    - DB exists + no docs  → load existing DB (fast, no re-embedding)
    - DB exists + docs     → load existing DB and append new documents
    - DB missing + docs    → create new DB from documents
    - DB missing + no docs → create empty DB (ready for future adds)

    Args:
        db_path (str): Directory path for ChromaDB storage
        chunked_documents (list | None): Pre-chunked LangChain Documents
        embeddings: Initialized HuggingFaceEmbeddings instance
        collection_name (str): ChromaDB collection name (default: "unified")

    Returns:
        Chroma: Initialized vector store

    Raises:
        ValueError: If embeddings is None
        Exception: If any ChromaDB operation fails
    """
    if embeddings is None:
        raise ValueError("❌ embeddings parameter is required")

    try:
        db_exists = os.path.exists(db_path) and os.path.isdir(db_path)

        if db_exists:
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            if chunked_documents:
                vector_store.add_documents(documents=chunked_documents)
                print(f"✓ Appended {len(chunked_documents)} chunks to existing DB at {db_path}")
            else:
                count = vector_store._collection.count()
                print(f"✓ Loaded existing ChromaDB at {db_path} ({count} chunks)")
        else:
            if chunked_documents:
                vector_store = Chroma.from_documents(
                    documents=chunked_documents,
                    embedding=embeddings,
                    persist_directory=db_path,
                    collection_name=collection_name
                )
                print(f"✓ Created ChromaDB with {len(chunked_documents)} chunks at {db_path}")
            else:
                vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                print(f"✓ Initialized empty ChromaDB at {db_path}")

        return vector_store

    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"❌ Error with vector store operation: {str(e)}")


# ── Ingest with Advanced RAG ────────────────────────────────────────────────

def ingest_documents(vector_store, child_chunks: list, parent_records: list,
                     summary_docs: list = None):
    """
    Ingest child chunks (and optional summaries) into the unified ChromaDB,
    and save parent chunks to SQLite.

    Args:
        vector_store: Chroma instance (unified collection)
        child_chunks: List of child Document objects with parent_id in metadata
        parent_records: List of dicts from create_parent_child_chunks()
        summary_docs: Optional list of summary Document objects
    """
    # Save parent chunks to SQLite
    for parent in parent_records:
        database.save_parent_chunk(
            parent_id=parent['id'],
            content=parent['content'],
            source_file=parent.get('source_file'),
            page_number=parent.get('page_number'),
            section=parent.get('section'),
        )
    print(f"✓ Saved {len(parent_records)} parent chunks to SQLite")

    # Add child chunks to ChromaDB
    if child_chunks:
        vector_store.add_documents(documents=child_chunks)
        print(f"✓ Added {len(child_chunks)} child chunks to ChromaDB")

    # Add summary embeddings to ChromaDB
    if summary_docs:
        vector_store.add_documents(documents=summary_docs)
        print(f"✓ Added {len(summary_docs)} summary embeddings to ChromaDB")


def ingest_note(vector_store, note_id: int, title: str, content: str,
                embeddings, user_id: str = None, project_id: str = None):
    """
    Ingest a research note into the unified collection with source_type='note'.
    Uses small chunks for precise retrieval.

    Args:
        vector_store: Chroma instance (unified collection)
        note_id: SQLite note ID
        title: Note title
        content: Note content text
        embeddings: Initialized embeddings instance
        user_id: Optional user identifier
        project_id: Optional project identifier

    Returns:
        Chroma: The vector store (created if it was None)
    """
    from datetime import date
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    if user_id:
        doc.metadata['user_id'] = user_id
    if project_id:
        doc.metadata['project_id'] = project_id

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    if vector_store is None:
        vector_store = get_or_create_vector_store(
            db_path=UNIFIED_DB_PATH,
            chunked_documents=chunks,
            embeddings=embeddings,
            collection_name=UNIFIED_COLLECTION,
        )
    else:
        vector_store.add_documents(chunks)

    print(f"✓ Ingested note '{title}' ({len(chunks)} chunks, size={chunk_size})")
    return vector_store


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_similar_documents(vector_store, query: str, k: int = 3) -> list:
    """Standard cosine-similarity search."""
    if not query or not isinstance(query, str):
        raise ValueError("❌ Query must be a non-empty string")
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        raise Exception(f"❌ Error during similarity search: {str(e)}")


def retrieve_mmr(vector_store, query: str, k: int = 5,
                 fetch_k: int = 20, lambda_mult: float = 0.6,
                 where_filter: dict = None) -> list:
    """
    Maximal Marginal Relevance (MMR) retrieval with optional metadata filtering.

    Args:
        vector_store: Chroma instance to search
        query: Natural language query
        k: Number of final results to return
        fetch_k: Candidate pool size before MMR re-ranking
        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
        where_filter: Optional ChromaDB where filter for metadata

    Returns:
        list[Document]: k diverse and relevant documents
    """
    if not query or not isinstance(query, str):
        raise ValueError("❌ Query must be a non-empty string")
    try:
        kwargs = dict(query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        if where_filter:
            kwargs['filter'] = where_filter
        return vector_store.max_marginal_relevance_search(**kwargs)
    except Exception as e:
        raise Exception(f"❌ Error during MMR retrieval: {str(e)}")


def retrieve_unified(vector_store, query: str, k: int = 3,
                     source_type: str = None,
                     expand_parents: bool = True) -> list:
    """
    Unified retrieval from the single collection with optional metadata filtering,
    parent-child expansion, and deduplication.

    This replaces the old retrieve_from_both_stores() function.

    Args:
        vector_store: Chroma instance (unified collection)
        query: Natural language query
        k: Number of results to retrieve
        source_type: Optional filter — "document", "note", or None (all)
        expand_parents: If True, replace child chunks with their parent content

    Returns:
        list[Document]: Retrieved and optionally parent-expanded documents
    """
    if vector_store is None:
        return []

    # Build optional metadata filter
    where_filter = None
    if source_type:
        where_filter = {"source_type": source_type}

    try:
        results = retrieve_mmr(
            vector_store, query, k=k, fetch_k=k * 4,
            where_filter=where_filter
        )
    except Exception as e:
        print(f"⚠ Retrieval error: {e}")
        return []

    # Parent-Child expansion: swap child content with parent content
    if expand_parents and results:
        parent_ids = list({
            doc.metadata.get('parent_id')
            for doc in results
            if doc.metadata.get('parent_id')
        })
        if parent_ids:
            parents = database.get_parent_chunks_batch(parent_ids)
            expanded = []
            seen_parents = set()
            for doc in results:
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
            results = expanded if expanded else results

    # Deduplicate by first 200 chars
    seen: set = set()
    unique: list = []
    for doc in results:
        sig = doc.page_content[:200]
        if sig not in seen:
            seen.add(sig)
            unique.append(doc)

    return unique


# ── Legacy compatibility wrapper ─────────────────────────────────────────────

def retrieve_from_both_stores(doc_store, note_store, query: str, k: int = 3) -> list:
    """
    Legacy compatibility wrapper — redirects to retrieve_unified().

    If a unified store is available (doc_store or note_store), uses it directly.
    This function exists so that existing app.py code works during the transition.
    """
    store = doc_store or note_store
    return retrieve_unified(store, query, k=k)


def print_retrieval_results(query: str, results: list):
    """Pretty-print retrieval results to stdout."""
    print("\n" + "=" * 80)
    print("📋 RETRIEVAL RESULTS")
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
                print(f"  {key}: {value}")

    print("\n" + "=" * 80 + "\n")
