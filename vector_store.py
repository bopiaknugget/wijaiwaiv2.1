"""
Vector Store Module
Handles embeddings initialization and ChromaDB management.

Key features:
- Separate stores for uploaded Documents and research Notes
- Maximal Marginal Relevance (MMR) for diverse, high-quality retrieval
- Merged retrieval across both stores with deduplication
- @st.cache_resource on embeddings to avoid repeated 400 MB model loads
"""

import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


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


def get_or_create_vector_store(db_path="./chroma_db", chunked_documents=None,
                                embeddings=None, collection_name="documents"):
    """
    Intelligent vector store lifecycle manager.

    Behaviour:
    - DB exists + no docs  → load existing DB (fast, no re-embedding)
    - DB exists + docs     → load existing DB and append new documents
    - DB missing + docs    → create new DB from documents
    - DB missing + no docs → create empty DB (ready for future adds)

    Args:
        db_path (str): Directory path for ChromaDB storage
        chunked_documents (list | None): Pre-chunked LangChain Documents
        embeddings: Initialized HuggingFaceEmbeddings instance
        collection_name (str): ChromaDB collection name (use "notes" for the notes store)

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


def retrieve_similar_documents(vector_store, query: str, k: int = 3) -> list:
    """
    Standard cosine-similarity search.

    Args:
        vector_store: Chroma instance to search
        query: Natural language query
        k: Number of results to return

    Returns:
        list[Document]: Top-k most similar documents
    """
    if not query or not isinstance(query, str):
        raise ValueError("❌ Query must be a non-empty string")
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        raise Exception(f"❌ Error during similarity search: {str(e)}")


def retrieve_mmr(vector_store, query: str, k: int = 5,
                 fetch_k: int = 20, lambda_mult: float = 0.6) -> list:
    """
    Maximal Marginal Relevance (MMR) retrieval.

    MMR balances relevance and diversity: it picks results that are highly
    relevant to the query while also being different from each other,
    reducing redundant / near-duplicate chunks in the context window.

    Args:
        vector_store: Chroma instance to search
        query: Natural language query
        k: Number of final results to return
        fetch_k: Candidate pool size before MMR re-ranking (should be >> k)
        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity (0.6 recommended)

    Returns:
        list[Document]: k diverse and relevant documents
    """
    if not query or not isinstance(query, str):
        raise ValueError("❌ Query must be a non-empty string")
    try:
        return vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    except Exception as e:
        raise Exception(f"❌ Error during MMR retrieval: {str(e)}")


def retrieve_from_both_stores(doc_store, note_store, query: str, k: int = 3) -> list:
    """
    Query both the document store and the notes store using MMR,
    then merge and deduplicate the results.

    This gives the LLM richer context by pulling from both uploaded
    documents and the user's saved research notes simultaneously.

    Args:
        doc_store: Chroma instance for uploaded documents (may be None)
        note_store: Chroma instance for research notes (may be None)
        query: Natural language query
        k: Number of results to retrieve from each store

    Returns:
        list[Document]: Merged, deduplicated results (up to 2*k items)
    """
    results = []

    if doc_store is not None:
        try:
            doc_results = retrieve_mmr(doc_store, query, k=k, fetch_k=k * 4)
            results.extend(doc_results)
        except Exception as e:
            print(f"⚠ Doc store retrieval error: {e}")

    if note_store is not None:
        try:
            note_results = retrieve_mmr(note_store, query, k=k, fetch_k=k * 4)
            results.extend(note_results)
        except Exception as e:
            print(f"⚠ Note store retrieval error: {e}")

    # Deduplicate by first 200 chars of content (fast signature)
    seen: set = set()
    unique: list = []
    for doc in results:
        sig = doc.page_content[:200]
        if sig not in seen:
            seen.add(sig)
            unique.append(doc)

    return unique


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
