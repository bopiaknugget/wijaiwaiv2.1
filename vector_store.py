"""
Vector Store Module
Handles embeddings initialization and ChromaDB management with intelligent
database lifecycle (create if new, load if exists).
"""

import os
import shutil
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


@st.cache_resource
def initialize_embeddings():
    """
    Initialize a local multilingual embedding model using HuggingFace.
    Uses the sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 model,
    which works well for Thai and many other languages and requires no API key.

    The decorator ensures the 400MB+ model is loaded only once per Streamlit
    session; subsequent calls return the cached instance, keeping the UI
    responsive and avoiding repeated downloads.
    
    Returns:
        HuggingFaceEmbeddings: Initialized local embedding model
    """
    try:
        # Instantiate the HuggingFaceEmbeddings with the multilingual model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        print("✓ HuggingFaceEmbeddings initialized (paraphrase-multilingual-MiniLM-L12-v2)")
        
        return embeddings
    except Exception as e:
        raise ValueError(f"❌ Error initializing HuggingFace embeddings: {str(e)}")


def get_or_create_vector_store(db_path="./chroma_db", chunked_documents=None, embeddings=None):
    """
    CRITICAL FIX: Intelligent vector store management.
    If database exists, load it from disk. If not, create it from documents.
    This prevents expensive re-embedding and ensures cost efficiency.
    
    Key Features:
    - Checks if ChromaDB already exists before creating
    - Loads existing DB if present (skips expensive embedding)
    - Creates new DB only if documents are provided and DB doesn't exist
    - Automatically persists (Chroma v0.4.x+ auto-persist with persist_directory)
    - Does NOT call .persist() manually (deprecated, auto-handled by Chroma)
    
    Args:
        db_path (str): Path where ChromaDB will be stored/loaded (default: ./chroma_db)
        chunked_documents (list, optional): Chunked documents for creating new DB.
                                          If None, only loads existing DB.
        embeddings: GoogleGenerativeAIEmbeddings object required for both
                   create and load operations
        
    Returns:
        Chroma: Initialized Chroma vector store
        
    Raises:
        ValueError: If attempting to load non-existent DB without documents
        Exception: If vector store operations fail
    """
    
    # Validate embeddings parameter
    if embeddings is None:
        raise ValueError("❌ embeddings parameter is required")
    
    try:
        # Check if ChromaDB already exists on disk
        db_exists = os.path.exists(db_path) and os.path.isdir(db_path)
        
        if db_exists:
            print(f"✓ Existing ChromaDB found at: {os.path.abspath(db_path)}")
            print(f"✓ Loading vector store from disk...")
            
            # โหลดฐานข้อมูลที่มีอยู่แล้วขึ้นมา
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name="documents"
            )
            
            # --- โค้ดส่วนที่เพิ่มเข้ามาเพื่อแก้บั๊ก ---
            # ถ้าเป็นโหมด Ingest (มีการส่ง chunked_documents มาด้วย) ต้องยัดข้อมูลเพิ่มเข้าไปใน DB เดิม!
            if chunked_documents:  # guards against None AND empty list []
                print(f"✓ Appending {len(chunked_documents)} new chunks to existing database...")
                vector_store.add_documents(documents=chunked_documents)
                print(f"✓ New documents embedded and added successfully!")
            # ----------------------------------
            
            collection = vector_store._collection
            doc_count = collection.count() if hasattr(collection, 'count') else 'unknown'
            print(f"✓ Documents currently in store: {doc_count}")
            
            return vector_store
        
        else:
            # Database doesn't exist - need documents to create it
            if chunked_documents is None:
                raise ValueError(
                    f"❌ ChromaDB not found at {db_path}\n"
                    f"Please provide chunked_documents to create a new database\n"
                    f"Use: --ingest flag to create the database from a PDF"
                )
            
            print(f"✓ Creating new ChromaDB at: {os.path.abspath(db_path)}")
            print(f"✓ Embedding {len(chunked_documents)} chunks with Google Gemini...")
            
            # Create new ChromaDB from documents
            # Chroma automatically persists to persist_directory in v0.4.x+
            vector_store = Chroma.from_documents(
                documents=chunked_documents,        # Chunked documents to embed and store
                embedding=embeddings,               # Embedding function (Gemini)
                persist_directory=db_path,          # Local storage path - enables auto-persist
                collection_name="documents"         # Collection name for organization
            )
            
            # NOTE: Do NOT call vector_store.persist() - Chroma v0.4.x+ auto-persists
            # Calling persist() is deprecated and unnecessary
            
            print(f"✓ ChromaDB created and persisted successfully")
            print(f"✓ Database location: {os.path.abspath(db_path)}")
            print(f"✓ Total chunks embedded: {len(chunked_documents)}")
            
            return vector_store
    
    except ValueError as e:
        # Re-raise ValueErrors (validation errors)
        raise e
    except Exception as e:
        raise Exception(f"❌ Error with vector store operation: {str(e)}")


def retrieve_similar_documents(vector_store, query, k=3):
    """
    Retrieve the top k most similar documents from the vector store
    based on semantic similarity with the query.
    
    Args:
        vector_store (Chroma): The Chroma vector store instance
        query (str): The search query in natural language
        k (int): Number of top results to retrieve (default: 3)
        
    Returns:
        list: List of top-k Document objects sorted by similarity
        
    Raises:
        ValueError: If query is empty or invalid
        Exception: If similarity search fails
    """
    
    # Validate query
    if not query or not isinstance(query, str):
        raise ValueError("❌ Query must be a non-empty string")
    
    try:
        # Perform similarity search on the vector store
        # Chroma embeds the query using the same embedding function and finds nearest neighbors
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            print(f"⚠ No similar documents found for query: '{query}'")
        
        return results
    
    except Exception as e:
        raise Exception(f"❌ Error during retrieval: {str(e)}")


def print_retrieval_results(query, results):
    """
    Pretty-print the retrieved documents with formatting and metadata.
    
    Args:
        query (str): The original search query
        results (list): List of retrieved Document objects
    """
    print("\n" + "="*80)
    print(f"📋 RETRIEVAL RESULTS")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Retrieved: {len(results)} document(s)\n")
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content:\n{doc.page_content[:500]}")
        if len(doc.page_content) > 500:
            print("...")
        
        # Print metadata if available
        if doc.metadata:
            print(f"\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
    
    print("\n" + "="*80 + "\n")