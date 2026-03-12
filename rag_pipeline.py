"""
RAG Pipeline Phase 1: Document Ingestion, Embedding, and Retrieval
This module handles loading PDFs, chunking text, creating embeddings,
and storing them in a local ChromaDB vector store.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================

def load_environment():
    """
    Load environment variables from .env file.
    Ensures GOOGLE_API_KEY is available for Gemini API access.
    
    Returns:
        str: The Google API key
        
    Raises:
        ValueError: If GOOGLE_API_KEY is not found in environment
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Retrieve the Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Error handling: Check if API key exists
    if not api_key:
        raise ValueError(
            "❌ GOOGLE_API_KEY not found. Please ensure:\n"
            "  1. You have a .env file in the current directory\n"
            "  2. GOOGLE_API_KEY=your_key_here is set in the .env file\n"
            "  3. You've obtained your API key from: https://ai.google.dev/"
        )
    
    print("✓ Environment loaded successfully")
    return api_key


# ============================================================================
# 2. DOCUMENT LOADING
# ============================================================================

def load_pdf_document(pdf_path):
    """
    Load a PDF document using PyPDFLoader.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF loading fails
    """
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"❌ PDF file not found at: {pdf_path}\n"
            f"Please ensure the file exists in your working directory."
        )
    
    try:
        # Initialize PyPDFLoader with the PDF path
        loader = PyPDFLoader(pdf_path)
        
        # Load all pages as documents
        documents = loader.load()
        
        print(f"✓ Successfully loaded PDF: {pdf_path}")
        print(f"✓ Total pages loaded: {len(documents)}")
        
        return documents
    
    except Exception as e:
        raise Exception(f"❌ Error loading PDF: {str(e)}")


# ============================================================================
# 3. TEXT CHUNKING
# ============================================================================

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into overlapping chunks for better embedding context.
    Uses RecursiveCharacterTextSplitter to maintain semantic coherence.
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Maximum characters per chunk (default: 1000)
        chunk_overlap (int): Characters to overlap between chunks (default: 200)
        
    Returns:
        list: List of chunked Document objects
    """
    # Initialize the text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,           # Maximum chunk size
        chunk_overlap=chunk_overlap,     # Overlap for context continuity
        separators=["\n\n", "\n", " ", ""]  # Split on sentences first, then words
    )
    
    # Split documents into chunks
    chunked_documents = text_splitter.split_documents(documents)
    
    print(f"✓ Document chunking completed")
    print(f"✓ Total chunks created: {len(chunked_documents)}")
    print(f"✓ Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    
    return chunked_documents


# ============================================================================
# 4. EMBEDDING SETUP
# ============================================================================

def initialize_embeddings():
    """
    Initialize Google Generative AI embeddings using Gemini's text-embedding-004 model.
    This creates embeddings for both documents and queries.
    
    Returns:
        GoogleGenerativeAIEmbeddings: Initialized embedding model
        
    Raises:
        ValueError: If API key is invalid or embedding initialization fails
    """
    try:
        # Initialize GoogleGenerativeAIEmbeddings with text-embedding-004 model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  # Latest Google Gemini embedding model
        )
        
        print("✓ GoogleGenerativeAIEmbeddings initialized (text-embedding-004)")
        
        return embeddings
    
    except Exception as e:
        raise ValueError(f"❌ Error initializing embeddings: {str(e)}")


# ============================================================================
# 5. VECTOR STORE SETUP
# ============================================================================

def create_vector_store(chunked_documents, embeddings, db_path="./chroma_db"):
    """
    Create and persist a ChromaDB vector store with embedded documents.
    This allows for efficient similarity-based retrieval later.
    
    Args:
        chunked_documents (list): List of chunked Document objects
        embeddings: Initialized embedding model
        db_path (str): Path where ChromaDB will be persisted (default: ./chroma_db)
        
    Returns:
        Chroma: Initialized and populated Chroma vector store
        
    Raises:
        Exception: If vector store creation or persistence fails
    """
    try:
        # Create Chroma vector store with embeddings and documents
        # persist_directory ensures the DB is saved locally for future use
        vector_store = Chroma.from_documents(
            documents=chunked_documents,        # Chunked documents to embed and store
            embedding=embeddings,               # Embedding function (Gemini)
            persist_directory=db_path,          # Local storage path
            collection_name="documents"         # Collection name for organization
        )
        
        # Persist the vector store to disk
        vector_store.persist()
        
        print(f"✓ ChromaDB vector store created successfully")
        print(f"✓ Database persisted at: {os.path.abspath(db_path)}")
        print(f"✓ Total documents in store: {len(chunked_documents)}")
        
        return vector_store
    
    except Exception as e:
        raise Exception(f"❌ Error creating vector store: {str(e)}")


# ============================================================================
# 6. RETRIEVAL TESTING
# ============================================================================

def retrieve_similar_documents(vector_store, query, k=3):
    """
    Retrieve the top k most similar documents from the vector store
    based on a similarity search with the query.
    
    Args:
        vector_store (Chroma): The Chroma vector store
        query (str): The search query
        k (int): Number of top results to retrieve (default: 3)
        
    Returns:
        list: List of retrieved Document objects
    """
    try:
        # Perform similarity search on the vector store
        results = vector_store.similarity_search(query, k=k)
        
        return results
    
    except Exception as e:
        raise Exception(f"❌ Error during retrieval: {str(e)}")


def print_retrieval_results(query, results):
    """
    Pretty-print the retrieved documents with formatting.
    
    Args:
        query (str): The original search query
        results (list): List of retrieved Document objects
    """
    print("\n" + "="*80)
    print(f"📋 RETRIEVAL TEST RESULTS")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Retrieved: {len(results)} documents\n")
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content:\n{doc.page_content[:500]}...")  # Print first 500 chars
        if "source" in doc.metadata:
            print(f"Source: {doc.metadata['source']}")
        if "page" in doc.metadata:
            print(f"Page: {doc.metadata['page']}")
    
    print("\n" + "="*80)


# ============================================================================
# 7. MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """
    Main function that orchestrates the entire RAG pipeline.
    Executes all steps: environment setup, document loading,
    chunking, embedding, vector store creation, and retrieval testing.
    """
    
    print("\n" + "="*80)
    print("🚀 RAG PIPELINE - PHASE 1: INGESTION & RETRIEVAL")
    print("="*80 + "\n")
    
    try:
        # Step 1: Load environment variables and API key
        print("[1/6] Loading environment configuration...")
        api_key = load_environment()
        
        # Step 2: Load PDF document
        print("\n[2/6] Loading PDF document...")
        pdf_path = "hesse_philosophy_analysis.pdf"  # Sample PDF file name
        documents = load_pdf_document(pdf_path)
        
        # Step 3: Chunk documents for better semantic processing
        print("\n[3/6] Chunking documents...")
        chunked_documents = chunk_documents(
            documents,
            chunk_size=1000,        # Maximum characters per chunk
            chunk_overlap=200       # Overlap for context continuity
        )
        
        # Step 4: Initialize embedding model
        print("\n[4/6] Initializing embeddings...")
        embeddings = initialize_embeddings()
        
        # Step 5: Create and persist vector store
        print("\n[5/6] Creating and persisting vector store...")
        vector_store = create_vector_store(chunked_documents, embeddings)
        
        # Step 6: Test retrieval with sample query
        print("\n[6/6] Testing retrieval system...")
        
        # Define a hardcoded test query
        test_query = "What is the philosophical significance of consciousness?"
        
        # Retrieve similar documents
        retrieved_docs = retrieve_similar_documents(vector_store, test_query, k=3)
        
        # Display results
        print_retrieval_results(test_query, retrieved_docs)
        
        print("\n✅ RAG Pipeline Phase 1 completed successfully!")
        print("📌 Vector store is now ready for production use.\n")
        
        return vector_store
    
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main pipeline
    vector_store = main()
    
    # The vector_store is now ready for use in subsequent phases
    # (e.g., Phase 2: LLM integration for Q&A)
