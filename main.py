"""
RAG Pipeline - Main Entry Point
Production-ready CLI for managing document ingestion and semantic retrieval.
Supports two modes: --ingest (load PDFs) and --query (retrieve documents).
"""

import os
import sys
import argparse

from document_loader import load_pdf_document, chunk_documents
from vector_store import initialize_embeddings, get_or_create_vector_store, retrieve_similar_documents, print_retrieval_results
from generator import generate_answer, print_generated_answer


# ============================================================================
# INGEST MODE
# ============================================================================

def ingest_mode(pdf_path, db_path="./chroma_db"):
    """
    Ingest Mode: Load PDF, chunk it, and store in ChromaDB.
    
    Workflow:
    1. Load PDF document
    2. Chunk text into overlapping segments
    3. Initialize embeddings model
    4. Create or update vector store
    
    Args:
        pdf_path (str): Path to the PDF file
        db_path (str): Path for ChromaDB storage (default: ./chroma_db)
    """
    
    print("\n" + "="*80)
    print("🚀 RAG PIPELINE - INGEST MODE")
    print("="*80 + "\n")
    
    try:
        # Step 1: Load and chunk PDF document
        print("[1/3] Loading and chunking PDF document...")
        documents = load_pdf_document(pdf_path)
        chunked_documents = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # Step 2: Initialize embeddings
        print("\n[2/3] Initializing embeddings model...")
        embeddings = initialize_embeddings()
        
        # Step 3: Create/update vector store
        # CRITICAL FIX: If DB exists, it's loaded. If not, it's created from the PDF.
        print("\n[3/3] Creating/updating vector store...")
        vector_store = get_or_create_vector_store(
            db_path=db_path,
            chunked_documents=chunked_documents,
            embeddings=embeddings
        )
        
        print("\n✅ RAG Pipeline - Ingest completed successfully!")
        print(f"📌 Vector store is ready at: {os.path.abspath(db_path)}\n")
        
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
# QUERY MODE
# ============================================================================

def query_mode(query_text, db_path="./chroma_db", k=3):
    """
    Query Mode: Load existing ChromaDB and retrieve similar documents.
    
    Workflow:
    1. Load environment variables
    2. Initialize embeddings model
    3. Load existing vector store from disk (no re-embedding)
    4. Perform similarity search
    5. Display results
    
    Args:
        query_text (str): The retrieval query
        db_path (str): Path to ChromaDB storage (default: ./chroma_db)
        k (int): Number of top results to retrieve (default: 3)
    """
    
    print("\n" + "="*80)
    print("🔍 RAG PIPELINE - QUERY MODE")
    print("="*80 + "\n")
    
    try:
        # Step 1: Initialize embeddings
        print("[1/3] Initializing embeddings model...")
        embeddings = initialize_embeddings()
        
        # Step 2: Load existing vector store (without documents = faster)
        # CRITICAL FIX: Load from disk, don't recreate
        print("\n[2/3] Loading vector store and retrieving documents...")
        vector_store = get_or_create_vector_store(
            db_path=db_path,
            chunked_documents=None,  # None = load existing DB only
            embeddings=embeddings
        )
        
        # Perform retrieval
        print(f"\n⏳ Searching for documents similar to: '{query_text}'")
        retrieved_docs = retrieve_similar_documents(vector_store, query_text, k=k)
        
        # Display results
        print_retrieval_results(query_text, retrieved_docs)
        
        # Step 3: Generate AI answer using Gemini
        print("\n[3/3] Generating AI answer...")
        answer = generate_answer(query_text, retrieved_docs)
        
        # Display the generated answer
        print_generated_answer(query_text, answer)
        
        print("✅ Query and generation completed successfully!\n")
        
    except ValueError as e:
        print(f"\n{e}")
        print(f"💡 Hint: Have you ingested a PDF yet? Run:\n")
        print(f"   python main.py --ingest your_document.pdf")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def create_parser():
    """
    Create and configure the argument parser for CLI.
    
    Returns:
        ArgumentParser: Configured parser with subcommands
    """
    parser = argparse.ArgumentParser(
        description="🤖 RAG Pipeline CLI - Document Ingestion & Semantic Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Ingest a PDF:
    python main.py --ingest hesse_philosophy_analysis.pdf
    python main.py --ingest documents/paper.pdf --db ./my_db

  Query the vector store:
    python main.py --query "What is consciousness?"
    python main.py --query "What is consciousness?" --db ./my_db --k 5

  Show help:
    python main.py --help
        """
    )
    
   # Subcommands with mutual exclusivity
    parser.add_argument(
        "--ingest",
        type=str,
        metavar="PDF_PATH",
        help="Ingest mode: Load a PDF, chunk it, and store embeddings in ChromaDB"
    )

    
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUERY_TEXT",
        help="Query mode: Retrieve documents similar to the query from existing ChromaDB"
    )
    
    # Optional parameters
    parser.add_argument(
        "--db",
        type=str,
        default="./chroma_db",
        metavar="DB_PATH",
        help="Path to ChromaDB storage directory (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        metavar="K",
        help="Number of top results to retrieve (default: 3)"
    )
    
    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the RAG Pipeline CLI.
    Parses arguments and routes to appropriate mode (ingest or query).
    """
    
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate that either --ingest or --query is provided
    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(1)
    
    # Ensure mutual exclusivity (can't do both ingest and query at once)
    if args.ingest and args.query:
        print("❌ Error: Cannot use both --ingest and --query in the same command")
        sys.exit(1)
    
    # Route to appropriate mode
    if args.ingest:
        ingest_mode(args.ingest, db_path=args.db)
    
    elif args.query:
        query_mode(args.query, db_path=args.db, k=args.k)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
