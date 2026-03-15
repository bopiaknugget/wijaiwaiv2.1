"""
RAG Pipeline — CLI Entry Point
Production-ready CLI for document ingestion and semantic retrieval.

Modes:
  --ingest PDF_PATH   Load a PDF and store embeddings in ChromaDB
  --query  QUERY_TEXT Retrieve documents and generate an AI answer

Optional:
  --db  DB_PATH  ChromaDB directory (default: ./Database/chroma_db)
  --k   K        Top-k results for retrieval (default: 3)
"""

import os
import sys
import argparse

from document_loader import load_pdf_document, chunk_documents
from vector_store import (
    initialize_embeddings,
    get_or_create_vector_store,
    retrieve_mmr,
    print_retrieval_results,
)
from generator import generate_answer, print_generated_answer


# ============================================================================
# INGEST MODE
# ============================================================================

def ingest_mode(pdf_path: str, db_path: str = "./Database/chroma_db"):
    """
    Load a PDF, chunk it, embed, and store in ChromaDB.
    If the DB already exists, new chunks are appended (no re-embedding of old data).
    """
    print("\n" + "=" * 80)
    print("🚀 RAG PIPELINE — INGEST MODE")
    print("=" * 80 + "\n")

    try:
        print("[1/3] Loading and chunking PDF...")
        documents = load_pdf_document(pdf_path)
        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        print("\n[2/3] Initializing embeddings model...")
        embeddings = initialize_embeddings()

        print("\n[3/3] Creating/updating vector store...")
        get_or_create_vector_store(
            db_path=db_path,
            chunked_documents=chunks,
            embeddings=embeddings
        )

        print(f"\n✅ Ingest complete. Vector store ready at: {os.path.abspath(db_path)}\n")

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

def query_mode(query_text: str, db_path: str = "./Database/chroma_db", k: int = 3):
    """
    Load an existing ChromaDB and run MMR retrieval + AI answer generation.
    Uses Maximal Marginal Relevance to return diverse, high-quality context.
    """
    print("\n" + "=" * 80)
    print("🔍 RAG PIPELINE — QUERY MODE")
    print("=" * 80 + "\n")

    try:
        print("[1/3] Initializing embeddings model...")
        embeddings = initialize_embeddings()

        print("\n[2/3] Loading vector store and retrieving documents...")
        vector_store = get_or_create_vector_store(
            db_path=db_path,
            chunked_documents=None,   # Load existing DB only
            embeddings=embeddings
        )

        print(f"\n⏳ MMR search for: '{query_text}'")
        retrieved_docs = retrieve_mmr(
            vector_store, query_text,
            k=k, fetch_k=k * 4, lambda_mult=0.6
        )
        print_retrieval_results(query_text, retrieved_docs)

        print("\n[3/3] Generating AI answer...")
        _action, answer, _new_editor, input_tokens, output_tokens = generate_answer(
            query_text, retrieved_docs, chat_history=[]
        )
        print_generated_answer(query_text, answer)
        print(f"📊 Tokens used — input: {input_tokens}, output: {output_tokens}\n")
        print("✅ Query completed successfully!\n")

    except ValueError as e:
        print(f"\n{e}")
        print("💡 Hint: Run ingestion first:\n   python main.py --ingest your_document.pdf")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🤖 RAG Pipeline CLI — Document Ingestion & Semantic Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Ingest a PDF:
    python main.py --ingest data/paper.pdf
    python main.py --ingest documents/paper.pdf --db ./my_db

  Query the vector store:
    python main.py --query "What is the main finding?"
    python main.py --query "What is consciousness?" --db ./my_db --k 5
        """
    )

    parser.add_argument(
        "--ingest",
        type=str, metavar="PDF_PATH",
        help="Ingest mode: chunk and embed a PDF into ChromaDB"
    )
    parser.add_argument(
        "--query",
        type=str, metavar="QUERY_TEXT",
        help="Query mode: retrieve documents and generate an answer"
    )
    parser.add_argument(
        "--db",
        type=str, default="./Database/chroma_db", metavar="DB_PATH",
        help="ChromaDB storage directory (default: ./Database/chroma_db)"
    )
    parser.add_argument(
        "--k",
        type=int, default=3, metavar="K",
        help="Number of results to retrieve (default: 3)"
    )

    return parser


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(1)

    if args.ingest and args.query:
        print("❌ Error: --ingest and --query are mutually exclusive.")
        sys.exit(1)

    if args.ingest:
        ingest_mode(args.ingest, db_path=args.db)
    else:
        query_mode(args.query, db_path=args.db, k=args.k)


if __name__ == "__main__":
    main()
