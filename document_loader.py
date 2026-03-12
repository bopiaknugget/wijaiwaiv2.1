"""
Document Loader Module
Handles PDF, TXT, DOCX/DOC loading and text chunking for the RAG pipeline.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path):
    """
    Load a document based on its file extension.
    Supports: .pdf, .txt, .docx, .doc

    Args:
        file_path (str): Path to the document file

    Returns:
        list: List of Document objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is unsupported or file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except Exception:
                loader = TextLoader(file_path, encoding="cp874")  # Thai Windows encoding fallback
        elif ext in (".docx", ".doc"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"❌ Unsupported file type: {ext}. Supported: .pdf, .txt, .docx, .doc")

        documents = loader.load()

        if not documents:
            raise ValueError(f"❌ File is empty or could not be parsed: {file_path}")

        print(f"✓ Loaded {ext} file: {os.path.basename(file_path)} ({len(documents)} page(s)/section(s))")
        return documents

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise Exception(f"❌ Error loading {os.path.basename(file_path)}: {str(e)}")


def load_pdf_document(pdf_path):
    """
    Load a PDF document using PyPDFLoader (fixed import from langchain_community).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF loading fails
    """
    # Validate that the PDF file exists before attempting to load
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"❌ PDF file not found at: {pdf_path}\n"
            f"Please ensure the file exists in your working directory."
        )
    
    try:
        # Initialize PyPDFLoader with the PDF path
        # Using the correct import: langchain_community.document_loaders
        loader = PyPDFLoader(pdf_path)
        
        # Load all pages as Document objects
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"❌ PDF '{pdf_path}' is empty or could not be parsed")
        
        print(f"✓ Successfully loaded PDF: {pdf_path}")
        print(f"✓ Total pages loaded: {len(documents)}")
        
        return documents
    
    except Exception as e:
        raise Exception(f"❌ Error loading PDF: {str(e)}")


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into overlapping chunks for better embedding context.
    Uses RecursiveCharacterTextSplitter to maintain semantic coherence.
    
    Args:
        documents (list): List of Document objects from the loader
        chunk_size (int): Maximum characters per chunk (default: 1000)
        chunk_overlap (int): Characters to overlap between chunks (default: 200)
        
    Returns:
        list: List of chunked Document objects
        
    Raises:
        ValueError: If documents list is empty
        Exception: If chunking fails
    """
    # Validate input documents
    if not documents:
        raise ValueError("❌ No documents provided for chunking")
    
    try:
        # Initialize the text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,           # Maximum chunk size in characters
            chunk_overlap=chunk_overlap,     # Overlap for context continuity
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs, then sentences, then words
        )
        
        # Split documents into chunks
        # This preserves metadata from original documents
        chunked_documents = text_splitter.split_documents(documents)
        
        if not chunked_documents:
            raise ValueError("❌ Chunking resulted in no documents")
        
        print(f"✓ Document chunking completed")
        print(f"✓ Total chunks created: {len(chunked_documents)}")
        print(f"✓ Chunk size: {chunk_size} chars | Overlap: {chunk_overlap} chars")
        
        return chunked_documents
    
    except Exception as e:
        raise Exception(f"❌ Error chunking documents: {str(e)}")
