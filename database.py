"""
Database Module for Research Workbench
Handles storage and retrieval of research notes and document metadata.
Uses context managers for all connections to prevent locks and leaks.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).parent / "research_notes.db"


@contextmanager
def get_db_connection():
    """
    Context manager for SQLite connections.
    Guarantees the connection is closed even if an exception is raised.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            ...
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()


def initialize_database():
    """
    Create tables if they don't exist. Safe to call on every startup.
    Does NOT drop existing tables — data is fully persistent.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Research notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_notes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                title     TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Uploaded document metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                file_type   TEXT,
                chunk_count INTEGER DEFAULT 0,
                db_path     TEXT,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()


# ── Research Notes ────────────────────────────────────────────────────────────

def save_note(title: str, content: str) -> int:
    """Save a research note. Returns the new note ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO research_notes (title, content) VALUES (?, ?)',
            (title, content)
        )
        conn.commit()
        return cursor.lastrowid


def load_all_notes() -> list:
    """Load all research notes ordered newest-first."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, title, content, timestamp FROM research_notes ORDER BY timestamp DESC'
        )
        return [
            {'id': row[0], 'title': row[1], 'content': row[2], 'timestamp': row[3]}
            for row in cursor.fetchall()
        ]


def delete_note_by_id(note_id: int) -> bool:
    """Delete a research note by ID. Returns True if a row was deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM research_notes WHERE id = ?', (note_id,))
        conn.commit()
        return cursor.rowcount > 0


# ── Document Metadata ─────────────────────────────────────────────────────────

def save_document_metadata(filename: str, file_type: str,
                            chunk_count: int, db_path: str) -> int:
    """Save uploaded document metadata. Returns the new document ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO documents (filename, file_type, chunk_count, db_path) VALUES (?, ?, ?, ?)',
            (filename, file_type, chunk_count, db_path)
        )
        conn.commit()
        return cursor.lastrowid


def load_all_documents() -> list:
    """Load all document metadata ordered newest-first."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, filename, file_type, chunk_count, db_path, timestamp '
            'FROM documents ORDER BY timestamp DESC'
        )
        return [
            {
                'id': row[0], 'filename': row[1], 'file_type': row[2],
                'chunk_count': row[3], 'db_path': row[4], 'timestamp': row[5]
            }
            for row in cursor.fetchall()
        ]


def delete_document_by_id(doc_id: int) -> bool:
    """Delete a document metadata record by ID. Returns True if a row was deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        conn.commit()
        return cursor.rowcount > 0


# Initialize database on import (idempotent — CREATE TABLE IF NOT EXISTS)
initialize_database()
