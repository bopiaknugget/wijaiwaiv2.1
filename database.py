"""
Database Module for Research Workbench
Handles storage and retrieval of research notes, document metadata,
and parent chunks for Parent-Child Chunking (Advanced RAG).
Uses context managers for all connections to prevent locks and leaks.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).parent / "Database" / "research_notes.db"


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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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

        # Parent chunks table for Parent-Child Chunking (Advanced RAG)
        # Child chunks in Pinecone reference parent_id to retrieve larger context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                source_file TEXT,
                page_number INTEGER,
                section     TEXT,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ตาราง web_pages สำหรับเก็บข้อมูลเว็บที่ scrape มา
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS web_pages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL,
                title       TEXT NOT NULL,
                summary     TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Users table for Google OAuth
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id         TEXT PRIMARY KEY,
                email      TEXT UNIQUE NOT NULL,
                name       TEXT,
                picture    TEXT,
                created_at TEXT,
                last_login TEXT
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


# ── Parent Chunks (Advanced RAG) ─────────────────────────────────────────────

def save_parent_chunk(parent_id: str, content: str, source_file: str = None,
                      page_number: int = None, section: str = None):
    """Save a parent chunk. Uses INSERT OR REPLACE to allow updates."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO parent_chunks (id, content, source_file, page_number, section) '
            'VALUES (?, ?, ?, ?, ?)',
            (parent_id, content, source_file, page_number, section)
        )
        conn.commit()


def save_parent_chunks_batch(records: list):
    """Save multiple parent chunks in a single transaction.

    Args:
        records: list of dicts with keys: id, content, source_file, page_number, section
    """
    if not records:
        return
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            'INSERT OR REPLACE INTO parent_chunks (id, content, source_file, page_number, section) '
            'VALUES (?, ?, ?, ?, ?)',
            [(r['id'], r['content'], r.get('source_file'), r.get('page_number'), r.get('section'))
             for r in records]
        )
        conn.commit()


def get_parent_chunk(parent_id: str) -> dict | None:
    """Retrieve a parent chunk by ID. Returns dict or None."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, content, source_file, page_number, section FROM parent_chunks WHERE id = ?',
            (parent_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0], 'content': row[1], 'source_file': row[2],
                'page_number': row[3], 'section': row[4]
            }
        return None


def get_parent_chunks_batch(parent_ids: list) -> dict:
    """Retrieve multiple parent chunks by IDs. Returns {id: dict}."""
    if not parent_ids:
        return {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in parent_ids)
        cursor.execute(
            f'SELECT id, content, source_file, page_number, section '
            f'FROM parent_chunks WHERE id IN ({placeholders})',
            parent_ids
        )
        return {
            row[0]: {
                'id': row[0], 'content': row[1], 'source_file': row[2],
                'page_number': row[3], 'section': row[4]
            }
            for row in cursor.fetchall()
        }


def delete_parent_chunks_by_source(source_file: str) -> int:
    """Delete all parent chunks for a given source file. Returns count deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM parent_chunks WHERE source_file = ?', (source_file,))
        conn.commit()
        return cursor.rowcount


# ── Web Pages ──────────────────────────────────────────────────────────────────

def save_web_page(url: str, title: str, summary: str, chunk_count: int = 0) -> int:
    """บันทึกข้อมูลเว็บเพจ Returns new web_page ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO web_pages (url, title, summary, chunk_count) VALUES (?, ?, ?, ?)',
            (url, title, summary, chunk_count)
        )
        conn.commit()
        return cursor.lastrowid


def load_all_web_pages() -> list:
    """โหลดรายการเว็บเพจทั้งหมด เรียงจากใหม่ไปเก่า"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, url, title, summary, chunk_count, timestamp '
            'FROM web_pages ORDER BY timestamp DESC'
        )
        return [
            {
                'id': row[0], 'url': row[1], 'title': row[2],
                'summary': row[3], 'chunk_count': row[4], 'timestamp': row[5]
            }
            for row in cursor.fetchall()
        ]


def delete_web_page_by_id(page_id: int) -> bool:
    """ลบเว็บเพจตาม ID Returns True ถ้าลบสำเร็จ"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM web_pages WHERE id = ?', (page_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_web_page_title(page_id: int, new_title: str) -> bool:
    """อัปเดตชื่อ Title ของเว็บเพจ Returns True ถ้าอัปเดตสำเร็จ"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE web_pages SET title = ? WHERE id = ?',
            (new_title, page_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def update_web_page(page_id: int, new_title: str, new_summary: str,
                    chunk_count: int = None) -> bool:
    """อัปเดต Title, Summary (และ chunk_count ถ้าระบุ) ของเว็บเพจ"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if chunk_count is not None:
            cursor.execute(
                'UPDATE web_pages SET title = ?, summary = ?, chunk_count = ? WHERE id = ?',
                (new_title, new_summary, chunk_count, page_id)
            )
        else:
            cursor.execute(
                'UPDATE web_pages SET title = ?, summary = ? WHERE id = ?',
                (new_title, new_summary, page_id)
            )
        conn.commit()
        return cursor.rowcount > 0


def get_web_page_by_id(page_id: int) -> dict | None:
    """ดึงข้อมูลเว็บเพจตาม ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, url, title, summary, chunk_count, timestamp '
            'FROM web_pages WHERE id = ?',
            (page_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0], 'url': row[1], 'title': row[2],
                'summary': row[3], 'chunk_count': row[4], 'timestamp': row[5]
            }
        return None


# ── Users (Google OAuth) ──────────────────────────────────────────────────────

def save_user(user_info: dict) -> None:
    """
    Save or update a Google OAuth user.
    Uses INSERT OR REPLACE to upsert on the primary key (id).

    Args:
        user_info: dict with keys: id, email, name, picture
    """
    from datetime import datetime
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if user already exists to preserve created_at
        cursor.execute('SELECT created_at FROM users WHERE id = ?', (user_info['id'],))
        row = cursor.fetchone()
        created_at = row[0] if row else now

        cursor.execute(
            'INSERT OR REPLACE INTO users (id, email, name, picture, created_at, last_login) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (
                user_info['id'],
                user_info['email'],
                user_info.get('name', ''),
                user_info.get('picture', ''),
                created_at,
                now,
            )
        )
        conn.commit()


def get_user(user_id: str) -> dict | None:
    """Retrieve a user by ID. Returns dict or None."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, email, name, picture, created_at, last_login '
            'FROM users WHERE id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0], 'email': row[1], 'name': row[2],
                'picture': row[3], 'created_at': row[4], 'last_login': row[5],
            }
        return None


# Initialize database on import (idempotent — CREATE TABLE IF NOT EXISTS)
initialize_database()
