"""
Database Module for Research Workbench
Handles storage and retrieval of research notes with text content.
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def get_db_connection():
    """Get database connection."""
    db_path = Path(__file__).parent / "research_notes.db"
    return sqlite3.connect(str(db_path))


def initialize_database():
    """Create the research_notes table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop table if it exists with old schema
    cursor.execute('DROP TABLE IF EXISTS research_notes')

    cursor.execute('''
        CREATE TABLE research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def save_note(title, content):
    """
    Save a research note to the database.

    Args:
        title (str): Title of the note
        content (str): Text content of the note

    Returns:
        int: ID of the saved note
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO research_notes (title, content)
        VALUES (?, ?)
    ''', (title, content))

    note_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return note_id


def load_all_notes():
    """
    Load all research notes from the database.

    Returns:
        list: List of note dictionaries with keys: id, title, canvas_json, image_data, timestamp
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT id, title, content, timestamp FROM research_notes ORDER BY timestamp DESC')
    rows = cursor.fetchall()

    notes = []
    for row in rows:
        note = {
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'timestamp': row[3]
        }
        notes.append(note)

    conn.close()
    return notes


# Initialize database on import
initialize_database()