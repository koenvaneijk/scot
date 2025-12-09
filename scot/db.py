"""SQLite database for chunk storage and retrieval."""
import sqlite3
from pathlib import Path
from typing import Optional
import numpy as np

from .config import DB_PATH, ensure_scot_dir, EMBEDDING_DIM


def get_connection() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    ensure_scot_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    """Initialize database tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY,
            abs_path TEXT UNIQUE,
            last_indexed REAL
        );
        
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            repo_id INTEGER REFERENCES repos(id),
            file_path TEXT,
            file_mtime REAL,
            start_line INTEGER,
            end_line INTEGER,
            chunk_text TEXT,
            embedding BLOB
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(repo_id, file_path);
    """)
    conn.commit()


def get_or_create_repo(conn: sqlite3.Connection, repo_path: str) -> int:
    """Get repo ID, creating if it doesn't exist."""
    cursor = conn.execute(
        "SELECT id FROM repos WHERE abs_path = ?", (repo_path,)
    )
    row = cursor.fetchone()
    if row:
        return row["id"]
    
    cursor = conn.execute(
        "INSERT INTO repos (abs_path, last_indexed) VALUES (?, 0)",
        (repo_path,)
    )
    conn.commit()
    return cursor.lastrowid


def get_file_mtime(conn: sqlite3.Connection, repo_id: int, file_path: str) -> Optional[float]:
    """Get the cached mtime for a file, or None if not indexed."""
    cursor = conn.execute(
        "SELECT file_mtime FROM chunks WHERE repo_id = ? AND file_path = ? LIMIT 1",
        (repo_id, file_path)
    )
    row = cursor.fetchone()
    return row["file_mtime"] if row else None


def delete_file_chunks(conn: sqlite3.Connection, repo_id: int, file_path: str):
    """Delete all chunks for a file."""
    conn.execute(
        "DELETE FROM chunks WHERE repo_id = ? AND file_path = ?",
        (repo_id, file_path)
    )


def insert_chunk(
    conn: sqlite3.Connection,
    repo_id: int,
    file_path: str,
    file_mtime: float,
    start_line: int,
    end_line: int,
    chunk_text: str,
    embedding: np.ndarray
):
    """Insert a chunk with its embedding."""
    conn.execute(
        """INSERT INTO chunks 
           (repo_id, file_path, file_mtime, start_line, end_line, chunk_text, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (repo_id, file_path, file_mtime, start_line, end_line, chunk_text, embedding.tobytes())
    )


def get_repo_chunks(conn: sqlite3.Connection, repo_id: int) -> list[dict]:
    """Get all chunks for a repo with their embeddings."""
    cursor = conn.execute(
        """SELECT id, file_path, start_line, end_line, chunk_text, embedding
           FROM chunks WHERE repo_id = ?""",
        (repo_id,)
    )
    results = []
    for row in cursor:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        results.append({
            "id": row["id"],
            "file_path": row["file_path"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "chunk_text": row["chunk_text"],
            "embedding": embedding,
        })
    return results


def get_repo_files(conn: sqlite3.Connection, repo_id: int) -> set[str]:
    """Get all indexed file paths for a repo."""
    cursor = conn.execute(
        "SELECT DISTINCT file_path FROM chunks WHERE repo_id = ?",
        (repo_id,)
    )
    return {row["file_path"] for row in cursor}