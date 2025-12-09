"""Indexing logic - scan repo and update embeddings."""
from pathlib import Path

from .db import (
    get_connection, get_or_create_repo, get_file_mtime,
    delete_file_chunks, insert_chunk, get_repo_files
)
from .chunker import chunk_file
from .git import get_tracked_files
from .embedder import Embedder


def index_repo(repo_root: Path, embedder: Embedder, force: bool = False) -> dict:
    """Index or update a repository.
    
    Returns stats about what was indexed.
    """
    conn = get_connection()
    repo_id = get_or_create_repo(conn, str(repo_root))
    
    tracked_files = get_tracked_files(repo_root)
    indexed_files = get_repo_files(conn, repo_id)
    
    stats = {
        "files_scanned": 0,
        "files_updated": 0,
        "chunks_added": 0,
        "files_removed": 0,
    }
    
    # Find files to remove (no longer tracked)
    current_files = {str(f) for f in tracked_files}
    for old_file in indexed_files - current_files:
        delete_file_chunks(conn, repo_id, old_file)
        stats["files_removed"] += 1
    
    # Process each tracked file
    files_to_embed = []
    chunks_to_embed = []
    
    for rel_path in tracked_files:
        stats["files_scanned"] += 1
        abs_path = repo_root / rel_path
        
        if not abs_path.exists():
            continue
        
        current_mtime = abs_path.stat().st_mtime
        cached_mtime = get_file_mtime(conn, repo_id, str(rel_path))
        
        # Skip if unchanged (unless force reindex)
        if not force and cached_mtime is not None and cached_mtime >= current_mtime:
            continue
        
        # Read and chunk the file
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        chunks = chunk_file(rel_path, content)
        if not chunks:
            continue
        
        # Queue for embedding
        for chunk in chunks:
            files_to_embed.append((rel_path, current_mtime, chunk))
            chunks_to_embed.append(chunk.text)
    
    if not chunks_to_embed:
        conn.commit()
        conn.close()
        return stats
    
    # Batch embed all chunks
    print(f"Embedding {len(chunks_to_embed)} chunks...")
    embeddings = embedder.embed_documents(chunks_to_embed)
    
    # Store in database
    processed_files = set()
    for i, (rel_path, mtime, chunk) in enumerate(files_to_embed):
        # Delete old chunks for this file (only once per file)
        if str(rel_path) not in processed_files:
            delete_file_chunks(conn, repo_id, str(rel_path))
            processed_files.add(str(rel_path))
            stats["files_updated"] += 1
        
        insert_chunk(
            conn, repo_id, str(rel_path), mtime,
            chunk.start_line, chunk.end_line,
            chunk.text, embeddings[i]
        )
        stats["chunks_added"] += 1
    
    conn.commit()
    conn.close()
    
    return stats