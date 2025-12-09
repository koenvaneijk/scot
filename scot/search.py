"""Search functionality."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .db import get_connection, get_or_create_repo, get_repo_chunks
from .embedder import Embedder, cosine_similarity_matrix
from .indexer import index_repo


@dataclass
class SearchResult:
    """A single search result."""
    file_path: str
    start_line: int
    end_line: int
    score: float
    chunk_text: str


def search(
    repo_root: Path,
    query: str,
    embedder: Embedder,
    top_k: int = 5,
    file_pattern: str = None,
) -> list[SearchResult]:
    """Search for chunks matching the query.
    
    Automatically indexes/updates the repo if needed.
    """
    # Ensure repo is indexed
    stats = index_repo(repo_root, embedder)
    if stats["files_updated"] > 0:
        print(f"Indexed {stats['files_updated']} files, {stats['chunks_added']} chunks")
    
    # Get all chunks for this repo
    conn = get_connection()
    repo_id = get_or_create_repo(conn, str(repo_root))
    chunks = get_repo_chunks(conn, repo_id)
    conn.close()
    
    if not chunks:
        return []
    
    # Filter by file pattern if specified
    if file_pattern:
        import fnmatch
        chunks = [c for c in chunks if fnmatch.fnmatch(c["file_path"], file_pattern)]
    
    if not chunks:
        return []
    
    # Embed query
    query_embedding = embedder.embed_query(query)
    
    # Build matrix of document embeddings
    doc_embeddings = np.stack([c["embedding"] for c in chunks])
    
    # Compute similarities
    similarities = cosine_similarity_matrix(query_embedding, doc_embeddings)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(SearchResult(
            file_path=chunk["file_path"],
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            score=float(similarities[idx]),
            chunk_text=chunk["chunk_text"],
        ))
    
    return results