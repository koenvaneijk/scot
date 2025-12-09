"""Search functionality."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .db import get_connection, get_or_create_repo, get_repo_chunks
from .embedder import Embedder, cosine_similarity_matrix
from .indexer import index_repo
from .bm25 import BM25Index, reciprocal_rank_fusion


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
    mode: str = "hybrid",  # "hybrid", "semantic", "lexical"
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
    
    # Number of candidates to fetch from each method before fusion
    fetch_k = min(top_k * 3, len(chunks))
    
    rankings = []
    
    # Semantic search
    if mode in ("hybrid", "semantic"):
        query_embedding = embedder.embed_query(query)
        doc_embeddings = np.stack([c["embedding"] for c in chunks])
        similarities = cosine_similarity_matrix(query_embedding, doc_embeddings)
        
        # Get top indices with scores
        top_indices = np.argsort(similarities)[::-1][:fetch_k]
        semantic_ranking = [(int(idx), float(similarities[idx])) for idx in top_indices]
        rankings.append(semantic_ranking)
    
    # Lexical search (BM25)
    if mode in ("hybrid", "lexical"):
        bm25 = BM25Index()
        bm25.index([c["chunk_text"] for c in chunks])
        lexical_ranking = bm25.search(query, top_k=fetch_k)
        rankings.append(lexical_ranking)
    
    # Combine results
    if mode == "hybrid":
        fused = reciprocal_rank_fusion(rankings)
        top_indices = [idx for idx, _ in fused[:top_k]]
        scores = {idx: score for idx, score in fused}
    elif mode == "semantic":
        top_indices = [idx for idx, _ in rankings[0][:top_k]]
        scores = {idx: score for idx, score in rankings[0]}
    else:  # lexical
        top_indices = [idx for idx, _ in rankings[0][:top_k]]
        scores = {idx: score for idx, score in rankings[0]}
    
    # Build results
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(SearchResult(
            file_path=chunk["file_path"],
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            score=scores[idx],
            chunk_text=chunk["chunk_text"],
        ))
    
    return results