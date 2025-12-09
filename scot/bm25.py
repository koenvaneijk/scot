"""Simple BM25 implementation - no external dependencies."""
import math
import re
from collections import Counter
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    """Simple tokenizer - split on non-alphanumeric, lowercase."""
    # Split camelCase and snake_case
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    # Extract alphanumeric tokens
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


@dataclass
class BM25Index:
    """BM25 index for a collection of documents."""
    # Parameters
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization
    
    # Index data
    doc_freqs: dict[str, int] = None  # term -> num docs containing term
    doc_lens: list[int] = None  # Length of each doc
    avg_doc_len: float = 0.0
    num_docs: int = 0
    doc_term_freqs: list[dict[str, int]] = None  # term freqs per doc
    
    def __post_init__(self):
        self.doc_freqs = {}
        self.doc_lens = []
        self.doc_term_freqs = []
    
    def index(self, documents: list[str]):
        """Build index from documents."""
        self.doc_freqs = {}
        self.doc_lens = []
        self.doc_term_freqs = []
        self.num_docs = len(documents)
        
        for doc in documents:
            tokens = tokenize(doc)
            self.doc_lens.append(len(tokens))
            
            # Term frequencies for this doc
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))
            
            # Update document frequencies
            for term in term_freqs:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        self.avg_doc_len = sum(self.doc_lens) / max(self.num_docs, 1)
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search and return list of (doc_index, score) tuples."""
        query_tokens = tokenize(query)
        scores = []
        
        for doc_idx in range(self.num_docs):
            score = self._score_doc(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            
            # Term frequency in this doc
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            
            # Document frequency
            df = self.doc_freqs[term]
            
            # IDF component
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
            
            # TF component with saturation and length normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )
            
            score += idf * tf_norm
        
        return score


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple rankings using RRF.
    
    Args:
        rankings: List of rankings, each is [(doc_idx, score), ...]
        k: RRF constant (default 60)
    
    Returns:
        Fused ranking as [(doc_idx, fused_score), ...]
    """
    fused_scores: dict[int, float] = {}
    
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking):
            if doc_idx not in fused_scores:
                fused_scores[doc_idx] = 0.0
            fused_scores[doc_idx] += 1.0 / (k + rank + 1)
    
    # Sort by fused score
    results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return results