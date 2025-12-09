"""Embedding model wrapper using EmbeddingGemma."""
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import MODEL_NAME


class Embedder:
    """Wrapper for EmbeddingGemma embedding model."""
    
    def __init__(self):
        self.model = None
    
    def load(self):
        """Load the model into memory."""
        if self.model is None:
            print(f"Loading model {MODEL_NAME}...")
            try:
                self.model = SentenceTransformer(MODEL_NAME)
            except OSError as e:
                error_msg = str(e)
                if "401" in error_msg or "Access" in error_msg or "gated" in error_msg.lower():
                    raise RuntimeError(
                        f"Access denied to {MODEL_NAME}\n\n"
                        "To fix:\n"
                        f"  1. Visit https://huggingface.co/{MODEL_NAME}\n"
                        "  2. Click 'Acknowledge license' to accept the terms\n"
                        "  3. Run: huggingface-cli login"
                    ) from e
                raise
            print("Model loaded.")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        self.load()
        # Use code retrieval prompt for queries
        embedding = self.model.encode(
            query,
            prompt_name="query",
        )
        return embedding.astype(np.float32)
    
    def embed_document(self, text: str) -> np.ndarray:
        """Embed a code/document chunk."""
        self.load()
        embedding = self.model.encode(
            text,
            prompt_name="document",
        )
        return embedding.astype(np.float32)
    
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed multiple documents efficiently."""
        self.load()
        embeddings = self.model.encode(
            texts,
            prompt_name="document",
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and multiple documents."""
    # Normalize with epsilon to avoid division by zero
    eps = 1e-10
    query_norm_val = np.linalg.norm(query)
    query_norm = query / (query_norm_val + eps) if query_norm_val > eps else query
    
    doc_norm_vals = np.linalg.norm(documents, axis=1, keepdims=True)
    doc_norms = documents / np.where(doc_norm_vals > eps, doc_norm_vals, 1.0)
    
    return np.dot(doc_norms, query_norm)