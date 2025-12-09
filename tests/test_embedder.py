"""Tests for embedder (mocked to avoid loading model in tests)."""
import numpy as np
import pytest

from scot.embedder import cosine_similarity, cosine_similarity_matrix


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""
    
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(a, a)
        assert np.isclose(sim, 1.0)
    
    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 0.0)
    
    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim, -1.0)
    
    def test_similar_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = cosine_similarity(a, b)
        assert sim > 0.99


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix function."""
    
    def test_single_document(self):
        query = np.array([1.0, 0.0, 0.0])
        docs = np.array([[1.0, 0.0, 0.0]])
        sims = cosine_similarity_matrix(query, docs)
        assert sims.shape == (1,)
        assert np.isclose(sims[0], 1.0)
    
    def test_multiple_documents(self):
        query = np.array([1.0, 0.0])
        docs = np.array([
            [1.0, 0.0],  # Same as query
            [0.0, 1.0],  # Orthogonal
            [0.7071, 0.7071],  # 45 degrees
        ])
        sims = cosine_similarity_matrix(query, docs)
        assert sims.shape == (3,)
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert np.isclose(sims[2], 0.7071, atol=0.01)
    
    def test_returns_correct_order(self):
        query = np.array([1.0, 0.0])
        docs = np.array([
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        sims = cosine_similarity_matrix(query, docs)
        # Second doc should have highest similarity
        assert np.argmax(sims) == 1
    
    def test_handles_zero_vectors(self):
        query = np.array([1.0, 0.0])
        docs = np.array([
            [0.0, 0.0],  # Zero vector
            [1.0, 0.0],
        ])
        # Should not raise, should handle gracefully
        sims = cosine_similarity_matrix(query, docs)
        assert sims.shape == (2,)