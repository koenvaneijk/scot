"""Tests for BM25 search."""
import pytest

from scot.bm25 import tokenize, BM25Index, reciprocal_rank_fusion


class TestTokenize:
    """Tests for tokenize function."""
    
    def test_simple_text(self):
        tokens = tokenize("hello world")
        assert tokens == ["hello", "world"]
    
    def test_camel_case(self):
        tokens = tokenize("camelCaseWord")
        assert "camel" in tokens
        assert "case" in tokens
        assert "word" in tokens
    
    def test_snake_case(self):
        tokens = tokenize("snake_case_word")
        assert "snake" in tokens
        assert "case" in tokens
        assert "word" in tokens
    
    def test_mixed_case(self):
        tokens = tokenize("getUserById")
        assert "get" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens
    
    def test_filters_short_tokens(self):
        tokens = tokenize("a b c ab cd")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "ab" in tokens
        assert "cd" in tokens
    
    def test_lowercase(self):
        tokens = tokenize("HELLO World")
        assert tokens == ["hello", "world"]
    
    def test_numbers(self):
        tokens = tokenize("test123 foo456")
        assert "test123" in tokens
        assert "foo456" in tokens
    
    def test_special_characters(self):
        tokens = tokenize("hello@world.com foo-bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "com" in tokens
        assert "foo" in tokens
        assert "bar" in tokens


class TestBM25Index:
    """Tests for BM25Index class."""
    
    def test_index_empty(self):
        bm25 = BM25Index()
        bm25.index([])
        assert bm25.num_docs == 0
    
    def test_index_single_doc(self):
        bm25 = BM25Index()
        bm25.index(["hello world"])
        assert bm25.num_docs == 1
        assert "hello" in bm25.doc_freqs
        assert "world" in bm25.doc_freqs
    
    def test_index_multiple_docs(self):
        bm25 = BM25Index()
        bm25.index(["hello world", "hello python", "world python"])
        assert bm25.num_docs == 3
        assert bm25.doc_freqs["hello"] == 2
        assert bm25.doc_freqs["world"] == 2
        assert bm25.doc_freqs["python"] == 2
    
    def test_search_exact_match(self):
        bm25 = BM25Index()
        bm25.index([
            "python programming language",
            "java programming language",
            "python snake animal",
        ])
        results = bm25.search("python programming", top_k=3)
        # First result should be the doc with both terms
        assert results[0][0] == 0
    
    def test_search_returns_scores(self):
        bm25 = BM25Index()
        bm25.index(["hello world", "foo bar"])
        results = bm25.search("hello", top_k=2)
        assert len(results) == 2
        assert results[0][1] > results[1][1]  # First has higher score
    
    def test_search_top_k(self):
        bm25 = BM25Index()
        bm25.index(["doc1", "doc2", "doc3", "doc4", "doc5"])
        results = bm25.search("doc", top_k=3)
        assert len(results) == 3
    
    def test_search_no_match(self):
        bm25 = BM25Index()
        bm25.index(["hello world", "foo bar"])
        results = bm25.search("xyz", top_k=2)
        # All scores should be 0
        assert all(score == 0 for _, score in results)
    
    def test_search_code_like(self):
        bm25 = BM25Index()
        bm25.index([
            "def calculate_sum(a, b): return a + b",
            "def calculate_product(a, b): return a * b",
            "class Calculator: pass",
        ])
        results = bm25.search("calculate sum", top_k=3)
        assert results[0][0] == 0  # First doc matches best


class TestReciprocalRankFusion:
    """Tests for reciprocal rank fusion."""
    
    def test_single_ranking(self):
        rankings = [[(0, 0.9), (1, 0.8), (2, 0.7)]]
        fused = reciprocal_rank_fusion(rankings)
        assert fused[0][0] == 0
        assert fused[1][0] == 1
        assert fused[2][0] == 2
    
    def test_two_rankings_same_order(self):
        rankings = [
            [(0, 0.9), (1, 0.8), (2, 0.7)],
            [(0, 0.95), (1, 0.85), (2, 0.75)],
        ]
        fused = reciprocal_rank_fusion(rankings)
        # Same order should be preserved
        assert fused[0][0] == 0
        assert fused[1][0] == 1
        assert fused[2][0] == 2
    
    def test_two_rankings_different_order(self):
        rankings = [
            [(0, 0.9), (1, 0.8), (2, 0.7)],
            [(2, 0.9), (1, 0.8), (0, 0.7)],
        ]
        fused = reciprocal_rank_fusion(rankings)
        # Doc 1 is consistently in second place
        # Docs 0 and 2 alternate
        assert len(fused) == 3
    
    def test_empty_rankings(self):
        rankings = []
        fused = reciprocal_rank_fusion(rankings)
        assert fused == []
    
    def test_disjoint_rankings(self):
        rankings = [
            [(0, 0.9), (1, 0.8)],
            [(2, 0.9), (3, 0.8)],
        ]
        fused = reciprocal_rank_fusion(rankings)
        assert len(fused) == 4