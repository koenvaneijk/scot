"""Tests for search functionality."""
import pytest

from scot.search import SearchResult, add_context_lines


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_create_result(self):
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=10,
            score=0.9,
            chunk_text="def test(): pass",
        )
        assert result.file_path == "test.py"
        assert result.start_line == 1
        assert result.end_line == 10
        assert result.score == 0.9
        assert result.chunk_text == "def test(): pass"


class TestAddContextLines:
    """Tests for add_context_lines function."""
    
    def test_adds_context(self, git_repo):
        # Create a test file
        content = "\n".join([f"line {i}" for i in range(20)])
        test_file = git_repo / "test.py"
        test_file.write_text(content)
        
        result = SearchResult(
            file_path="test.py",
            start_line=10,
            end_line=12,
            score=0.9,
            chunk_text="line 9\nline 10\nline 11",
        )
        
        new_result = add_context_lines(result, git_repo, context_lines=2)
        
        assert new_result.start_line == 8
        assert new_result.end_line == 14
    
    def test_context_at_file_start(self, git_repo):
        content = "\n".join([f"line {i}" for i in range(10)])
        test_file = git_repo / "test.py"
        test_file.write_text(content)
        
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=2,
            score=0.9,
            chunk_text="line 0\nline 1",
        )
        
        new_result = add_context_lines(result, git_repo, context_lines=3)
        
        # Should not go below line 1
        assert new_result.start_line == 1
    
    def test_context_at_file_end(self, git_repo):
        content = "\n".join([f"line {i}" for i in range(10)])
        test_file = git_repo / "test.py"
        test_file.write_text(content)
        
        result = SearchResult(
            file_path="test.py",
            start_line=8,
            end_line=10,
            score=0.9,
            chunk_text="line 7\nline 8\nline 9",
        )
        
        new_result = add_context_lines(result, git_repo, context_lines=5)
        
        # Should not exceed file length
        assert new_result.end_line == 10
    
    def test_file_not_found(self, git_repo):
        result = SearchResult(
            file_path="nonexistent.py",
            start_line=1,
            end_line=5,
            score=0.9,
            chunk_text="some code",
        )
        
        # Should return original result if file not found
        new_result = add_context_lines(result, git_repo, context_lines=3)
        assert new_result == result