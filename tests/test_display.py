"""Tests for display formatting."""
import pytest

from scot.display import format_results, format_oneline, format_full, format_compact
from scot.search import SearchResult


@pytest.fixture
def sample_results():
    """Create sample search results."""
    return [
        SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=5,
            score=0.95,
            chunk_text="def hello():\n    print('hello')",
        ),
        SearchResult(
            file_path="other.py",
            start_line=10,
            end_line=20,
            score=0.85,
            chunk_text="class Foo:\n    pass",
        ),
    ]


class TestFormatResults:
    """Tests for format_results."""
    
    def test_empty_results(self):
        output = format_results([])
        assert output == "No results found."
    
    def test_with_results(self, sample_results):
        output = format_results(sample_results)
        assert "test.py" in output
        assert "other.py" in output
    
    def test_oneline_flag(self, sample_results):
        output = format_results(sample_results, oneline=True)
        lines = output.strip().split("\n")
        assert len(lines) == 2


class TestFormatOneline:
    """Tests for format_oneline."""
    
    def test_basic_output(self, sample_results):
        output = format_oneline(sample_results)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert "test.py:1" in lines[0]
        assert "0.95" in lines[0] or "0.950" in lines[0]
    
    def test_truncates_long_lines(self):
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=5,
            score=0.9,
            chunk_text="x" * 100,
        )
        output = format_oneline([result])
        assert "..." in output


class TestFormatCompact:
    """Tests for format_compact."""
    
    def test_basic_output(self, sample_results):
        output = format_compact(sample_results, full_context=False)
        assert "## test.py:1-5" in output
        assert "## other.py:10-20" in output
    
    def test_includes_code(self, sample_results):
        output = format_compact(sample_results, full_context=False)
        assert "def hello()" in output
        assert "class Foo" in output
    
    def test_no_decorative_separators(self, sample_results):
        output = format_compact(sample_results, full_context=False)
        assert "━" not in output
        assert "score:" not in output
    
    def test_truncates_long_chunks(self):
        long_code = "\n".join([f"line {i}" for i in range(50)])
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=50,
            score=0.9,
            chunk_text=long_code,
        )
        output = format_compact([result], full_context=False)
        assert "# ... 38 more lines" in output
    
    def test_full_context_no_truncation(self):
        long_code = "\n".join([f"line {i}" for i in range(50)])
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=50,
            score=0.9,
            chunk_text=long_code,
        )
        output = format_compact([result], full_context=True)
        assert "more lines" not in output
        assert "line 49" in output
    
    def test_compact_flag_in_format_results(self, sample_results):
        output = format_results(sample_results, compact=True)
        assert "## test.py:1-5" in output
        assert "━" not in output


class TestFormatFull:
    """Tests for format_full."""
    
    def test_includes_file_info(self, sample_results):
        output = format_full(sample_results, full_context=False)
        assert "test.py:1-5" in output
        assert "other.py:10-20" in output
    
    def test_includes_score(self, sample_results):
        output = format_full(sample_results, full_context=False)
        assert "0.95" in output or "0.950" in output
    
    def test_includes_code(self, sample_results):
        output = format_full(sample_results, full_context=False)
        assert "def hello()" in output
        assert "class Foo" in output
    
    def test_truncates_long_chunks(self):
        long_code = "\n".join([f"line {i}" for i in range(50)])
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=50,
            score=0.9,
            chunk_text=long_code,
        )
        output = format_full([result], full_context=False)
        assert "more lines" in output
    
    def test_full_context_no_truncation(self):
        long_code = "\n".join([f"line {i}" for i in range(50)])
        result = SearchResult(
            file_path="test.py",
            start_line=1,
            end_line=50,
            score=0.9,
            chunk_text=long_code,
        )
        output = format_full([result], full_context=True)
        assert "more lines" not in output
        assert "line 49" in output