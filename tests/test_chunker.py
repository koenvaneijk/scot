"""Tests for code chunking."""
import pytest
from pathlib import Path

from scot.chunker import Chunk, chunk_file, chunk_python, chunk_markdown, chunk_lines


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_create_chunk(self):
        chunk = Chunk(text="hello", start_line=1, end_line=5)
        assert chunk.text == "hello"
        assert chunk.start_line == 1
        assert chunk.end_line == 5


class TestChunkPython:
    """Tests for Python chunking."""
    
    def test_chunk_function(self):
        code = '''def hello():
    """Say hello."""
    print("hello")
'''
        chunks = chunk_python(code)
        assert len(chunks) >= 1
        assert any("def hello" in c.text for c in chunks)
    
    def test_chunk_class(self):
        code = '''class MyClass:
    """A class."""
    
    def method(self):
        pass
'''
        chunks = chunk_python(code)
        assert len(chunks) >= 1
        # Should have both class and method chunks
        assert any("class MyClass" in c.text for c in chunks)
        assert any("def method" in c.text for c in chunks)
    
    def test_chunk_method_includes_class_context(self):
        code = '''class Calculator:
    def add(self, a, b):
        return a + b
'''
        chunks = chunk_python(code)
        method_chunks = [c for c in chunks if "def add" in c.text]
        assert len(method_chunks) >= 1
        # Method should include class context
        assert any("class Calculator" in c.text for c in method_chunks)
    
    def test_chunk_async_function(self):
        code = '''async def fetch_data():
    """Fetch data asynchronously."""
    return await some_call()
'''
        chunks = chunk_python(code)
        assert len(chunks) >= 1
        assert any("async def fetch_data" in c.text for c in chunks)
    
    def test_chunk_multiple_functions(self):
        code = '''def foo():
    pass

def bar():
    pass

def baz():
    pass
'''
        chunks = chunk_python(code)
        func_names = ["foo", "bar", "baz"]
        for name in func_names:
            assert any(f"def {name}" in c.text for c in chunks)
    
    def test_chunk_syntax_error_fallback(self):
        # Invalid Python should fall back to line-based
        code = '''def broken(
    # missing closing paren
    pass
'''
        chunks = chunk_python(code)
        # Should not raise, should fall back
        assert len(chunks) >= 1
    
    def test_chunk_empty_file(self):
        chunks = chunk_python("")
        assert chunks == []
    
    def test_chunk_preserves_line_numbers(self):
        code = '''# comment
# comment

def hello():
    pass
'''
        chunks = chunk_python(code)
        func_chunk = next(c for c in chunks if "def hello" in c.text)
        assert func_chunk.start_line == 4


class TestChunkMarkdown:
    """Tests for Markdown chunking."""
    
    def test_chunk_by_headers(self):
        content = '''# Header 1

Content under header 1.

## Header 2

Content under header 2.
'''
        chunks = chunk_markdown(content)
        assert len(chunks) >= 2
    
    def test_chunk_no_headers(self):
        content = '''Just some text
without any headers
multiple lines
'''
        chunks = chunk_markdown(content)
        # Should fall back to line-based
        assert len(chunks) >= 1
    
    def test_chunk_empty(self):
        chunks = chunk_markdown("")
        assert chunks == []
    
    def test_chunk_preserves_content(self):
        content = '''# Title

Some important content here.
'''
        chunks = chunk_markdown(content)
        assert any("Title" in c.text for c in chunks)
        assert any("important content" in c.text for c in chunks)


class TestChunkLines:
    """Tests for line-based chunking."""
    
    def test_small_file(self):
        content = "line1\nline2\nline3"
        chunks = chunk_lines(content)
        assert len(chunks) == 1
        assert chunks[0].text == content
    
    def test_empty_file(self):
        chunks = chunk_lines("")
        assert chunks == []
    
    def test_line_numbers(self):
        content = "line1\nline2\nline3"
        chunks = chunk_lines(content)
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3
    
    def test_large_file_splits(self):
        # Create content larger than CHUNK_SIZE_LINES
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        chunks = chunk_lines(content)
        assert len(chunks) > 1


class TestChunkFile:
    """Tests for chunk_file dispatch."""
    
    def test_python_file(self):
        code = "def hello(): pass"
        chunks = chunk_file(Path("test.py"), code)
        assert len(chunks) >= 1
    
    def test_markdown_file(self):
        content = "# Header\n\nContent"
        chunks = chunk_file(Path("test.md"), content)
        assert len(chunks) >= 1
    
    def test_other_file(self):
        content = "some text\nmore text"
        chunks = chunk_file(Path("test.txt"), content)
        assert len(chunks) >= 1
    
    def test_html_file(self):
        content = "<html><body>Hello</body></html>"
        chunks = chunk_file(Path("test.html"), content)
        assert len(chunks) >= 1