"""Code chunking - AST-based for Python, line-based for others."""
import ast
from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_SIZE_LINES, CHUNK_OVERLAP_LINES


@dataclass
class Chunk:
    """A chunk of code with location info."""
    text: str
    start_line: int
    end_line: int


def chunk_file(file_path: Path, content: str) -> list[Chunk]:
    """Chunk a file based on its type."""
    if file_path.suffix == ".py":
        try:
            return chunk_python(content)
        except SyntaxError:
            # Fall back to line-based if Python parsing fails
            return chunk_lines(content)
    else:
        return chunk_lines(content)


def chunk_python(content: str) -> list[Chunk]:
    """Chunk Python code using AST - extract functions, methods, classes."""
    tree = ast.parse(content)
    lines = content.splitlines()
    chunks = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Get the chunk text
            node_lines = lines[start_line - 1:end_line]
            chunk_text = "\n".join(node_lines)
            
            # For methods, include class context
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_class = _find_parent_class(tree, node)
                if parent_class:
                    chunk_text = f"class {parent_class}:\n" + _indent(chunk_text)
            
            # Truncate very long chunks
            if len(node_lines) > CHUNK_SIZE_LINES:
                preview_lines = node_lines[:CHUNK_SIZE_LINES - 2]
                remaining = len(node_lines) - len(preview_lines)
                chunk_text = "\n".join(preview_lines) + f"\n    # ... ({remaining} more lines)"
            
            chunks.append(Chunk(
                text=chunk_text,
                start_line=start_line,
                end_line=end_line,
            ))
    
    # If no AST nodes found, fall back to line-based
    if not chunks:
        return chunk_lines(content)
    
    return chunks


def _find_parent_class(tree: ast.AST, target_node: ast.AST) -> str | None:
    """Find the parent class name for a method."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if child is target_node:
                    return node.name
    return None


def _indent(text: str, spaces: int = 4) -> str:
    """Indent text by given number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def chunk_lines(content: str) -> list[Chunk]:
    """Chunk content by lines with overlap."""
    lines = content.splitlines()
    chunks = []
    
    if not lines:
        return chunks
    
    # Single chunk if small enough
    if len(lines) <= CHUNK_SIZE_LINES:
        return [Chunk(
            text=content,
            start_line=1,
            end_line=len(lines),
        )]
    
    # Sliding window with overlap
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE_LINES, len(lines))
        chunk_text = "\n".join(lines[start:end])
        
        chunks.append(Chunk(
            text=chunk_text,
            start_line=start + 1,
            end_line=end,
        ))
        
        # Move window
        start += CHUNK_SIZE_LINES - CHUNK_OVERLAP_LINES
        if start >= len(lines):
            break
    
    return chunks