"""Output formatting for search results."""
import sys

from .search import SearchResult


def format_results(
    results: list[SearchResult],
    full_context: bool = False,
    oneline: bool = False,
    compact: bool = False,
) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."
    
    if oneline:
        return format_oneline(results)
    
    if compact:
        return format_compact(results, full_context)
    
    return format_full(results, full_context)


def format_oneline(results: list[SearchResult]) -> str:
    """One-line-per-result format."""
    lines = []
    for r in results:
        # Get first meaningful line of chunk
        first_line = r.chunk_text.split("\n")[0][:60]
        if len(r.chunk_text.split("\n")[0]) > 60:
            first_line += "..."
        lines.append(f"{r.score:.3f}  {r.file_path}:{r.start_line:<6} {first_line}")
    return "\n".join(lines)


def format_compact(results: list[SearchResult], full_context: bool) -> str:
    """Compact format optimized for LLM consumption - minimal tokens."""
    output = []
    
    for r in results:
        # Simple header: file:start-end
        output.append(f"## {r.file_path}:{r.start_line}-{r.end_line}")
        
        # Code block
        chunk_lines = r.chunk_text.splitlines()
        if full_context or len(chunk_lines) <= 15:
            output.append(r.chunk_text)
        else:
            output.extend(chunk_lines[:12])
            remaining = len(chunk_lines) - 12
            output.append(f"# ... {remaining} more lines")
    
    return "\n".join(output)


def format_full(results: list[SearchResult], full_context: bool) -> str:
    """Full format with code display."""
    output = []
    separator = "━" * 70
    
    for r in results:
        # Header
        header = f"{r.file_path}:{r.start_line}-{r.end_line}"
        score_str = f"score: {r.score:.3f}"
        padding = 70 - len(header) - len(score_str) - 2
        
        output.append(separator)
        output.append(f"{header}{' ' * max(1, padding)}{score_str}")
        output.append(separator)
        output.append("")
        
        # Code
        chunk_lines = r.chunk_text.splitlines()
        if full_context or len(chunk_lines) <= 15:
            output.append(r.chunk_text)
        else:
            # Truncate
            output.extend(chunk_lines[:12])
            remaining = len(chunk_lines) - 12
            output.append(f"    # ... ({remaining} more lines)")
        
        output.append("")
    
    return "\n".join(output)


def print_results(
    results: list[SearchResult],
    full_context: bool = False,
    oneline: bool = False,
    compact: bool = False,
):
    """Print formatted results to stdout."""
    print(format_results(results, full_context, oneline, compact))