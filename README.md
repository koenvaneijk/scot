# SCOT - Semantic Code Ordering Tool

A modern, embedding-based replacement for grep that searches codebases by meaning rather than by exact text.

## Installation

```bash
pip install -e .
```

**Note:** Requires accepting the Gemma license on Hugging Face:
1. Visit [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
2. Click "Acknowledge license"
3. Run `huggingface-cli login` and enter your token

## Usage

```bash
# Search for code semantically
scot "authentication logic"

# More results
scot -n 10 "database connection handling"

# Full context (no truncation)
scot -c "error handling"

# Context lines (like grep -C)
scot -C 5 "search function"         # Show 5 lines before/after each result

# One-line output (good for piping)
scot -1 "parsing"

# Filter by file pattern
scot -f "*.py" "API endpoints"

# Search modes
scot -m hybrid "chunk_python"     # Default: combines semantic + lexical
scot -m semantic "authentication" # Pure embedding similarity
scot -m lexical "def search"      # Pure BM25 keyword matching

# Index management
scot --reindex                    # Force reindex current repo
scot --status                     # Show index stats

# Daemon management
scot --daemon-status
scot --stop-daemon
```

## How It Works

1. **First query**: SCOT starts a background daemon that loads the EmbeddingGemma model (~10-30 seconds)
2. **Indexing**: Parses git-tracked files, chunks them semantically (AST for Python, headers for Markdown)
3. **Hybrid search**: Combines BM25 (keyword matching) with semantic embeddings using Reciprocal Rank Fusion
4. **Caching**: Index cached in SQLite, BM25 index cached in memory, updates incrementally based on mtime

## Search Modes

| Mode | Best for | How it works |
|------|----------|--------------|
| `hybrid` (default) | General search | Combines BM25 + embeddings via RRF |
| `semantic` | Conceptual queries | "authentication" finds "login", "verify user" |
| `lexical` | Exact matches | Function names, variable names, specific terms |

## Status

```bash
$ scot --status
Repository:   /home/user/project
Daemon:       running (pid 12345)
Files:        42
Chunks:       187
Last indexed: 2025-01-15 14:30:22
BM25 cached:  yes
```

## Architecture

```
┌─────────┐     Unix socket     ┌─────────────────────┐
│  scot   │ ←─────────────────→ │       scotd         │
│  (CLI)  │                     │  (daemon process)   │
└─────────┘                     │                     │
                                │  • EmbeddingGemma   │
                                │  • BM25 index cache │
                                └──────────┬──────────┘
                                           │
                                           ▼
                                    ~/.scot/index.db
```

- **`scot`**: Lightweight CLI client
- **`scotd`**: Background daemon holding the embedding model and BM25 index in memory
- **Storage**: `~/.scot/` contains the Unix socket, SQLite database, and logs

## Chunking

| File Type | Strategy |
|-----------|----------|
| Python (`.py`) | AST-based: functions, methods (with class context), class signatures |
| Markdown (`.md`) | Header-based: splits on `#` headers, respects section boundaries |
| HTML (`.html`) | Line-based: sliding window with overlap |

### Python Chunking Details

- **Functions/methods**: Full code, truncated if >50 lines
- **Methods**: Prefixed with `class ClassName:` for context
- **Classes**: Signature + docstring only (methods indexed separately)

## Dependencies

Only two runtime dependencies:
- `sentence-transformers` - For loading EmbeddingGemma
- `numpy` - For vector operations

BM25 is implemented from scratch (no external library).