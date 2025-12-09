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

# One-line output (good for piping)
scot -1 "parsing"

# Filter by file pattern
scot -f "*.py" "API endpoints"

# Search modes
scot -m hybrid "chunk_python"    # Default: combines semantic + lexical
scot -m semantic "authentication" # Pure embedding similarity
scot -m lexical "def search"      # Pure BM25 keyword matching

# Force reindex
scot --reindex

# Daemon management
scot --daemon-status
scot --stop-daemon
```

## How It Works

1. **First query**: SCOT starts a background daemon that loads the EmbeddingGemma model (~10-30 seconds)
2. **Indexing**: Parses git-tracked files (`.py`, `.md`, `.html`), chunks them semantically
3. **Hybrid search**: Combines BM25 (keyword matching) with semantic embeddings using Reciprocal Rank Fusion
4. **Caching**: Index is cached and updated incrementally based on file modification times

## Search Modes

| Mode | Best for | How it works |
|------|----------|--------------|
| `hybrid` (default) | General search | Combines BM25 + embeddings via RRF |
| `semantic` | Conceptual queries | "authentication" finds "login", "verify user" |
| `lexical` | Exact matches | Function names, variable names, specific terms |

## Architecture

- **`scot`**: Lightweight CLI client
- **`scotd`**: Background daemon holding the embedding model in memory
- **Storage**: `~/.scot/` contains the Unix socket, database, and logs

## Chunking

- **Python (`.py`)**: AST-based chunking - each function, method, and class becomes a chunk with parent context preserved
- **Markdown (`.md`)**: Line-based chunking with overlap
- **HTML (`.html`)**: Line-based chunking with overlap

## Dependencies

Only two runtime dependencies:
- `sentence-transformers` - For loading EmbeddingGemma
- `numpy` - For vector operations

BM25 is implemented from scratch (no external library).