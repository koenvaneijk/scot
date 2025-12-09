# SCOT - Semantic Code Ordering Tool

A modern, embedding-based replacement for grep that searches codebases by meaning rather than by exact text.

## Installation

```bash
pip install -e .
```

**Note:** Requires accepting the Gemma license on Hugging Face. Visit [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) and click "Acknowledge license".

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

# Force reindex
scot --reindex

# Daemon management
scot --daemon-status
scot --stop-daemon
```

## How It Works

1. **First query**: SCOT starts a background daemon that loads the EmbeddingGemma model (~10-30 seconds)
2. **Indexing**: Parses git-tracked files (`.py`, `.md`, `.html`), chunks them semantically
3. **Search**: Embeds your query, finds most similar chunks via cosine similarity
4. **Caching**: Index is cached and updated incrementally based on file modification times

## Architecture

- **`scot`**: Lightweight CLI client
- **`scotd`**: Background daemon holding the embedding model in memory
- **Storage**: `~/.scot/` contains the Unix socket, database, and logs

## Supported File Types

- **Python (`.py`)**: AST-based chunking (functions, methods, classes)
- **Markdown (`.md`)**: Line-based chunking with overlap
- **HTML (`.html`)**: Line-based chunking with overlap