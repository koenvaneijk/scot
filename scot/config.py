"""Configuration and paths for SCOT."""
import os
from pathlib import Path

# Central storage directory
SCOT_DIR = Path(os.environ.get("SCOT_DIR", Path.home() / ".scot"))
SOCKET_PATH = SCOT_DIR / "scotd.sock"
PID_FILE = SCOT_DIR / "scotd.pid"
DB_PATH = SCOT_DIR / "index.db"

# Model configuration
MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768

# Indexing configuration
SUPPORTED_EXTENSIONS = {".py", ".md", ".html"}
CHUNK_SIZE_LINES = 50
CHUNK_OVERLAP_LINES = 10

# Search defaults
DEFAULT_TOP_K = 5


def ensure_scot_dir():
    """Create the SCOT directory if it doesn't exist."""
    SCOT_DIR.mkdir(parents=True, exist_ok=True)