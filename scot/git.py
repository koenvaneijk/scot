"""Git utilities for finding repo root and tracked files."""
import subprocess
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS


def get_repo_root(path: Path = None) -> Path | None:
    """Get the git repository root for the given path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def get_tracked_files(repo_root: Path) -> list[Path]:
    """Get list of git-tracked files with supported extensions."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            path = Path(line)
            if path.suffix in SUPPORTED_EXTENSIONS:
                files.append(path)
        return files
    except subprocess.CalledProcessError:
        return []