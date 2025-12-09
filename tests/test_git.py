"""Tests for git utilities."""
import subprocess
import pytest
from pathlib import Path

from scot.git import get_repo_root, get_tracked_files


class TestGetRepoRoot:
    """Tests for get_repo_root."""
    
    def test_in_git_repo(self, git_repo):
        root = get_repo_root(git_repo)
        assert root == git_repo
    
    def test_in_subdirectory(self, git_repo):
        subdir = git_repo / "subdir"
        subdir.mkdir()
        root = get_repo_root(subdir)
        assert root == git_repo
    
    def test_not_in_git_repo(self, temp_dir):
        root = get_repo_root(temp_dir)
        assert root is None


class TestGetTrackedFiles:
    """Tests for get_tracked_files."""
    
    def test_empty_repo(self, git_repo):
        files = get_tracked_files(git_repo)
        assert files == []
    
    def test_with_python_file(self, git_repo, sample_python_file):
        files = get_tracked_files(git_repo)
        assert Path("calculator.py") in files
    
    def test_filters_unsupported_extensions(self, git_repo):
        # Create a .txt file (not in SUPPORTED_EXTENSIONS)
        txt_file = git_repo / "notes.txt"
        txt_file.write_text("some notes")
        subprocess.run(["git", "add", "notes.txt"], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add notes"], cwd=git_repo, capture_output=True)
        
        files = get_tracked_files(git_repo)
        assert Path("notes.txt") not in files
    
    def test_includes_markdown(self, git_repo, sample_markdown_file):
        files = get_tracked_files(git_repo)
        assert Path("README.md") in files
    
    def test_multiple_files(self, git_repo, sample_python_file, sample_markdown_file):
        files = get_tracked_files(git_repo)
        assert len(files) == 2
        assert Path("calculator.py") in files
        assert Path("README.md") in files