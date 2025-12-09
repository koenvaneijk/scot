"""Pytest fixtures for scot tests."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def git_repo(temp_dir):
    """Create a temporary git repository."""
    subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=temp_dir, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=temp_dir, capture_output=True
    )
    yield temp_dir


@pytest.fixture
def sample_python_file(git_repo):
    """Create a sample Python file in the git repo."""
    code = '''"""Sample module."""

class Calculator:
    """A simple calculator class."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def divide(x: int, y: int) -> float:
    """Divide x by y."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y
'''
    file_path = git_repo / "calculator.py"
    file_path.write_text(code)
    subprocess.run(["git", "add", "calculator.py"], cwd=git_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add calculator"], cwd=git_repo, capture_output=True)
    return file_path


@pytest.fixture
def sample_markdown_file(git_repo):
    """Create a sample Markdown file in the git repo."""
    content = '''# Project README

This is a sample project.

## Installation

Run `pip install project` to install.

## Usage

Here's how to use it:

```python
from project import main
main()
```

## Contributing

Please submit PRs.

### Code Style

Follow PEP8.

### Testing

Run pytest.
'''
    file_path = git_repo / "README.md"
    file_path.write_text(content)
    subprocess.run(["git", "add", "README.md"], cwd=git_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add README"], cwd=git_repo, capture_output=True)
    return file_path


@pytest.fixture
def scot_dir(temp_dir):
    """Create a temporary SCOT directory."""
    scot_path = temp_dir / ".scot"
    scot_path.mkdir()
    # Set environment variable for tests
    old_env = os.environ.get("SCOT_DIR")
    os.environ["SCOT_DIR"] = str(scot_path)
    yield scot_path
    # Restore
    if old_env is not None:
        os.environ["SCOT_DIR"] = old_env
    else:
        del os.environ["SCOT_DIR"]