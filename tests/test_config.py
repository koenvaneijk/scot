"""Tests for configuration."""
import os
import pytest
from pathlib import Path

from scot.config import ensure_scot_dir, SCOT_DIR


class TestEnsureScotDir:
    """Tests for ensure_scot_dir."""
    
    def test_creates_directory(self, tmp_path, monkeypatch):
        scot_path = tmp_path / ".scot"
        monkeypatch.setattr("scot.config.SCOT_DIR", scot_path)
        
        assert not scot_path.exists()
        ensure_scot_dir()
        assert scot_path.exists()
        assert scot_path.is_dir()
    
    def test_idempotent(self, tmp_path, monkeypatch):
        scot_path = tmp_path / ".scot"
        monkeypatch.setattr("scot.config.SCOT_DIR", scot_path)
        
        ensure_scot_dir()
        ensure_scot_dir()  # Should not raise
        assert scot_path.exists()