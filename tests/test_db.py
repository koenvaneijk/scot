"""Tests for database operations."""
import os
import tempfile
import pytest
import numpy as np

from scot.db import (
    get_connection, get_or_create_repo, get_file_mtime,
    delete_file_chunks, insert_chunk, get_repo_chunks, get_repo_files
)


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    scot_dir = tmp_path / ".scot"
    scot_dir.mkdir()
    
    # Patch the config to use test paths (need to patch both config and db modules
    # since db.py imports DB_PATH at module load time)
    monkeypatch.setattr("scot.config.DB_PATH", db_path)
    monkeypatch.setattr("scot.config.SCOT_DIR", scot_dir)
    monkeypatch.setattr("scot.db.DB_PATH", db_path)
    
    conn = get_connection()
    yield conn
    conn.close()


class TestGetOrCreateRepo:
    """Tests for get_or_create_repo."""
    
    def test_create_new_repo(self, test_db):
        repo_id = get_or_create_repo(test_db, "/path/to/repo")
        assert repo_id > 0
    
    def test_get_existing_repo(self, test_db):
        repo_id1 = get_or_create_repo(test_db, "/path/to/repo")
        repo_id2 = get_or_create_repo(test_db, "/path/to/repo")
        assert repo_id1 == repo_id2
    
    def test_different_repos(self, test_db):
        repo_id1 = get_or_create_repo(test_db, "/path/to/repo1")
        repo_id2 = get_or_create_repo(test_db, "/path/to/repo2")
        assert repo_id1 != repo_id2


class TestChunkOperations:
    """Tests for chunk CRUD operations."""
    
    def test_insert_and_get_chunks(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/insert")
        embedding = np.random.randn(768).astype(np.float32)
        
        insert_chunk(
            test_db, repo_id, "test.py", 1234567890.0,
            1, 10, "def test(): pass", embedding
        )
        test_db.commit()
        
        chunks = get_repo_chunks(test_db, repo_id)
        assert len(chunks) == 1
        assert chunks[0]["file_path"] == "test.py"
        assert chunks[0]["chunk_text"] == "def test(): pass"
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 10
    
    def test_get_file_mtime(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/mtime")
        embedding = np.random.randn(768).astype(np.float32)
        
        insert_chunk(
            test_db, repo_id, "test.py", 1234567890.0,
            1, 10, "code", embedding
        )
        test_db.commit()
        
        mtime = get_file_mtime(test_db, repo_id, "test.py")
        assert mtime == 1234567890.0
    
    def test_get_file_mtime_not_found(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/mtime_notfound")
        mtime = get_file_mtime(test_db, repo_id, "nonexistent.py")
        assert mtime is None
    
    def test_delete_file_chunks(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/delete")
        embedding = np.random.randn(768).astype(np.float32)
        
        insert_chunk(test_db, repo_id, "test.py", 1234567890.0, 1, 5, "chunk1", embedding)
        insert_chunk(test_db, repo_id, "test.py", 1234567890.0, 6, 10, "chunk2", embedding)
        insert_chunk(test_db, repo_id, "other.py", 1234567890.0, 1, 5, "chunk3", embedding)
        test_db.commit()
        
        delete_file_chunks(test_db, repo_id, "test.py")
        test_db.commit()
        
        chunks = get_repo_chunks(test_db, repo_id)
        assert len(chunks) == 1
        assert chunks[0]["file_path"] == "other.py"
    
    def test_get_repo_files(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/files")
        embedding = np.random.randn(768).astype(np.float32)
        
        insert_chunk(test_db, repo_id, "file1.py", 1234567890.0, 1, 5, "code", embedding)
        insert_chunk(test_db, repo_id, "file2.py", 1234567890.0, 1, 5, "code", embedding)
        insert_chunk(test_db, repo_id, "file1.py", 1234567890.0, 6, 10, "code", embedding)
        test_db.commit()
        
        files = get_repo_files(test_db, repo_id)
        assert files == {"file1.py", "file2.py"}
    
    def test_embedding_roundtrip(self, test_db):
        repo_id = get_or_create_repo(test_db, "/test/repo/embedding")
        original_embedding = np.random.randn(768).astype(np.float32)
        
        insert_chunk(
            test_db, repo_id, "test.py", 1234567890.0,
            1, 10, "code", original_embedding
        )
        test_db.commit()
        
        chunks = get_repo_chunks(test_db, repo_id)
        retrieved_embedding = chunks[0]["embedding"]
        
        np.testing.assert_array_almost_equal(original_embedding, retrieved_embedding)