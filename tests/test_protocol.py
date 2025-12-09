"""Tests for communication protocol."""
import pytest

from scot.protocol import Request, Response


class TestRequest:
    """Tests for Request class."""
    
    def test_create_request(self):
        req = Request(action="search", query="hello")
        assert req.action == "search"
        assert req.query == "hello"
    
    def test_default_values(self):
        req = Request(action="ping")
        assert req.repo_path == ""
        assert req.query == ""
        assert req.top_k == 5
        assert req.file_pattern == ""
        assert req.force_reindex == False
        assert req.mode == "hybrid"
    
    def test_to_json(self):
        req = Request(action="search", query="test", top_k=10)
        json_str = req.to_json()
        assert '"action": "search"' in json_str
        assert '"query": "test"' in json_str
        assert '"top_k": 10' in json_str
    
    def test_from_json(self):
        req = Request(action="search", query="test", top_k=10)
        json_str = req.to_json()
        req2 = Request.from_json(json_str)
        assert req2.action == req.action
        assert req2.query == req.query
        assert req2.top_k == req.top_k
    
    def test_roundtrip(self):
        req = Request(
            action="search",
            repo_path="/path/to/repo",
            query="find function",
            top_k=15,
            file_pattern="*.py",
            force_reindex=True,
            mode="semantic",
        )
        json_str = req.to_json()
        req2 = Request.from_json(json_str)
        assert req2.action == req.action
        assert req2.repo_path == req.repo_path
        assert req2.query == req.query
        assert req2.top_k == req.top_k
        assert req2.file_pattern == req.file_pattern
        assert req2.force_reindex == req.force_reindex
        assert req2.mode == req.mode


class TestResponse:
    """Tests for Response class."""
    
    def test_create_success_response(self):
        resp = Response(success=True)
        assert resp.success == True
        assert resp.error == ""
    
    def test_create_error_response(self):
        resp = Response(success=False, error="Something went wrong")
        assert resp.success == False
        assert resp.error == "Something went wrong"
    
    def test_with_results(self):
        results = [
            {"file_path": "test.py", "score": 0.9},
            {"file_path": "other.py", "score": 0.8},
        ]
        resp = Response(success=True, results=results)
        assert len(resp.results) == 2
    
    def test_to_json_with_none_values(self):
        resp = Response(success=True)
        json_str = resp.to_json()
        # Should not raise even with None values
        assert '"success": true' in json_str
    
    def test_from_json(self):
        resp = Response(success=True, results=[{"a": 1}], stats={"count": 5})
        json_str = resp.to_json()
        resp2 = Response.from_json(json_str)
        assert resp2.success == True
        assert resp2.results == [{"a": 1}]
        assert resp2.stats == {"count": 5}
    
    def test_roundtrip(self):
        resp = Response(
            success=False,
            error="Test error",
            results=[{"key": "value"}],
            stats={"metric": 42},
        )
        json_str = resp.to_json()
        resp2 = Response.from_json(json_str)
        assert resp2.success == resp.success
        assert resp2.error == resp.error
        assert resp2.results == resp.results
        assert resp2.stats == resp.stats