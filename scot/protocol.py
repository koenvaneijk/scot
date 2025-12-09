"""Communication protocol between CLI and daemon."""
import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Request:
    """Request from CLI to daemon."""
    action: str  # "search", "index", "status", "ping"
    repo_path: str = ""
    query: str = ""
    top_k: int = 5
    file_pattern: str = ""
    force_reindex: bool = False
    mode: str = "hybrid"  # "hybrid", "semantic", "lexical"
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> "Request":
        return cls(**json.loads(data))


@dataclass
class Response:
    """Response from daemon to CLI."""
    success: bool
    error: str = ""
    results: list[dict] = None  # For search results
    stats: dict = None  # For index stats
    
    def to_json(self) -> str:
        d = asdict(self)
        if d["results"] is None:
            d["results"] = []
        if d["stats"] is None:
            d["stats"] = {}
        return json.dumps(d)
    
    @classmethod
    def from_json(cls, data: str) -> "Response":
        return cls(**json.loads(data))