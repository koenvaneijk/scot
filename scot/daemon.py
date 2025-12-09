"""SCOT daemon - serves embedding requests."""
import os
import sys
import socket
import signal
import json
from pathlib import Path

from .config import SOCKET_PATH, PID_FILE, ensure_scot_dir
from .protocol import Request, Response
from .embedder import Embedder
from .search import search, SearchResult
from .indexer import index_repo


class Daemon:
    """The SCOT daemon server."""
    
    def __init__(self):
        self.embedder = Embedder()
        self.running = False
        self.socket = None
        # Cache BM25 indexes per repo: {repo_path: (bm25_index, chunk_ids, last_update)}
        self.bm25_cache: dict[str, tuple] = {}
    
    def start(self, foreground: bool = False):
        """Start the daemon."""
        ensure_scot_dir()
        
        # Check if already running
        if self._is_running():
            print("Daemon already running.")
            return
        
        if not foreground:
            # Fork to background
            pid = os.fork()
            if pid > 0:
                # Parent - wait a moment for startup
                print(f"Daemon starting (pid {pid})...")
                return
            
            # Child - become session leader
            os.setsid()
            
            # Redirect stdout/stderr
            log_path = SOCKET_PATH.parent / "scotd.log"
            sys.stdout = open(log_path, "a")
            sys.stderr = sys.stdout
        
        self._run()
    
    def _run(self):
        """Main daemon loop."""
        # Write PID file
        PID_FILE.write_text(str(os.getpid()))
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # Remove old socket if exists
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        
        # Create Unix socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(SOCKET_PATH))
        self.socket.listen(5)
        self.socket.settimeout(1.0)  # Allow periodic checks
        
        print(f"Daemon listening on {SOCKET_PATH}")
        
        # Load model on startup
        self.embedder.load()
        
        self.running = True
        while self.running:
            try:
                conn, _ = self.socket.accept()
                self._handle_connection(conn)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")
        
        self._cleanup()
    
    def _handle_connection(self, conn: socket.socket):
        """Handle a client connection."""
        try:
            # Read complete message with length prefix
            data = self._recv_message(conn)
            request = Request.from_json(data)
            response = self._process_request(request)
            self._send_message(conn, response.to_json())
        except Exception as e:
            error_response = Response(success=False, error=str(e))
            try:
                self._send_message(conn, error_response.to_json())
            except Exception:
                pass
        finally:
            conn.close()
    
    def _recv_message(self, conn: socket.socket) -> str:
        """Receive a length-prefixed message."""
        # Read 8-byte length header
        header = b""
        while len(header) < 8:
            chunk = conn.recv(8 - len(header))
            if not chunk:
                raise ConnectionError("Connection closed while reading header")
            header += chunk
        
        msg_len = int(header.decode("utf-8"))
        
        # Read message body
        data = b""
        while len(data) < msg_len:
            chunk = conn.recv(min(65536, msg_len - len(data)))
            if not chunk:
                raise ConnectionError("Connection closed while reading message")
            data += chunk
        
        return data.decode("utf-8")
    
    def _send_message(self, conn: socket.socket, message: str):
        """Send a length-prefixed message."""
        data = message.encode("utf-8")
        header = f"{len(data):08d}".encode("utf-8")
        conn.sendall(header + data)
    
    def _process_request(self, request: Request) -> Response:
        """Process a request and return response."""
        if request.action == "ping":
            return Response(success=True)
        
        elif request.action == "search":
            repo_path = Path(request.repo_path)
            results = search(
                repo_path,
                request.query,
                self.embedder,
                top_k=request.top_k,
                file_pattern=request.file_pattern or None,
                mode=request.mode,
                bm25_cache=self.bm25_cache,
            )
            return Response(
                success=True,
                results=[{
                    "file_path": r.file_path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                    "chunk_text": r.chunk_text,
                } for r in results]
            )
        
        elif request.action == "index":
            repo_path = Path(request.repo_path)
            stats = index_repo(repo_path, self.embedder, force=request.force_reindex)
            return Response(success=True, stats=stats)
        
        elif request.action == "status":
            from .db import get_connection, get_or_create_repo
            repo_path = Path(request.repo_path) if request.repo_path else None
            stats = {"status": "running", "pid": os.getpid()}
            
            if repo_path:
                conn = get_connection()
                repo_id = get_or_create_repo(conn, str(repo_path))
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM chunks WHERE repo_id = ?",
                    (repo_id,)
                )
                chunk_count = cursor.fetchone()["count"]
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT file_path) as count FROM chunks WHERE repo_id = ?",
                    (repo_id,)
                )
                file_count = cursor.fetchone()["count"]
                cursor = conn.execute(
                    "SELECT last_indexed FROM repos WHERE id = ?",
                    (repo_id,)
                )
                last_indexed = cursor.fetchone()["last_indexed"]
                conn.close()
                
                stats.update({
                    "repo": str(repo_path),
                    "files": file_count,
                    "chunks": chunk_count,
                    "last_indexed": last_indexed,
                    "bm25_cached": str(repo_path) in self.bm25_cache,
                })
            
            return Response(success=True, stats=stats)
        
        else:
            return Response(success=False, error=f"Unknown action: {request.action}")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _cleanup(self):
        """Clean up resources."""
        if self.socket:
            self.socket.close()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()
        print("Daemon stopped.")
    
    def _is_running(self) -> bool:
        """Check if daemon is already running."""
        if not PID_FILE.exists():
            return False
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return True
        except (ProcessLookupError, ValueError):
            # Stale PID file
            PID_FILE.unlink()
            return False


def stop_daemon():
    """Stop the running daemon."""
    if not PID_FILE.exists():
        print("Daemon not running.")
        return
    
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (pid {pid})")
    except ProcessLookupError:
        print("Daemon not running (stale PID file).")
        PID_FILE.unlink()
    except Exception as e:
        print(f"Error stopping daemon: {e}")


def daemon_status() -> bool:
    """Check if daemon is running. Returns True if running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def main():
    """Entry point for scotd command."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SCOT daemon")
    parser.add_argument("command", nargs="?", default="start",
                       choices=["start", "stop", "status"],
                       help="Daemon command")
    parser.add_argument("--foreground", "-f", action="store_true",
                       help="Run in foreground")
    
    args = parser.parse_args()
    
    if args.command == "start":
        daemon = Daemon()
        daemon.start(foreground=args.foreground)
    elif args.command == "stop":
        stop_daemon()
    elif args.command == "status":
        if daemon_status():
            pid = PID_FILE.read_text().strip()
            print(f"Daemon running (pid {pid})")
        else:
            print("Daemon not running")


if __name__ == "__main__":
    main()