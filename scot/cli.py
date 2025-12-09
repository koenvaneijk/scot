"""SCOT command-line interface."""
import argparse
import sys
from pathlib import Path

from .git import get_repo_root
from .client import ensure_daemon_running, send_request
from .protocol import Request
from .search import SearchResult
from .display import print_results
from .daemon import stop_daemon, daemon_status, PID_FILE


def main():
    """Main entry point for scot command."""
    parser = argparse.ArgumentParser(
        description="SCOT - Semantic Code Ordering Tool",
        usage="scot [options] <query>",
    )
    
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-n", "--num", type=int, default=5,
                       help="Number of results (default: 5)")
    parser.add_argument("-c", "--context", action="store_true",
                       help="Show full context (no truncation)")
    parser.add_argument("-1", "--oneline", action="store_true",
                       help="One-line output format")
    parser.add_argument("-f", "--filter", type=str, default="",
                       help="Filter by file pattern (e.g., '*.py')")
    
    # Management commands
    parser.add_argument("--reindex", action="store_true",
                       help="Force reindex current repo")
    parser.add_argument("--status", action="store_true",
                       help="Show index status")
    parser.add_argument("--start-daemon", action="store_true",
                       help="Start the daemon")
    parser.add_argument("--stop-daemon", action="store_true",
                       help="Stop the daemon")
    parser.add_argument("--daemon-status", action="store_true",
                       help="Check daemon status")
    
    args = parser.parse_args()
    
    # Handle daemon management commands
    if args.stop_daemon:
        stop_daemon()
        return
    
    if args.daemon_status:
        if daemon_status():
            pid = PID_FILE.read_text().strip()
            print(f"Daemon running (pid {pid})")
        else:
            print("Daemon not running")
        return
    
    if args.start_daemon:
        if ensure_daemon_running():
            print("Daemon is running")
        else:
            print("Failed to start daemon")
            sys.exit(1)
        return
    
    # Find repo root
    repo_root = get_repo_root()
    if repo_root is None:
        print("Error: Not in a git repository", file=sys.stderr)
        sys.exit(1)
    
    # Ensure daemon is running
    if not ensure_daemon_running():
        sys.exit(1)
    
    # Handle reindex
    if args.reindex:
        request = Request(
            action="index",
            repo_path=str(repo_root),
            force_reindex=True,
        )
        response = send_request(request)
        if response.success:
            stats = response.stats
            print(f"Reindexed: {stats.get('files_updated', 0)} files, "
                  f"{stats.get('chunks_added', 0)} chunks")
        else:
            print(f"Error: {response.error}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Handle status
    if args.status:
        request = Request(
            action="status",
            repo_path=str(repo_root),
        )
        response = send_request(request)
        if response.success:
            print(f"Repo: {repo_root}")
            print(f"Daemon: running")
        else:
            print(f"Error: {response.error}", file=sys.stderr)
        return
    
    # Search query required for search
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    # Perform search
    request = Request(
        action="search",
        repo_path=str(repo_root),
        query=args.query,
        top_k=args.num,
        file_pattern=args.filter,
    )
    
    response = send_request(request)
    
    if not response.success:
        print(f"Error: {response.error}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to SearchResult objects
    results = [
        SearchResult(
            file_path=r["file_path"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            score=r["score"],
            chunk_text=r["chunk_text"],
        )
        for r in response.results
    ]
    
    print_results(results, full_context=args.context, oneline=args.oneline)


if __name__ == "__main__":
    main()