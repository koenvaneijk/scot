"""Client for communicating with the daemon."""
import socket
import time
import subprocess
import sys
from pathlib import Path

from .config import SOCKET_PATH
from .protocol import Request, Response
from .daemon import daemon_status


def ensure_daemon_running() -> bool:
    """Ensure the daemon is running, starting it if needed.
    
    Returns True if daemon is ready, False if failed to start.
    """
    if daemon_status():
        return True
    
    print("Starting daemon...")
    
    # Start daemon in background
    subprocess.Popen(
        [sys.executable, "-m", "scot.daemon", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for it to be ready
    for _ in range(60):  # Wait up to 60 seconds for model to load
        time.sleep(1)
        if SOCKET_PATH.exists():
            # Try to ping
            try:
                response = send_request(Request(action="ping"))
                if response.success:
                    print("Daemon ready.")
                    return True
            except Exception:
                pass
    
    print("Failed to start daemon.")
    return False


def send_request(request: Request) -> Response:
    """Send a request to the daemon and get response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(SOCKET_PATH))
        sock.sendall(request.to_json().encode("utf-8"))
        
        # Read response
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        
        return Response.from_json(data.decode("utf-8"))
    finally:
        sock.close()