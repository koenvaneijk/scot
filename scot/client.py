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
    
    print("Starting daemon (first run loads model, may take 10-30s)...")
    
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
        
        # Send length-prefixed message
        data = request.to_json().encode("utf-8")
        header = f"{len(data):08d}".encode("utf-8")
        sock.sendall(header + data)
        
        # Read length-prefixed response
        header = b""
        while len(header) < 8:
            chunk = sock.recv(8 - len(header))
            if not chunk:
                raise ConnectionError("Connection closed while reading header")
            header += chunk
        
        msg_len = int(header.decode("utf-8"))
        
        response_data = b""
        while len(response_data) < msg_len:
            chunk = sock.recv(min(65536, msg_len - len(response_data)))
            if not chunk:
                raise ConnectionError("Connection closed while reading response")
            response_data += chunk
        
        return Response.from_json(response_data.decode("utf-8"))
    finally:
        sock.close()