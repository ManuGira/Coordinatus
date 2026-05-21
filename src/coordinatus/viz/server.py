"""SocketServer — background TCP listener that puts decoded JSON dicts into a queue.

No Qt dependency; the Presenter drains the queue on each timer tick.
The server is agnostic of message content — it puts every decoded dict
into the queue unchanged; interpretation is left to the Model.

Protocol — one JSON object per line (newline-terminated):

    Compound state update (spaces and/or points, replaces previous state):
        {
            "spaces": [
                {"id": "world", "parent_id": null,
                 "transform": [[1,0,0],[0,1,0],[0,0,1]]},
                {"id": "sensor", "parent_id": "world",
                 "transform": [[0.707,-0.707,1],[0.707,0.707,0],[0,0,1]]}
            ],
            "points": [
                {"channel": "lidar", "space_id": "sensor",
                 "coords": [[1.0,2.0],[3.0,4.0]]}
            ]
        }

    Display-space override:
        {"type": "set_display_space", "space_id": "world"}
"""

from __future__ import annotations

import json
import queue
import socket
import threading

HOST = "127.0.0.1"
PORT = 9876


class SocketServer(threading.Thread):
    """Background daemon that accepts TCP connections and puts decoded dicts into a queue.

    Args:
        inbox: The queue to put decoded message dicts into.
        host:  TCP bind address.
        port:  TCP bind port.
    """

    def __init__(
        self,
        inbox: queue.Queue[dict],
        host: str = HOST,
        port: int = PORT,
    ) -> None:
        super().__init__(daemon=True, name="viz-socket-server")
        self._inbox = inbox
        self._host = host
        self._port = port

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(16)
            print(f"[server] listening on {self._host}:{self._port}", flush=True)
            while True:
                try:
                    conn, addr = srv.accept()
                    print(f"[server] client: {addr}", flush=True)
                    threading.Thread(
                        target=self._handle, args=(conn, addr), daemon=True
                    ).start()
                except OSError as exc:
                    print(f"[server] accept error: {exc}", flush=True)

    def _handle(self, conn: socket.socket, addr: tuple) -> None:
        buf = b""
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    print(f"[server] disconnected: {addr}", flush=True)
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(f"[server] parse error ({exc}): {raw!r}", flush=True)
                        continue
                    self._inbox.put(msg)
