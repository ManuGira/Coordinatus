"""Tests for viz/server.py — SocketServer (no Qt dependency)."""

from __future__ import annotations

import json
import queue
import socket
import time

import pytest

from coordinatus.viz.server import SocketServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> tuple[SocketServer, queue.Queue[dict]]:
    inbox: queue.Queue[dict] = queue.Queue()
    srv = SocketServer(inbox, host="127.0.0.1", port=port)
    srv.start()
    # Give the server thread a moment to bind and listen
    time.sleep(0.05)
    return srv, inbox


def _send_lines(port: int, *lines: str) -> None:
    """Connect and send newline-terminated JSON strings, then close."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
        c.connect(("127.0.0.1", port))
        payload = "".join(line + "\n" for line in lines)
        c.sendall(payload.encode())


def _drain(inbox: queue.Queue[dict], n: int, timeout: float = 1.0) -> list[dict]:
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            items.append(inbox.get(timeout=remaining))
        except queue.Empty:
            break
    return items


class TestSocketServerQueues:
    def test_scalar_value_message(self) -> None:
        port = _find_free_port()
        _, inbox = _start_server(port)

        _send_lines(port, json.dumps({"channel": "rpm", "value": 3000.0}))

        msgs = _drain(inbox, 1)
        assert len(msgs) == 1
        assert msgs[0] == {"channel": "rpm", "value": 3000.0}

    def test_compound_state_message(self) -> None:
        """A single message carrying spaces + points lists is passed through intact."""
        port = _find_free_port()
        _, inbox = _start_server(port)

        msg = {
            "spaces": [
                {"id": "world", "parent_id": None,
                 "transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                {"id": "sensor", "parent_id": "world",
                 "transform": [[1, 0, 1], [0, 1, 0], [0, 0, 1]]},
            ],
            "points": [
                {"channel": "lidar", "space_id": "sensor",
                 "coords": [[1.0, 2.0], [3.0, 4.0]]},
            ],
        }
        _send_lines(port, json.dumps(msg))

        msgs = _drain(inbox, 1)
        assert len(msgs) == 1
        # Server is a pass-through — the dict arrives verbatim.
        assert msgs[0] == msg
        assert len(msgs[0]["spaces"]) == 2
        assert len(msgs[0]["points"]) == 1

    def test_multiple_messages_in_one_send(self) -> None:
        port = _find_free_port()
        _, inbox = _start_server(port)

        _send_lines(
            port,
            json.dumps({"channel": "a", "value": 1.0}),
            json.dumps({"channel": "b", "value": 2.0}),
            json.dumps({"channel": "c", "value": 3.0}),
        )

        msgs = _drain(inbox, 3)
        assert len(msgs) == 3
        channels = {m["channel"] for m in msgs}
        assert channels == {"a", "b", "c"}

    def test_invalid_json_is_dropped(self) -> None:
        port = _find_free_port()
        _, inbox = _start_server(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
            c.connect(("127.0.0.1", port))
            c.sendall(b"not-json\n")
            c.sendall(json.dumps({"channel": "ok", "value": 9.0}).encode() + b"\n")

        msgs = _drain(inbox, 1)
        assert len(msgs) == 1
        assert msgs[0]["channel"] == "ok"

    def test_blank_lines_are_ignored(self) -> None:
        port = _find_free_port()
        _, inbox = _start_server(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
            c.connect(("127.0.0.1", port))
            c.sendall(b"\n\n")
            c.sendall(json.dumps({"channel": "x", "value": 5.0}).encode() + b"\n")

        msgs = _drain(inbox, 1)
        assert len(msgs) == 1
        assert msgs[0]["channel"] == "x"

    def test_multiple_clients(self) -> None:
        port = _find_free_port()
        _, inbox = _start_server(port)

        _send_lines(port, json.dumps({"channel": "c1", "value": 1.0}))
        _send_lines(port, json.dumps({"channel": "c2", "value": 2.0}))

        msgs = _drain(inbox, 2)
        assert len(msgs) == 2
        channels = {m["channel"] for m in msgs}
        assert channels == {"c1", "c2"}
