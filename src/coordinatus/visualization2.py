"""Two-panel PyQtGraph + PySide6 live application.

Left panel  — interactive directed graph: hover highlights nodes, click selects.
Right panel — rolling time-series: animated by timer, mouse (throttled), socket.

Run:
    uv run python src/coordinatus/visualization2.py

Send right-panel data (rolling channels):
    {"channel": "rpm", "value": 3000.0}

Add/update graph nodes and edges dynamically:
    {"type": "graph_node", "id": "n1", "x": 0.0, "y": 0.0, "label": "Node 1"}
    {"type": "graph_edge", "from": "n1", "to": "n2"}

All messages must be newline-terminated JSON sent to HOST:PORT.

Requirements (dev group):
    uv add pyqtgraph PySide6 --dev
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Running as `python src/coordinatus/visualization2.py` inserts the package
# directory into sys.path[0], which shadows the stdlib `types` module with our
# own types.py and causes a circular import.  Remove it before any other import.
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
while _pkg_dir in _sys.path:
    _sys.path.remove(_pkg_dir)
del _pkg_dir, _os, _sys

import json
import math
import socket
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import pyqtgraph as pg
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv add pyqtgraph PySide6 --dev"
    ) from exc

# ── Configuration ──────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 9876
MOUSE_THROTTLE_MS = 50    # min gap (ms) between mouse-driven buffer writes
BUFFER_LEN = 400           # rolling history length per channel
RENDER_INTERVAL_MS = 16    # ~60 fps tick
TIMER_FREQ_HZ = 0.4        # frequency of built-in sine / cosine channels

# Catppuccin Mocha palette
_BG       = "#1e1e2e"
_FG       = "#cdd6f4"
_SURFACE0 = "#313244"
_OVERLAY0 = "#6c7086"
_BLUE     = "#89b4fa"
_MAUVE    = "#cba6f7"
_PINK     = "#f5c2e7"
_GREEN    = "#a6e3a1"
_YELLOW   = "#f9e2af"
_RED      = "#f38ba8"

_CHANNEL_COLORS = [_BLUE, "#f06292", "#aed581", _YELLOW, _MAUVE, _GREEN, "#ffcc02", _RED]

# ── Thread-safe bridge (socket thread → Qt event loop) ────────────────────────


class _DataBridge(QObject):
    value_received = Signal(str, float)              # channel, value  → right panel
    node_received  = Signal(str, float, float, str)  # id, x, y, label → left panel
    edge_received  = Signal(str, str)                # from_id, to_id  → left panel


# ── Socket server ──────────────────────────────────────────────────────────────


class SocketServer(threading.Thread):
    """Background daemon that accepts TCP connections and forwards data via signals.

    Protocol — one JSON object per line:
        Right-panel value : {"channel": "name", "value": 1.23}
        Add/update node   : {"type": "graph_node", "id": "n1",
                             "x": 0.0, "y": 0.0, "label": "optional"}
        Add edge          : {"type": "graph_edge", "from": "n1", "to": "n2"}
    """

    def __init__(self, bridge: _DataBridge, host: str = HOST, port: int = PORT) -> None:
        super().__init__(daemon=True, name="viz-socket-server")
        self._bridge = bridge
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
                        mtype = msg.get("type", "value")
                        if mtype == "graph_node":
                            self._bridge.node_received.emit(
                                str(msg["id"]),
                                float(msg.get("x", 0.0)),
                                float(msg.get("y", 0.0)),
                                str(msg.get("label", msg["id"])),
                            )
                        elif mtype == "graph_edge":
                            self._bridge.edge_received.emit(
                                str(msg["from"]), str(msg["to"])
                            )
                        else:
                            self._bridge.value_received.emit(
                                str(msg["channel"]), float(msg["value"])
                            )
                    except (KeyError, ValueError, json.JSONDecodeError) as exc:
                        print(f"[server] parse error ({exc}): {raw!r}", flush=True)


# ── Left panel: interactive directed graph ─────────────────────────────────────


class DirectedGraphPanel(pg.PlotWidget):
    """pg.PlotWidget showing a directed graph with hover + click interaction.

    Nodes   — ScatterPlotItem spots (hoverable=True, per-spot data = node id).
    Edges   — PlotDataItem lines.
    Arrows  — ArrowItem at 70% along each edge, pointing toward the destination.
    Labels  — TextItem centered on each node, drawn above the scatter layer.

    Hover  → node highlights blue  (sigHovered on the ScatterPlotItem).
    Click  → node toggles purple selection (click again to deselect).
    """

    sig_node_hovered = Signal(str)   # node id, or '' when cursor leaves all nodes
    sig_node_clicked = Signal(str)   # node id

    def __init__(self) -> None:
        super().__init__(background=_BG)
        self.setTitle("Directed Graph", color=_FG, size="11pt")
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.getViewBox().setMouseEnabled(x=True, y=True)

        self._nodes: dict[str, dict] = {}          # id → {"pos": (x,y), "label": str}
        self._edges: list[tuple[str, str]] = []

        # Static topology items (recreated on graph changes)
        self._edge_items:  list[pg.PlotDataItem] = []
        self._arrow_items: list[pg.ArrowItem] = []
        self._label_items: dict[str, pg.TextItem] = {}

        # Dynamic node layer (updated cheaply on hover/click)
        self._scatter = pg.ScatterPlotItem(hoverable=True)
        self._scatter.sigClicked.connect(self._on_clicked)
        self._scatter.sigHovered.connect(self._on_hovered)
        self.addItem(self._scatter)

        self._hovered:  str | None = None
        self._selected: str | None = None

        self._load_example()

    # ── Example DAG ───────────────────────────────────────────────────────

    def _load_example(self) -> None:
        for nid, pos, label in [
            ("input",  ( 0.0,  0.0), "input"),
            ("proc_A", (-1.6, -1.6), "proc A"),
            ("proc_B", ( 1.6, -1.6), "proc B"),
            ("merge",  ( 0.0, -3.2), "merge"),
            ("output", ( 0.0, -4.8), "output"),
        ]:
            self._nodes[nid] = {"pos": pos, "label": label}
        self._edges = [
            ("input",  "proc_A"),
            ("input",  "proc_B"),
            ("proc_A", "merge"),
            ("proc_B", "merge"),
            ("merge",  "output"),
        ]
        self._rebuild_topology()

    # ── Public API (also called from socket) ──────────────────────────────

    def add_node(self, nid: str, x: float, y: float, label: str = "") -> None:
        self._nodes[nid] = {"pos": (x, y), "label": label or nid}
        self._rebuild_topology()

    def add_edge(self, from_id: str, to_id: str) -> None:
        if (from_id, to_id) not in self._edges:
            self._edges.append((from_id, to_id))
            self._rebuild_topology()

    # ── Rendering: topology ───────────────────────────────────────────────

    def _rebuild_topology(self) -> None:
        """Recreate edges, arrowheads, and labels.  Does NOT touch the scatter."""
        for item in self._edge_items:
            self.removeItem(item)
        for item in self._arrow_items:
            self.removeItem(item)
        for item in self._label_items.values():
            self.removeItem(item)
        self._edge_items.clear()
        self._arrow_items.clear()
        self._label_items.clear()

        node_pos = {nid: info["pos"] for nid, info in self._nodes.items()}

        for src, dst in self._edges:
            if src not in node_pos or dst not in node_pos:
                continue
            sx, sy = node_pos[src]
            dx, dy = node_pos[dst]

            line = self.plot([sx, dx], [sy, dy],
                             pen=pg.mkPen(_OVERLAY0, width=1.5))
            self._edge_items.append(line)

            # Arrowhead at 70 % of the edge.
            # ArrowItem angle=0 → right (+x).  QPainter.rotate() is clockwise,
            # so we negate the standard CCW atan2 angle.
            t = 0.70
            ax = sx + t * (dx - sx)
            ay = sy + t * (dy - sy)
            angle = -math.degrees(math.atan2(dy - sy, dx - sx))
            arrow = pg.ArrowItem(
                pos=(ax, ay), angle=angle,
                headLen=14, tailLen=0, tipAngle=28, baseAngle=12,
                brush=pg.mkBrush(_OVERLAY0), pen=pg.mkPen(None),
            )
            self.addItem(arrow)
            self._arrow_items.append(arrow)

        for nid, info in self._nodes.items():
            x, y = info["pos"]
            lbl = pg.TextItem(text=info["label"], color=_FG, anchor=(0.5, 0.5))
            lbl.setPos(x, y)
            lbl.setZValue(10)
            self.addItem(lbl)
            self._label_items[nid] = lbl

        self._update_node_styles()

    # ── Rendering: node styles (cheap) ────────────────────────────────────

    def _update_node_styles(self) -> None:
        spots = []
        for nid, info in self._nodes.items():
            if nid == self._selected:
                brush, pen, size = pg.mkBrush(_MAUVE), pg.mkPen(_PINK, width=3), 36
            elif nid == self._hovered:
                brush, pen, size = pg.mkBrush(_BLUE), pg.mkPen(_FG, width=2.5), 34
            else:
                brush, pen, size = pg.mkBrush(_SURFACE0), pg.mkPen(_FG, width=1.5), 30
            spots.append({
                "pos": info["pos"], "brush": brush,
                "pen": pen, "size": size, "data": nid,
            })
        self._scatter.setData(spots)

    # ── Interaction ───────────────────────────────────────────────────────

    def _on_hovered(self, _scatter, spots, _ev) -> None:
        nid = spots[0].data() if spots else None
        if nid != self._hovered:
            self._hovered = nid
            self._update_node_styles()
            self.sig_node_hovered.emit(nid or "")

    def _on_clicked(self, _scatter, spots, _ev) -> None:
        if not spots:
            return
        nid = spots[0].data()
        self._selected = None if nid == self._selected else nid
        self._update_node_styles()
        self.sig_node_clicked.emit(nid)


# ── Right panel: rolling time-series ──────────────────────────────────────────


class LivePlotPanel(pg.PlotWidget):
    """Rolling per-channel plots fed by a QTimer, mouse moves, and socket data."""

    def __init__(self) -> None:
        super().__init__(background=_BG)
        self.setTitle("Live Channels", color=_FG, size="11pt")
        self.setLabel("bottom", "Sample index", color=_FG)
        self.setLabel("left", "Value", color=_FG)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.addLegend(offset=(10, 10))
        self.setMouseTracking(True)

        self._buffers: dict[str, deque[float]] = {}
        self._curves:  dict[str, pg.PlotDataItem] = {}
        self._phase:   float = 0.0
        self._last_mouse_t: float = 0.0

        for name in ("timer_sin", "timer_cos", "mouse_x", "mouse_y"):
            self._ensure_channel(name)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(RENDER_INTERVAL_MS)

    def _ensure_channel(self, name: str) -> None:
        if name in self._buffers:
            return
        color = _CHANNEL_COLORS[len(self._buffers) % len(_CHANNEL_COLORS)]
        self._buffers[name] = deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)
        self._curves[name]  = self.plot(
            np.zeros(BUFFER_LEN), name=name,
            pen=pg.mkPen(color=color, width=1.8),
        )

    def on_socket_data(self, channel: str, value: float) -> None:
        self._ensure_channel(channel)
        self._buffers[channel].append(value)

    def _tick(self) -> None:
        self._phase += RENDER_INTERVAL_MS / 1000.0
        self._buffers["timer_sin"].append(
            math.sin(2 * math.pi * TIMER_FREQ_HZ * self._phase)
        )
        self._buffers["timer_cos"].append(
            math.cos(2 * math.pi * TIMER_FREQ_HZ * self._phase)
        )
        for name, curve in self._curves.items():
            curve.setData(np.fromiter(self._buffers[name], dtype=float))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        now = time.monotonic()
        if (now - self._last_mouse_t) * 1000.0 >= MOUSE_THROTTLE_MS:
            self._last_mouse_t = now
            w, h = max(self.width(), 1), max(self.height(), 1)
            pos = event.position()
            self._buffers["mouse_x"].append(pos.x() / w * 2.0 - 1.0)
            self._buffers["mouse_y"].append(pos.y() / h * 2.0 - 1.0)
        super().mouseMoveEvent(event)


# ── Main window ────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PyQtGraph Live  |  socket {HOST}:{PORT}")
        self.resize(1280, 700)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self.graph_panel = DirectedGraphPanel()
        self.plot_panel  = LivePlotPanel()
        splitter.addWidget(self.graph_panel)
        splitter.addWidget(self.plot_panel)
        splitter.setSizes([480, 800])

        self.statusBar().showMessage(
            f"Hover nodes to highlight  |  click to select  |  socket on {HOST}:{PORT}"
        )
        self.graph_panel.sig_node_hovered.connect(self._on_node_hovered)
        self.graph_panel.sig_node_clicked.connect(self._on_node_clicked)

    def _on_node_hovered(self, nid: str) -> None:
        if nid:
            self.statusBar().showMessage(f"Hovering: {nid}")
        else:
            self.statusBar().showMessage(
                f"Hover nodes to highlight  |  click to select  |  socket on {HOST}:{PORT}"
            )

    def _on_node_clicked(self, nid: str) -> None:
        self.statusBar().showMessage(f"Selected: {nid}  (click again to deselect)")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    bridge = _DataBridge()
    SocketServer(bridge).start()

    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background=_BG, foreground=_FG)

    win = MainWindow()
    bridge.value_received.connect(win.plot_panel.on_socket_data)
    bridge.node_received.connect(
        lambda nid, x, y, label: win.graph_panel.add_node(nid, x, y, label)
    )
    bridge.edge_received.connect(win.graph_panel.add_edge)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
