"""Minimal demo: VisualizerModel + VisualizerView wired with a QTimer.

No Presenter yet — this exercises the new viz/view.py directly.
Starts the SocketServer so live messages can also be injected while running.

Run:
    uv run python examples/viz_view_demo.py

Send data (see examples/pyqtgraph_socket_sender.py for full examples):
    {"spaces": [{"id": "world", "parent_id": null, "transform": [[1,0,0],[0,1,0],[0,0,1]]}]}
"""

from __future__ import annotations

import math
import queue
import sys

import numpy as np

try:
    import pyqtgraph as pg
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv add pyqtgraph PySide6 --dev"
    ) from exc

from coordinatus.transforms import translate2D
from coordinatus.transforms.rotate import rotate2D
from coordinatus.transforms.scale import scale2D
from coordinatus.viz.model import VisualizerModel
from coordinatus.viz.server import SocketServer
from coordinatus.viz.view import VisualizerView

# ── Simple event handler ───────────────────────────────────────────────────────


class _DemoHandler:
    """Logs events to stdout and delegates interaction to the Model."""

    def __init__(self, model: VisualizerModel) -> None:
        self._model = model

    def on_node_hovered(self, node_id: str | None) -> None:
        self._model.set_interaction(hovered_node_id=node_id)

    def on_node_clicked(self, node_id: str) -> None:
        new_id = None if node_id == self._model.selected_node_id else node_id
        self._model.select_space(new_id)
        print(f"[demo] selected: {new_id!r}")

    def on_plot_pan(self, dx: float, dy: float) -> None:
        self._model.pan(dx, dy)

    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None:
        self._model.zoom(factor, cx, cy)

    def on_mouse_moved(self, norm_x: float, norm_y: float) -> None:
        self._model.set_interaction(mouse_x=norm_x, mouse_y=norm_y)


# ── Example scene ──────────────────────────────────────────────────────────────


def _example_message() -> dict:
    """Build a compound state-update message with a small space hierarchy
    and a few point channels."""

    def mat(tx: float = 0.0, ty: float = 0.0,
            angle: float = 0.0, sx: float = 1.0, sy: float = 1.0) -> list:
        T = translate2D(tx, ty)
        R = rotate2D(angle)
        S = scale2D(sx, sy)
        return (T @ R @ S).tolist()

    spaces = [
        {"id": "world",    "parent_id": None,      "transform": mat()},
        {"id": "sensor_A", "parent_id": "world",   "transform": mat(tx=2.0, ty=1.0,
                                                                     angle=math.radians(30))},
        {"id": "sensor_B", "parent_id": "world",   "transform": mat(tx=-2.0, ty=1.5,
                                                                     angle=math.radians(-45))},
        {"id": "child_A1", "parent_id": "sensor_A","transform": mat(tx=1.5, ty=0.5,
                                                                     sx=0.7, sy=0.7)},
    ]

    # Points in sensor_A's local frame
    rng = np.random.default_rng(42)
    pts_a = (rng.standard_normal((20, 2)) * 0.4).tolist()
    pts_b = (rng.standard_normal((15, 2)) * 0.3 + np.array([0.5, 0.0])).tolist()

    points = [
        {"channel": "lidar",   "space_id": "sensor_A", "coords": pts_a},
        {"channel": "targets", "space_id": "sensor_B", "coords": pts_b},
    ]

    return {"spaces": spaces, "points": points}


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    inbox: queue.Queue[dict] = queue.Queue()
    server = SocketServer(inbox)
    server.start()

    model = VisualizerModel()
    model.apply_message(_example_message())

    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    view = VisualizerView()
    handler = _DemoHandler(model)
    view.set_event_handler(handler)

    # Render loop: drain socket inbox, then render.
    def _tick() -> None:
        drained = 0
        while drained < 200:
            try:
                msg = inbox.get_nowait()
                model.apply_message(msg)
                drained += 1
            except queue.Empty:
                break
        view.display(model.to_scene())

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(16)

    # Initial render before showing (avoids blank flash).
    _tick()
    view.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
