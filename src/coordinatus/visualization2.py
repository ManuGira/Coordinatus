"""Two-panel PyQtGraph + PySide6 live application.

Left panel  — interactive directed graph of Space hierarchy:
              hover highlights nodes, click selects and sets the view origin.
Right panel — coordinates projected through the selected view_space.

Run:
    uv run python src/coordinatus/visualization2.py

Send spaces and points over TCP (newline-terminated JSON):
    {
        "spaces": [
            {"id": "world",  "parent_id": null,
             "transform": [[1,0,0],[0,1,0],[0,0,1]]},
            {"id": "sensor", "parent_id": "world",
             "transform": [[0.707,-0.707,1],[0.707,0.707,0],[0,0,1]]}
        ],
        "points": [
            {"channel": "lidar", "space_id": "sensor",
             "coords": [[1.0,2.0],[3.0,4.0]]}
        ]
    }

Set the view origin to a specific space:
    {"type": "set_display_space", "space_id": "world"}

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

import math
import queue
import sys

try:
    import pyqtgraph as pg  # noqa: F401 — imported here to fail early with a clear message
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv add pyqtgraph PySide6 --dev"
    ) from exc

from coordinatus.viz.model import VisualizerModel
from coordinatus.viz.presenter import Presenter
from coordinatus.viz.server import HOST, PORT, SocketServer
from coordinatus.viz.view import VisualizerView

# ── Example scene ──────────────────────────────────────────────────────────────
#
# Pre-loaded into the Model on startup so the panels are populated immediately.
# Three 2-D spaces arranged in a parent → child hierarchy:
#
#   world  (identity root at origin)
#     └─ robot   (translated +2, +1 and rotated 30° CCW w.r.t. world)
#          └─ sensor  (translated +1 along robot's x-axis)

_ANGLE = math.radians(30)
_COS, _SIN = math.cos(_ANGLE), math.sin(_ANGLE)

_EXAMPLE_MESSAGE: dict = {
    "spaces": [
        {
            "id": "world",
            "parent_id": None,
            "transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        {
            "id": "robot",
            "parent_id": "world",
            "transform": [
                [_COS, -_SIN, 2.0],
                [_SIN,  _COS, 1.0],
                [0,     0,    1.0],
            ],
        },
        {
            "id": "sensor",
            "parent_id": "robot",
            "transform": [[1, 0, 1.0], [0, 1, 0.0], [0, 0, 1.0]],
        },
    ],
    "points": [
        {
            "channel": "world_pts",
            "space_id": "world",
            "coords": [
                [0.5,  0.5],
                [1.0,  0.0],
                [0.0,  1.0],
                [-0.5, 0.5],
            ],
        },
        {
            "channel": "sensor_pts",
            "space_id": "sensor",
            "coords": [
                [0.2, 0.1],
                [0.4, 0.3],
                [0.6, 0.2],
            ],
        },
    ],
}


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    inbox: queue.Queue[dict] = queue.Queue()

    model = VisualizerModel()
    model.apply_message(_EXAMPLE_MESSAGE)

    app = QApplication(sys.argv)

    view = VisualizerView(host=HOST, port=PORT)
    presenter = Presenter(model, view, inbox)
    server = SocketServer(inbox, host=HOST, port=PORT)

    server.start()
    presenter.start()
    view.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
