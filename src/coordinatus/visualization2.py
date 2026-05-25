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

import numpy as np

import queue
import sys

from coordinatus.viz.model import VisualizerModel
from coordinatus.viz.presenter import Presenter
from coordinatus.viz.server import HOST, PORT, SocketServer
from coordinatus.viz.view import VisualizerView

from coordinatus.coordinate import Coordinate, Point
from coordinatus.space import Space, Space2D
from coordinatus.transforms import translate2D, rotate2D

try:
    import pyqtgraph as pg  # noqa: F401 — imported here to fail early with a clear message
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv add pyqtgraph PySide6 --dev"
    ) from exc



# ── Example scene ──────────────────────────────────────────────────────────────
#
# Pre-loaded into the Model on startup so the panels are populated immediately.
# Three 2-D spaces arranged in a parent → child hierarchy:
#
#   world  (identity root at origin)
#     └─ robot   (translated +2, +1 and rotated 30° CCW w.r.t. world)
#          └─ sensor  (translated +1 along robot's x-axis)



def generate_example_scene() -> tuple[list[Space], list[Coordinate]]:
    world_space = Space2D(uid="world")
    robot_space = Space(
        transform=translate2D(2, 1) @ rotate2D(np.pi/6),
        parent=world_space,
        uid="robot",
    )
    sensor_space = Space(
        transform=translate2D(1, 0),
        parent=robot_space,
        uid="sensor",
    )
    spaces = [world_space, robot_space, sensor_space]

    coordinates = [
        Point(
            space=world_space,
            coords=np.array([
                [0.5, 1.0, 0.0, -0.5],
                [0.5, 0.0, 1.0, 0.5],
            ]),
        ),
        Point(
            space=sensor_space,
            coords=np.array([
                [0.2, 0.4, 0.6],
                [0.1, 0.3, 0.2],
            ]),
        ),
    ]
    return spaces, coordinates

def main() -> None:
    from coordinatus.serializer import to_json

    inbox: queue.Queue[dict] = queue.Queue()

    model = VisualizerModel()

    example_spaces, example_coordinates = generate_example_scene()
    example_message = to_json(example_spaces, example_coordinates)
    model.apply_message(example_message)

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
