"""VisualizerModel — domain data and interaction state for the MVP visualizer.

No Qt or pyqtgraph imports. All coordinate conversions use the coordinatus API.
"""

from __future__ import annotations

from math import atan2, degrees
from typing import Any

import numpy as np

from coordinatus.coordinate import Point
from coordinatus.space import Space, Space2D
from coordinatus.transforms import translate2D
from coordinatus.transforms.scale import scale2D
from coordinatus.viz.scene import (
    ArrowSpec,
    CurveSpec,
    GraphScene,
    LabelSpec,
    PlotScene,
    ScatterSpec,
    Scene,
)

# --------------------------------------------------------------------------- #
# Visual style constants
# --------------------------------------------------------------------------- #

_COLOR_NODE_DEFAULT = "#808080"
_COLOR_NODE_HOVERED = "#4488ff"
_COLOR_NODE_SELECTED = "#aa44ff"
_COLOR_EDGE = "#808080"

_NODE_SIZE_DEFAULT = 10.0

_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
]

# --------------------------------------------------------------------------- #
# VisualizerModel
# --------------------------------------------------------------------------- #


class VisualizerModel:
    """Domain data and interaction state for the MVP visualizer.

    Populated via ``apply_message(msg)`` from decoded socket dicts.
    A Scene snapshot is produced by ``to_scene()`` on the Qt thread.
    """

    def __init__(self) -> None:
        # The single shared root for all scene-graph spaces.
        self._implicit_root: Space2D = Space2D()

        # Domain data
        self.spaces: dict[str, Space] = {}
        self._space_parent_ids: dict[str, str | None] = {}  # space_id → parent_id
        self.point_channels: dict[str, list[Point]] = {}

        # Interaction / camera state
        self.view_space: Space = Space(transform=np.eye(3), parent=self._implicit_root)
        self.hovered_node_id: str | None = None
        self.selected_node_id: str | None = None
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0

        # Per-channel color assignment
        self._channel_colors: dict[str, str] = {}
        self._color_index: int = 0

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def apply_message(self, msg: dict[str, Any]) -> None:
        """Mutate domain data from one decoded socket message. Not thread-safe.

        A state-update message carries the **complete** current set of spaces
        and/or points under the ``"spaces"`` and ``"points"`` keys; these replace
        all previously stored data.

        Message shapes
        --------------
        Full state update::

            {"spaces": [{"id": ..., "parent_id": ..., "transform": ...}, ...],
             "points": [{"channel": ..., "space_id": ..., "coords": [...]}, ...]}

        Display-space override::

            {"type": "set_display_space", "space_id": "world"}
        """
        if "spaces" in msg:
            self.spaces.clear()
            self._space_parent_ids.clear()
            self._resolve_spaces(msg["spaces"])
            # Re-anchor view_space to the rebuilt Space object (preserves pan/zoom).
            if self.selected_node_id is not None and self.selected_node_id in self.spaces:
                self.view_space = Space(
                    transform=self.view_space.transform,
                    parent=self.spaces[self.selected_node_id],
                )
            elif self.selected_node_id is not None:
                # Selected space no longer exists in the new state.
                self.selected_node_id = None
                self.view_space = Space(
                    transform=self.view_space.transform,
                    parent=self._implicit_root,
                )

        if "points" in msg:
            self.point_channels.clear()
            for pts_msg in msg["points"]:
                self._apply_points(pts_msg)

        if msg.get("type") == "set_display_space":
            self.select_space(msg.get("space_id"))

    def to_scene(self) -> Scene:
        """Build and return a full Scene snapshot. Called on the Qt thread."""
        return Scene(
            graph=self._build_graph_scene(),
            plot=self._build_plot_scene(),
        )

    def select_space(self, space_id: str | None) -> None:
        """Set the selected space and reset view_space to identity within it.

        If ``space_id`` is None or unknown the view reverts to the implicit root.
        """
        self.selected_node_id = space_id
        if space_id is not None and space_id in self.spaces:
            parent = self.spaces[space_id]
        else:
            parent = self._implicit_root
        self.view_space = Space(transform=np.eye(3), parent=parent)

    def set_interaction(self, **kwargs: Any) -> None:
        """Update hover / mouse state. Does NOT touch view_space.

        Accepted keys: ``hovered_node_id``, ``mouse_x``, ``mouse_y``.
        """
        if "hovered_node_id" in kwargs:
            self.hovered_node_id = kwargs["hovered_node_id"]
        if "mouse_x" in kwargs:
            self.mouse_x = float(kwargs["mouse_x"])
        if "mouse_y" in kwargs:
            self.mouse_y = float(kwargs["mouse_y"])

    def pan(self, dx: float, dy: float) -> None:
        """Translate view_space by (dx, dy) in current view-space units.

        A right-drag (dx > 0) shifts all projected view coords by +dx, so the
        scene follows the cursor: ``T_new = T @ translate(-dx, -dy)``.
        """
        new_transform = self.view_space.transform @ translate2D(-dx, -dy)
        self.view_space = Space(transform=new_transform, parent=self.view_space.parent)

    def zoom(self, factor: float, cx: float, cy: float) -> None:
        """Scale view_space by ``factor`` around cursor position (cx, cy).

        ``factor > 1`` zooms in (objects appear larger).
        The point at (cx, cy) in view-space remains fixed.

        ``T_new = T @ translate(cx, cy) @ scale(1/factor) @ translate(-cx, -cy)``
        """
        t_center = translate2D(cx, cy)
        t_uncenter = translate2D(-cx, -cy)
        s = scale2D(1.0 / factor, 1.0 / factor)
        new_transform = self.view_space.transform @ t_center @ s @ t_uncenter
        self.view_space = Space(transform=new_transform, parent=self.view_space.parent)

    # ---------------------------------------------------------------------- #
    # Private helpers — apply_message
    # ---------------------------------------------------------------------- #

    def _resolve_spaces(self, space_defs: list[dict[str, Any]]) -> None:
        """Register all spaces in ``space_defs``, resolving parent references.

        Uses a local pending buffer to handle any declaration order within a
        single message.  Unresolvable references (genuine cycles or missing
        parents not present in the same message) are silently discarded.
        """
        pending: list[dict[str, Any]] = []
        for space_def in space_defs:
            self._try_register_space(space_def, pending)

        changed = True
        while changed and pending:
            changed = False
            next_pending: list[dict[str, Any]] = []
            for space_def in pending:
                pid: str | None = space_def.get("parent_id")
                if pid is None or pid in self.spaces:
                    self._register_space(space_def)
                    changed = True
                else:
                    next_pending.append(space_def)
            pending = next_pending

    def _try_register_space(
        self, space_def: dict[str, Any], pending: list[dict[str, Any]]
    ) -> None:
        pid: str | None = space_def.get("parent_id")
        if pid is None or pid in self.spaces:
            self._register_space(space_def)
        else:
            pending.append(space_def)

    def _register_space(self, space_def: dict[str, Any]) -> None:
        space_id: str = space_def["id"]
        parent_id: str | None = space_def.get("parent_id")
        transform = np.array(space_def["transform"], dtype=float)
        parent: Space = (
            self._implicit_root if parent_id is None else self.spaces[parent_id]
        )
        self.spaces[space_id] = Space(transform=transform, parent=parent)
        self._space_parent_ids[space_id] = parent_id

    def _apply_points(self, msg: dict[str, Any]) -> None:
        channel_id: str = str(msg["channel"])
        space_id: str = msg["space_id"]
        coords_list: list[list[float]] = msg["coords"]

        if space_id not in self.spaces:
            return  # space unknown — silently drop

        space = self.spaces[space_id]
        if channel_id not in self.point_channels:
            self.point_channels[channel_id] = []

        for coord in coords_list:
            self.point_channels[channel_id].append(
                Point(coords=np.array(coord, dtype=float), space=space)
            )

    # ---------------------------------------------------------------------- #
    # Private helpers — to_scene
    # ---------------------------------------------------------------------- #

    def _get_channel_color(self, channel_id: str) -> str:
        if channel_id not in self._channel_colors:
            self._channel_colors[channel_id] = _PALETTE[
                self._color_index % len(_PALETTE)
            ]
            self._color_index += 1
        return self._channel_colors[channel_id]

    def _build_graph_scene(self) -> GraphScene:
        curves: list[CurveSpec] = []
        arrows: list[ArrowSpec] = []
        scatter_positions: list[list[float]] = []
        scatter_colors: list[str] = []
        scatter_sizes: list[float] = []
        scatter_ids: list[str] = []
        labels: list[LabelSpec] = []

        # 5.6.1 — Space origins → graph nodes (absolute world coords)
        for space_id, space in self.spaces.items():
            try:
                origin_abs = Point([0.0, 0.0], space=space).to_absolute()
                xy = origin_abs.coords  # shape (2,)
            except Exception:  # noqa: BLE001
                continue

            if space_id == self.selected_node_id:
                color = _COLOR_NODE_SELECTED
            elif space_id == self.hovered_node_id:
                color = _COLOR_NODE_HOVERED
            else:
                color = _COLOR_NODE_DEFAULT

            scatter_positions.append([float(xy[0]), float(xy[1])])
            scatter_colors.append(color)
            scatter_sizes.append(_NODE_SIZE_DEFAULT)
            scatter_ids.append(space_id)

            # Label aligned with the space's local x-axis
            try:
                x_end_abs = Point([1.0, 0.0], space=space).to_absolute()
                x_end = x_end_abs.coords
                rotation = degrees(
                    atan2(
                        float(x_end[1]) - float(xy[1]),
                        float(x_end[0]) - float(xy[0]),
                    )
                )
            except Exception:  # noqa: BLE001
                rotation = 0.0

            labels.append(
                LabelSpec(
                    x=float(xy[0]),
                    y=float(xy[1]),
                    text=space_id,
                    color=color,
                    rotation=rotation,
                    anchor=(0.5, -0.3),
                )
            )

        scatter: list[ScatterSpec] = []
        if scatter_positions:
            scatter.append(
                ScatterSpec(
                    positions=np.array(scatter_positions),
                    colors=scatter_colors,
                    sizes=scatter_sizes,
                    ids=scatter_ids,
                )
            )

        # 5.6.2 — Hierarchy edges (parent→child lines + directed arrowheads)
        for space_id, space in self.spaces.items():
            parent_id = self._space_parent_ids.get(space_id)
            if parent_id is None or parent_id not in self.spaces:
                continue

            try:
                p0_abs = (
                    Point([0.0, 0.0], space=self.spaces[parent_id])
                    .to_absolute()
                    .coords
                )
                p1_abs = Point([0.0, 0.0], space=space).to_absolute().coords
            except Exception:  # noqa: BLE001
                continue

            p0 = np.array([float(p0_abs[0]), float(p0_abs[1])])
            p1 = np.array([float(p1_abs[0]), float(p1_abs[1])])

            curves.append(
                CurveSpec(
                    points=np.array([[p0[0], p0[1]], [p1[0], p1[1]]]),
                    color=_COLOR_EDGE,
                    width=1.2,
                )
            )

            diff = p1 - p0
            length = float(np.linalg.norm(diff))
            if length > 0:
                t = 0.70
                apex = p0 + t * diff
                d = diff / length
                angle = degrees(atan2(float(d[1]), float(d[0])))
                size = 0.06 * length
                arrows.append(
                    ArrowSpec(
                        x=float(apex[0]),
                        y=float(apex[1]),
                        angle=angle,
                        size=size,
                        color=_COLOR_EDGE,
                    )
                )

        return GraphScene(curves=curves, arrows=arrows, scatter=scatter, labels=labels)

    def _build_plot_scene(self) -> PlotScene:
        # Todo: from the list of spaces and points, 
        # build curves and polygon to draw arrows of each spaces unit axes, relative to the view space. 
        # Then generate scatters from points, also rendered relative to the view_spaces
        return PlotScene()
