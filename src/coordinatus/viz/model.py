"""VisualizerModel — domain data and interaction state for the MVP visualizer.

No Qt or pyqtgraph imports. All coordinate conversions use the coordinatus API.
"""

from __future__ import annotations

from math import atan2, degrees
from typing import Any

import networkx as nx
import numpy as np

from coordinatus.coordinate import Point
from coordinatus.space import Space, Space2D
from coordinatus.transforms import translate2D
from coordinatus.transforms.rotate import rotate2D
from coordinatus.transforms.scale import scale2D
from coordinatus.viz.scene import (
    ArrowSpec,
    CurveSpec,
    FilledPolygonSpec,
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
    Messages must be produced by ``coordinatus.serializer.to_json``.
    A Scene snapshot is produced by ``to_scene()`` on the Qt thread.

    ``spaces`` is a flat ``list[Space]`` and ``coordinates`` is a flat
    ``list[Coordinate]``; both are replaced on every ``apply_message`` call.
    """

    def __init__(self) -> None:
        # The single shared root for all scene-graph spaces.
        self._implicit_root: Space2D = Space2D()

        # Domain data
        self.spaces: list[Space] = []
        self._space_parent_uids: dict[str, str | None] = {}  # space_id → parent_id
        self.coordinates: list[Point] = []

        # Interaction / camera state
        self.view_space: Space = Space2D(parent=self._implicit_root)
        self.hovered_node_id: str | None = None
        self.selected_node_id: str | None = None
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0

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
             "coordinates": [{"space_id": ..., "coords": [...]}, ...]}
        """
        self.spaces.clear()
        self._space_parent_uids.clear()
        self.coordinates.clear()

        from coordinatus.serializer import from_json
        self.spaces, self.coordinates = from_json(msg)

        for space in self.spaces:
            if space.parent is None:
                space.parent = self._implicit_root
            self._space_parent_uids[space.uid] = space.parent.uid

        # Re-anchor view_space to the newly created Space objects.
        if self.selected_node_id is not None:
            new_spaces_dict = {s.uid: s for s in self.spaces}
            if self.selected_node_id in new_spaces_dict:
                self.view_space.parent = new_spaces_dict[self.selected_node_id]
            else:
                self.selected_node_id = None
                self.view_space = Space2D(parent=self._implicit_root)


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
        spaces_dict = {space.uid: space for space in self.spaces}
        if self.selected_node_id in spaces_dict:
            parent = spaces_dict[self.selected_node_id]
        else:
            parent = self._implicit_root
        self.view_space = Space2D(parent=parent)

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
    # Private helpers — to_scene
    # ---------------------------------------------------------------------- #

    def _build_graph_scene(self) -> GraphScene:
        curves: list[CurveSpec] = []
        arrows: list[ArrowSpec] = []
        scatter_positions: list[list[float]] = []
        scatter_colors: list[str] = []
        scatter_sizes: list[float] = []
        scatter_ids: list[str] = []
        labels: list[LabelSpec] = []

        # Build directed graph and compute layout positions via networkx.
        G: nx.DiGraph = nx.DiGraph()
        for space in self.spaces:
            G.add_node(space.uid)
        for space_id, parent_id in self._space_parent_uids.items():
            if parent_id is not None:
                G.add_edge(parent_id, space_id)
            else:
                print()
        
        pos: dict[str, tuple[float, float]] = nx.spring_layout(G, seed=42)

        # 5.6.1 — Space origins → graph nodes (layout coords)
        for space in self.spaces:
            if space.uid not in pos:
                continue
            xy = pos[space.uid]

            if space.uid == self.selected_node_id:
                color = _COLOR_NODE_SELECTED
            elif space.uid == self.hovered_node_id:
                color = _COLOR_NODE_HOVERED
            else:
                color = _COLOR_NODE_DEFAULT

            scatter_positions.append([float(xy[0]), float(xy[1])])
            scatter_colors.append(color)
            scatter_sizes.append(_NODE_SIZE_DEFAULT)
            scatter_ids.append(space.uid)

            labels.append(
                LabelSpec(
                    x=float(xy[0]),
                    y=float(xy[1]),
                    text=space.uid,
                    color=color,
                    rotation=0.0,
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
        space_uids = {s.uid for s in self.spaces}
        for space in self.spaces:
            parent_id = self._space_parent_uids.get(space.uid)
            if parent_id is None or parent_id not in space_uids:
                continue
            if parent_id not in pos or space.uid not in pos:
                continue

            p0 = np.array([float(pos[parent_id][0]), float(pos[parent_id][1])])
            p1 = np.array([float(pos[space.uid][0]), float(pos[space.uid][1])])

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
                apex = p0 + 0.70 * diff
                d = diff / length
                angle = degrees(atan2(float(d[1]), float(d[0])))
                arrows.append(
                    ArrowSpec(
                        x=float(apex[0]),
                        y=float(apex[1]),
                        angle=angle,
                        size=0.06 * length,
                        color=_COLOR_EDGE,
                    )
                )

        return GraphScene(curves=curves, arrows=arrows, scatter=scatter, labels=labels)

    def _build_plot_scene(self) -> PlotScene:
        # Todo: from the list of spaces and points, 
        # build curves and polygon to draw arrows of each spaces unit axes, relative to the view space. 
        # Then generate scatters from points, also rendered relative to the view_spaces

        curves = []
        polygons = []
        scatters = []

        arrow_head_size = 0.1
        arrow_body_coords = np.array([
            [0.0, 0.0],
            [1.0 - arrow_head_size, 0.0]
        ]).transpose()
        arrow_head_coords = np.array([
            [1.0, 0.0], 
            [1-arrow_head_size, -arrow_head_size/3], 
            [1-arrow_head_size, arrow_head_size/3]]).transpose()

        # unit axes of each space, transformed to view space
        for space in self.spaces:
            origin = Point(coords=np.array([0.0, 0.0]), space=space)
            x_axis_subspace = Space2D(parent=space)
            y_axis_subspace = Space(transform=rotate2D(-np.pi/2) @ scale2D(-1, 1), parent=space)

            scatters.append(
                ScatterSpec(
                    positions=origin.relative_to(self.view_space).coords.reshape(-1, 2),
                    colors=['blue'],
                    sizes=[0.1],
                    ids=[f"{space.uid}_origin"]
                )
            )

            for axis_space in [x_axis_subspace, y_axis_subspace]:
                arrow_body = Point(coords=arrow_body_coords, space=axis_space).relative_to(self.view_space)

                curves.append(
                    CurveSpec(points=arrow_body.coords, color='red', width=2.0)
                )

                arrow_head = Point(coords=arrow_head_coords, space=axis_space).relative_to(self.view_space)
                polygons.append(
                    FilledPolygonSpec(vertices=arrow_head.coords, color='green', border_color='white')
                )

        return PlotScene(curves=curves, polygons=polygons, scatter=scatters)
