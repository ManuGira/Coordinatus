"""Scene dataclasses — display snapshots passed from Model to View.

No Qt or pyqtgraph imports. Only numpy is required.
All coordinates are already in the target panel's units when stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CurveSpec:
    points: np.ndarray    # (N, 2) in panel coords
    color: str            # hex string
    width: float
    name: str = ""        # shown in plot legend; empty = omit from legend


@dataclass
class FilledPolygonSpec:
    vertices: np.ndarray  # (N, 2) in panel coords; closed automatically by the View
    color: str            # fill hex
    border_color: str | None = None


@dataclass
class ScatterSpec:
    positions: np.ndarray   # (N, 2)
    colors: list[str]       # per-point hex
    sizes: list[float]      # per-point px
    ids: list[str]          # per-point identifier passed back in click callbacks


@dataclass
class LabelSpec:
    x: float
    y: float
    text: str
    color: str
    rotation: float = 0.0                      # degrees CCW
    scale: float = 1.0                         # multiplier on default font size
    anchor: tuple[float, float] = (0.5, 0.5)


@dataclass
class ArrowSpec:
    x: float      # tip position in panel coords
    y: float
    angle: float  # direction the arrow points, degrees CCW from +x axis
    size: float   # arrowhead length in panel units
    color: str
    border_color: str | None = None


@dataclass
class GraphScene:
    """Flat rendering primitives for the left (spatial) panel.

    Coordinates are in absolute (root) world units — the pyqtgraph ViewBox
    handles its own viewport via native pan/zoom.
    The Model bakes in all semantic decisions (color, size, label text).
    The View has zero domain knowledge; it renders whatever is here.
    """

    curves: list[CurveSpec] = field(default_factory=list)
    arrows: list[ArrowSpec] = field(default_factory=list)
    scatter: list[ScatterSpec] = field(default_factory=list)
    labels: list[LabelSpec] = field(default_factory=list)


@dataclass
class PlotScene:
    """Flat rendering primitives for the right (plot) panel.

    Coordinates are already projected into view_space units.
    The View renders them directly with no further transformation.
    """

    curves: list[CurveSpec] = field(default_factory=list)
    polygons: list[FilledPolygonSpec] = field(default_factory=list)
    scatter: list[ScatterSpec] = field(default_factory=list)
    labels: list[LabelSpec] = field(default_factory=list)


@dataclass
class Scene:
    graph: GraphScene = field(default_factory=GraphScene)
    plot: PlotScene = field(default_factory=PlotScene)
