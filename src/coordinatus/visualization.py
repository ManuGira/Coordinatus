"""Visualization utilities for spaces and coordinates.

This module provides plotting functions to visualize spaces and points.
Requires matplotlib to be installed. The draw_space_hierarchy function
additionally requires networkx.

Install with: pip install coordinatus[plotting]
"""

from dataclasses import dataclass
from typing import Any, Optional, List, TYPE_CHECKING
import numpy as np

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

from coordinatus.transforms import scale

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from matplotlib.axes import Axes as _Axes
    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    _HAS_MATPLOTLIB = False
    _Axes = None  # type: ignore

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    _HAS_NETWORKX = False
    nx = None  # type: ignore

from .space import Space, Space2D 
from .transforms import rotate2D, scale2D, translate2D, trs2D, trks2D
from .coordinate import Point, Vector


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not _HAS_MATPLOTLIB:  # pragma: no cover
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install coordinatus[plotting]"
        )


def draw_space_axes(
    ax: 'Axes',  # type: ignore[name-defined]
    space: Optional[Space],
    reference_space: Optional[Space] = None,
    color: str = 'blue',
    label: str = 'Space',
    alpha: float = 0.5,
    highlight: bool = False,
) -> None:
    """Draw a space's origin and axes from a given reference space's perspective.
    
    Args:
        ax: Matplotlib axes to draw on.
        space: The space to draw. If None, draws the absolute/world space.
        reference_space: The space from which to view. If None, uses absolute coordinates.
        color: Color for the space axes and origin.
        label: Label prefix for the legend.
    
    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from coordinatus import Space, create_space
        >>> from coordinatus.visualization import draw_space_axes
        >>> 
        >>> fig, ax = plt.subplots()
        >>> space = create_space(None, tx=2, ty=1, angle_rad=np.pi/4)
        >>> draw_space_axes(ax, space, color='blue', label='MySpace')
        >>> plt.show()
    """
    _check_matplotlib()
    
    if reference_space is None:
        reference_space = Space2D() 

    # Use absolute space if space is None
    if space is None:
        space = Space2D(parent=reference_space)
    
    # Get space origin and unit vectors in reference space
    origin = Point(np.array([0, 0]), space=space)
    y_axis_space = Space(transform=rotate2D(-np.pi/2) @ scale2D(-1, 1), parent=space)
    
    def draw_text(ax, position_xy, direction_xy, text, color, alpha):
        import matplotlib.transforms as mtransforms
        size = np.linalg.norm(direction_xy)
        angle = np.arctan2(direction_xy[1], direction_xy[0])

        tp = TextPath(position_xy, text, size=size)
        transform = (
            mtransforms.Affine2D()
            .rotate_around(position_xy[0], position_xy[1], angle)
            + ax.transData
        )
        ax.add_patch(PathPatch(tp, color=color, alpha=alpha, transform=transform))

    
    def draw_arrow_10(ax: 'Axes', reference_space: Space, space: Space, label="", color="black", alpha=1):
        """ Use a poly line to draw an arrow with a head size relative to the vector length."""
            
        head_size = 0.1
        start = Point(np.array([0, 0]), space=space).relative_to(reference_space)
        end = Point(np.array([1-head_size, 0]), space=space).relative_to(reference_space)

        xs, ys = [start.coords[0], end.coords[0]], [start.coords[1], end.coords[1]]
        # First draw a single line for the arrow body
        ax.plot(xs, ys, color=color, alpha=alpha)

        head_points = Point(np.array([
            [1, 1-head_size, 1-head_size, 1],
            [0, head_size/3, -head_size/3, 0],
        ]), space=space).relative_to(reference_space)
        ax.fill(head_points.coords[0], head_points.coords[1], color=color, alpha=alpha)

        start_txt = Point(np.array([0.7, -0.1]), space=space).relative_to(reference_space)
        vector = (end-start)*0.05
        draw_text(ax, start_txt.coords, vector.coords, label, color, alpha)

    def draw_grid(ax: 'Axes', reference_space: Space, space: Space, color="gray", alpha=0.3):
        """Draw a grid of lines at every integer coordinate in the given space."""
        N = 2

        for x in range(-N, N+1):
            start = Point(np.array([x, -N-0.2]), space=space).relative_to(reference_space)
            end = Point(np.array([x, N+0.2]), space=space).relative_to(reference_space)
            ax.plot([start.coords[0], end.coords[0]], [start.coords[1], end.coords[1]], color=color, alpha=alpha)
        for y in range(-N, N+1):
            start = Point(np.array([-N-0.2, y]), space=space).relative_to(reference_space)
            end = Point(np.array([N+0.2, y]), space=space).relative_to(reference_space)
            ax.plot([start.coords[0], end.coords[0]], [start.coords[1], end.coords[1]], color=color, alpha=alpha)

    if highlight:
        draw_grid(ax, reference_space, space, color=color, alpha=alpha*0.3)
        alpha = 1

    origin_size = 0.02
    origin_points = Point(np.array([
            [origin_size, 0, -origin_size, 0, origin_size],
            [0, origin_size, 0, -origin_size, 0],
        ]), space=space).relative_to(reference_space)       
    ax.fill(origin_points.coords[0], origin_points.coords[1], color=color, alpha=alpha) 
    
    draw_text(
        ax, 
        (origin + np.array([-0.1, -0.1])).relative_to(reference_space).coords, 
        Vector(np.array([0.08, 0]), space=space).relative_to(reference_space).coords, 
        label, color, alpha)
    
    draw_arrow_10(ax, reference_space, space, color=color, alpha=alpha, label=f"X axis")
    draw_arrow_10(ax, reference_space, y_axis_space, color=color, alpha=alpha, label=f"Y axis")


    
def draw_points(
    ax: 'Axes',  # type: ignore[name-defined]
    points: List[Point],
    reference_space: Optional[Space] = None,
    color: str = 'red',
    label: str = 'Point',
    connect: bool = True,
    show_labels: bool = True
) -> None:
    """Draw points from a given reference space's perspective.
    
    Args:
        ax: Matplotlib axes to draw on.
        points: List of Point objects to draw.
        reference_space: The space from which to view. If None, uses absolute coordinates.
        color: Color for the points and connecting lines.
        label: Label prefix for point annotations.
        connect: If True, connects points with lines.
        show_labels: If True, shows point labels (P1, P2, etc.).
    
    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from coordinatus import Space, Point, create_space
        >>> from coordinatus.visualization import draw_points
        >>> 
        >>> fig, ax = plt.subplots()
        >>> space = create_space(None, tx=1, ty=1)
        >>> points = [
        ...     Point(np.array([0, 0]), space),
        ...     Point(np.array([1, 0]), space)
        ... ]
        >>> draw_points(ax, points, color='red')
        >>> plt.show()
    """
    _check_matplotlib()
    
    if not points:
        return
    
    # Get point coordinates in reference space
    if reference_space is None:
        coords = [p.to_absolute().coords for p in points]
    else:
        coords = [p.relative_to(reference_space).coords for p in points]
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    
    # Draw connecting lines
    if connect and len(points) > 1:
        ax.plot(xs, ys, '-', color=color, zorder=10, linewidth=2, alpha=0.5)
    
    # Draw points
    ax.plot(xs, ys, '.', color=color, zorder=10)
    
    # Add labels
    if show_labels:
        for i, (x, y) in enumerate(zip(xs, ys), 1):
            ax.text(x + 0.1, y + 0.1, f'{label} {i}',
                    fontsize=10, color=color, fontweight='bold')


# ── transition animation helpers ─────────────────────────────────────────────

_ANIM_FRAMES: int = 20
_ANIM_INTERVAL_MS: int = 16  # ~60 fps


def _decompose_trks2d(
    M: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Decompose a 3x3 2D affine matrix into (tx, ty, angle_rad, kx, sx, sy).

    Uses a QR-based convention (upper-triangular shear) so the result is unique
    for any invertible matrix, including those built with :func:`trks2D`.
    The ``ky`` component is always 0 by convention.

    Given M = T @ R @ [[1, kx], [0, 1]] @ [[sx, 0], [0, sy]]:

    - sx = \u2016col0\u2016  →  sqrt(M00^2 + M10^2)
    - theta = atan2(M10, M00)
    - sy = det(M[:2,:2]) / sx  (sign-preserving)
    - kx = dot(col0, col1) / (sx * sy)
    """
    tx, ty = float(M[0, 2]), float(M[1, 2])
    a, b, c, d = float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1])
    sx = float(np.sqrt(a ** 2 + c ** 2))
    angle = float(np.arctan2(c, a))
    sy = float((a * d - b * c) / sx)  # det / sx, preserves sign
    kx = float((a * b + c * d) / (sx * sy))
    return tx, ty, angle, kx, sx, sy


def _interpolate_trks2d(M0: np.ndarray, M1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two 3x3 2D affine matrices with smoothstep easing at *t* \u2208 [0, 1].

    Handles TRS and TRKS(kx, ky=0) matrices; uses QR decomposition so any
    matrix built with :func:`trs2D` or :func:`trks2D` round-trips correctly.
    """
    tx0, ty0, a0, kx0, sx0, sy0 = _decompose_trks2d(M0)
    tx1, ty1, a1, kx1, sx1, sy1 = _decompose_trks2d(M1)
    da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi  # shortest-path angle delta
    t_s = t * t * (3.0 - 2.0 * t)  # smoothstep easing
    return trks2D(
        tx=tx0 + t_s * (tx1 - tx0),
        ty=ty0 + t_s * (ty1 - ty0),
        angle_rad=a0 + t_s * da,
        kx=kx0 + t_s * (kx1 - kx0),
        ky=0.0,
        sx=sx0 + t_s * (sx1 - sx0),
        sy=sy0 + t_s * (sy1 - sy0),
    )


# ── helpers for draw_space_hierarchy ────────────────────────────────────────

@dataclass
class _HierarchyRenderData:
    """Bundled, precomputed data for rendering a Space hierarchy."""
    spaces: list
    labels: dict          # {id(space): display_label}
    graph: Any            # nx.DiGraph (typed as Any to avoid import-time errors)
    pos: dict             # {node_id: (x, y)} layout positions
    node_labels: dict     # {node_id: display_label}
    colors: dict          # {node_id: hex_color}
    reference_space: "Space | None"  # root space for the axes subplot; None = absolute


def _build_digraph(spaces: list) -> Any:
    """Build a directed graph (edge: parent → child) from a list of Space objects."""
    G = nx.DiGraph()
    for s in spaces:
        G.add_node(id(s))
        if s.parent is not None:
            G.add_edge(id(s.parent), id(s))
    return G


def _tree_pos(
    graph, root, width: float = 1.0, vert_gap: float = 1.0,
    x_start: float = 0.0, depth: int = 0,
    pos: dict | None = None, parent=None,
) -> dict:
    """Recursively assign (x, y) positions for a single subtree (top-down layout)."""
    if pos is None:
        pos = {}
    children = [n for n in graph.successors(root) if n != parent]
    if not children:
        pos[root] = (x_start, -depth * vert_gap)
    else:
        dx = width / len(children)
        next_x = x_start - width / 2 + dx / 2
        for child in children:
            _tree_pos(graph, child, width=dx, vert_gap=vert_gap,
                      x_start=next_x, depth=depth + 1, pos=pos, parent=root)
            next_x += dx
        pos[root] = (x_start, -depth * vert_gap)
    return pos


def _compute_layout(G) -> dict:
    """Compute a top-down hierarchical layout for all subtrees in *G*."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    pos = {}
    x_offset = 0.0
    for root in roots:
        subtree_nodes = list(nx.dfs_preorder_nodes(G, root))
        leaf_count = max(1, sum(1 for n in subtree_nodes if G.out_degree(n) == 0))
        subtree_pos = _tree_pos(G, root, width=float(leaf_count),
                                x_start=x_offset + leaf_count / 2)
        pos.update(subtree_pos)
        x_offset += leaf_count + 1
    return pos


def _make_color_map(spaces: list) -> dict:
    """Return ``{id(space): hex_color}`` cycling through the *tab10* palette."""
    import matplotlib
    import matplotlib.colors as mcolors
    cmap = matplotlib.colormaps["tab10"]
    return {id(s): mcolors.to_hex(cmap(i % 10)) for i, s in enumerate(spaces)}


def _find_reference_space(spaces: list, G) -> "Space | None":
    """Return the single root space when there is exactly one, otherwise *None*."""
    id_to_space = {id(s): s for s in spaces}
    roots = [id_to_space[n] for n in G.nodes if G.in_degree(n) == 0 and n in id_to_space]
    return roots[0] if len(roots) == 1 else None


def _build_hierarchy_render_data(spaces: list, labels: dict) -> _HierarchyRenderData:
    """Assemble all precomputed data needed by the two draw_space_hierarchy subplots."""
    G = _build_digraph(spaces)
    pos = _compute_layout(G)
    return _HierarchyRenderData(
        spaces=spaces,
        labels=labels,
        graph=G,
        pos=pos,
        node_labels={n: labels.get(n, str(n)) for n in G.nodes},
        colors=_make_color_map(spaces),
        reference_space=_find_reference_space(spaces, G),
    )


def _compute_figure_size(pos: dict) -> tuple[float, float]:
    """Estimate a good panel (width, height) from the layout *pos* dict."""
    if not pos:
        return 4.0, 3.0
    depth = max(-y for _, y in pos.values())
    x_span = max(x for x, _ in pos.values()) - min(x for x, _ in pos.values()) + 2
    return max(4.0, x_span * 1.8), max(3.0, (depth + 1) * 1.2)


def _draw_hierarchy_subplot(
    ax, data: _HierarchyRenderData, selected_node=None, hovered_node=None
) -> None:
    """Draw the networkx directed graph (parent → child) on *ax*."""
    node_color_list = [data.colors[n] for n in data.graph.nodes]
    edgecolors = []
    linewidths = []
    for n in data.graph.nodes:
        if n == selected_node:
            edgecolors.append("black")
            linewidths.append(3.0)
        elif n == hovered_node:
            edgecolors.append("grey")
            linewidths.append(2.0)
        else:
            edgecolors.append("none")
            linewidths.append(1.0)
    nx.draw(
        data.graph, data.pos, ax=ax,
        labels=data.node_labels,
        with_labels=True,
        node_size=1800,
        node_color=node_color_list,
        font_color="white",
        font_size=9,
        arrows=True,
        arrowsize=18,
        edge_color="#888888",
        width=2,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    ax.set_title("Hierarchy (click a node to set as reference)")


def _draw_axes_subplot(
    ax,
    data: _HierarchyRenderData,
    hovered_node: int | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    reference_space_override: "Space | None" = None,
) -> "dict[int, list]":
    """Draw every space's coordinate frame on *ax* using :func:`draw_space_axes`.

    *reference_space_override* is used during animated transitions to supply a
    temporary interpolated space without mutating *data.reference_space*.

    Returns a mapping ``{id(space): [artists]}`` so callers can update
    individual spaces without a full clear+redraw cycle.
    """
    effective_ref = (
        reference_space_override if reference_space_override is not None
        else data.reference_space
    )
    space_artists: dict[int, list] = {}
    for space in data.spaces:
        color = data.colors[id(space)]
        label = data.labels[id(space)]
        # No highlight during transitions (reference_space_override is set).
        is_reference = (
            reference_space_override is None
            and data.reference_space is not None
            and space is data.reference_space
        )
        is_hovered = hovered_node is not None and id(space) == hovered_node
        n_lines = len(ax.lines)
        n_patches = len(ax.patches)
        draw_space_axes(ax, space, reference_space=effective_ref,
                        color=color, label=label, highlight=is_reference or is_hovered)
        space_artists[id(space)] = list(ax.lines[n_lines:]) + list(ax.patches[n_patches:])
    ref_name = (
        data.labels.get(id(data.reference_space), "root")
        if data.reference_space is not None
        else "absolute"
    )
    ax.set_aspect("equal")
    ax.set_xlim(xlim if xlim is not None else (-2.5, 2.5))
    ax.set_ylim(ylim if ylim is not None else (-2.5, 2.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Coordinate axes (in {ref_name} space)")
    return space_artists


class _HierarchyInteractor:
    """Manages all interactive state for :func:`draw_space_hierarchy`.

    Handles:
    - Left-click on either panel → change the reference space.
    - Hover over either panel → highlight the space under the cursor.
    - Left-click drag on the axes panel → pan.
    - Scroll wheel on the axes panel → zoom centred on the cursor.
    """

    def __init__(
        self,
        fig,
        ax_graph,
        ax_axes,
        data: _HierarchyRenderData,
        title: str,
        id_to_space: dict,
    ) -> None:
        self.fig = fig
        self.ax_graph = ax_graph
        self.ax_axes = ax_axes
        self.data = data
        self.title = title
        self.id_to_space = id_to_space

        self.selected_node: int | None = (
            id(data.reference_space) if data.reference_space is not None else None
        )
        self.hovered_node: int | None = None

        # Per-space artist tracking for fast hover updates.
        self._space_artists: dict[int, list] = {}
        self._prev_hovered: int | None = None

        # Pan state
        self._pan_start_display: tuple[float, float] | None = None
        self._pan_xlim: tuple[float, float] | None = None
        self._pan_ylim: tuple[float, float] | None = None
        self._pan_inv_transform = None  # captured at press time to avoid drift

        # Animation state
        self._anim_timer = None
        self._anim_frame: int = 0
        self._anim_M_old: np.ndarray | None = None
        self._anim_M_new: np.ndarray | None = None
        self._anim_root: Space | None = None
        self._anim_R_root_inv: np.ndarray | None = None
        self._anim_bg = None  # captured pixel background for blitting

        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        fig.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)

        # Initial draw — must happen after event connections so the canvas exists.
        self._redraw()

    # ── drawing helpers ────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """Full redraw: resets the axes view to [-2.5, 2.5]."""
        self.ax_graph.clear()
        _draw_hierarchy_subplot(
            self.ax_graph, self.data,
            selected_node=self.selected_node,
            hovered_node=self.hovered_node,
        )
        self.ax_axes.clear()
        self._space_artists = _draw_axes_subplot(
            self.ax_axes, self.data, hovered_node=self.hovered_node
        )
        self.fig.suptitle(self.title)
        self.fig.canvas.draw_idle()
        self._prev_hovered = self.hovered_node

    def _redraw_hover(self) -> None:
        """Fast hover update: only redraws the spaces whose highlight state changed.

        The graph subplot is cleared and redrawn (fast — few nodes).
        The axes subplot is updated in-place: only the artists belonging to the
        previously-hovered and newly-hovered spaces are removed and redrawn,
        so the rest of the scene is untouched and no ``ax.clear()`` is needed.
        """
        xlim = self.ax_axes.get_xlim()
        ylim = self.ax_axes.get_ylim()

        # Graph subplot: clear + redraw (fast — few nodes).
        self.ax_graph.clear()
        _draw_hierarchy_subplot(
            self.ax_graph, self.data,
            selected_node=self.selected_node,
            hovered_node=self.hovered_node,
        )

        # Axes subplot: swap only the spaces whose highlight state changed.
        prev = self._prev_hovered
        curr = self.hovered_node
        spaces_to_update: set[int] = set()
        if prev is not None and prev != curr:
            spaces_to_update.add(prev)
        if curr is not None and curr != prev:
            spaces_to_update.add(curr)

        for space_id in spaces_to_update:
            for artist in self._space_artists.get(space_id, []):
                try:
                    artist.remove()
                except ValueError:
                    pass
            space = self.id_to_space.get(space_id)
            if space is None:
                continue
            color = self.data.colors[space_id]
            label = self.data.labels.get(space_id, "")
            is_ref = (
                self.data.reference_space is not None
                and space is self.data.reference_space
            )
            is_hov = space_id == curr
            n_lines = len(self.ax_axes.lines)
            n_patches = len(self.ax_axes.patches)
            draw_space_axes(
                self.ax_axes, space,
                reference_space=self.data.reference_space,
                color=color, label=label,
                highlight=is_ref or is_hov,
            )
            self._space_artists[space_id] = (
                list(self.ax_axes.lines[n_lines:])
                + list(self.ax_axes.patches[n_patches:])
            )

        self.ax_axes.set_xlim(xlim)
        self.ax_axes.set_ylim(ylim)
        self.fig.canvas.draw_idle()
        self._prev_hovered = curr

    # ── hit-testing helpers ────────────────────────────────────────────────

    def _node_at_graph_pos(self, cx: float, cy: float) -> int | None:
        """Return the graph node id closest to (cx, cy) in graph-axes coords, or None."""
        if not self.data.pos:
            return None
        closest = min(
            self.data.pos,
            key=lambda n: (cx - self.data.pos[n][0]) ** 2 + (cy - self.data.pos[n][1]) ** 2,
        )
        dist_sq = (
            (cx - self.data.pos[closest][0]) ** 2
            + (cy - self.data.pos[closest][1]) ** 2
        )
        return closest if dist_sq <= 0.25 and closest in self.id_to_space else None

    def _node_at_axes_pos(self, cx: float, cy: float) -> int | None:
        """Return id(space) whose origin is closest to (cx, cy) in reference coords, or None."""
        best_id: int | None = None
        best_dist_sq = 0.15 ** 2  # threshold in data-units squared
        for space in self.data.spaces:
            try:
                if self.data.reference_space is not None:
                    origin = Point(np.array([0.0, 0.0]), space=space).relative_to(
                        self.data.reference_space
                    )
                else:
                    origin = Point(np.array([0.0, 0.0]), space=space).to_absolute()
                ox, oy = float(origin.coords[0]), float(origin.coords[1])
                dist_sq = (cx - ox) ** 2 + (cy - oy) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_id = id(space)
            except (ValueError, IndexError):
                pass
        return best_id

    # ── selection & transition ────────────────────────────────────────────

    def _select_node(self, node_id: int) -> None:
        old_ref = self.data.reference_space
        new_ref = self.id_to_space[node_id]
        if new_ref is old_ref:
            return
        self.selected_node = node_id
        self.data.reference_space = new_ref
        # Animate only when both spaces are in the same tree.
        if old_ref is not None and old_ref.get_root() is new_ref.get_root():
            self._start_anim(
                old_ref.compute_absolute_transform(),
                new_ref.compute_absolute_transform(),
                new_ref.get_root(),
            )
        else:
            self._redraw()

    def _start_anim(
        self, M_old_abs: np.ndarray, M_new_abs: np.ndarray, root: Space
    ) -> None:
        """Begin a smooth transition to the new reference space."""
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None

        # Update graph panel immediately (cheap — just a few nodes).
        self.ax_graph.clear()
        _draw_hierarchy_subplot(
            self.ax_graph, self.data,
            selected_node=self.selected_node,
            hovered_node=self.hovered_node,
        )

        # Precompute root inverse so temp_ref has the correct absolute transform.
        R_root = root.compute_absolute_transform()
        self._anim_R_root_inv = np.linalg.inv(R_root)
        self._anim_frame = 0
        self._anim_M_old = M_old_abs
        self._anim_M_new = M_new_abs
        self._anim_root = root

        # ── Blit setup: capture static background (no space artists) ────────
        # Hide all space artists temporarily so the background only contains
        # the axes frame, grid lines, ticks and labels.
        for artists in self._space_artists.values():
            for a in artists:
                a.set_visible(False)
        # Full draw so both panels are rendered correctly before we start.
        self.fig.canvas.draw()
        self._anim_bg = self.fig.canvas.copy_from_bbox(self.ax_axes.bbox)
        # Remove space artists — they will be recreated each frame.
        for artists in self._space_artists.values():
            for a in artists:
                try:
                    a.remove()
                except ValueError:
                    pass
        self._space_artists = {}

        self._anim_timer = self.fig.canvas.new_timer(interval=_ANIM_INTERVAL_MS)
        self._anim_timer.add_callback(self._anim_step)
        self._anim_timer.start()

    def _anim_step(self) -> None:
        """Timer callback: render one animation frame using blitting."""
        self._anim_frame += 1
        t = self._anim_frame / _ANIM_FRAMES
        if t >= 1.0:
            self._anim_timer.stop()
            self._anim_timer = None
            self._anim_bg = None
            self._redraw()
            return

        M_t_abs = _interpolate_trks2d(self._anim_M_old, self._anim_M_new, t)
        M_t_local = self._anim_R_root_inv @ M_t_abs
        temp_ref = Space(transform=M_t_local, parent=self._anim_root)

        # Remove previous frame's artists (no ax.clear()).
        for artists in self._space_artists.values():
            for a in artists:
                try:
                    a.remove()
                except ValueError:
                    pass

        # Draw new frame's artists into the axes (no grids — highlight=False).
        new_artists: dict[int, list] = {}
        for space in self.data.spaces:
            color = self.data.colors[id(space)]
            label = self.data.labels[id(space)]
            n_lines = len(self.ax_axes.lines)
            n_patches = len(self.ax_axes.patches)
            draw_space_axes(
                self.ax_axes, space,
                reference_space=temp_ref,
                color=color, label=label,
                highlight=False,
            )
            frame_artists = (
                list(self.ax_axes.lines[n_lines:])
                + list(self.ax_axes.patches[n_patches:])
            )
            for a in frame_artists:
                a.set_animated(True)
            new_artists[id(space)] = frame_artists
        self._space_artists = new_artists

        # Blit: restore static background, draw new artists on top, push to screen.
        renderer = self.fig.canvas.get_renderer()
        self.fig.canvas.restore_region(self._anim_bg)
        for frame_artists in self._space_artists.values():
            for a in frame_artists:
                self.ax_axes.draw_artist(a)
        self.fig.canvas.blit(self.ax_axes.bbox)

    # ── event handlers ────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if self._anim_timer is not None:
            return  # ignore clicks during transition
        if event.button != 1:
            return
        if event.inaxes is self.ax_graph and event.xdata is not None:
            node = self._node_at_graph_pos(event.xdata, event.ydata)
            if node is not None:
                self._select_node(node)
        elif event.inaxes is self.ax_axes and event.xdata is not None:
            node = self._node_at_axes_pos(event.xdata, event.ydata)
            if node is not None:
                self._select_node(node)
            else:
                self._start_pan(event)

    def _start_pan(self, event) -> None:
        if event.x is None:
            return
        self._pan_start_display = (event.x, event.y)
        self._pan_xlim = self.ax_axes.get_xlim()
        self._pan_ylim = self.ax_axes.get_ylim()
        # Capture the data transform at press time so deltas are stable throughout the drag.
        self._pan_inv_transform = self.ax_axes.transData.inverted()

    def _on_motion(self, event) -> None:
        if self._anim_timer is not None:
            return  # ignore motion during transition
        # Pan takes priority over hover detection.
        if self._pan_start_display is not None:
            if event.x is not None:
                start_data = self._pan_inv_transform.transform(self._pan_start_display)
                curr_data = self._pan_inv_transform.transform((event.x, event.y))
                dx = start_data[0] - curr_data[0]
                dy = start_data[1] - curr_data[1]
                self.ax_axes.set_xlim(self._pan_xlim[0] + dx, self._pan_xlim[1] + dx)
                self.ax_axes.set_ylim(self._pan_ylim[0] + dy, self._pan_ylim[1] + dy)
                self.fig.canvas.draw_idle()
            return

        # Hover detection.
        new_hover: int | None = None
        if event.inaxes is self.ax_graph and event.xdata is not None:
            new_hover = self._node_at_graph_pos(event.xdata, event.ydata)
        elif event.inaxes is self.ax_axes and event.xdata is not None:
            new_hover = self._node_at_axes_pos(event.xdata, event.ydata)

        if new_hover != self.hovered_node:
            self.hovered_node = new_hover
            self._redraw_hover()

    def _on_release(self, event) -> None:
        if event.button == 1:
            self._pan_start_display = None
            self._pan_xlim = None
            self._pan_ylim = None
            self._pan_inv_transform = None

    def _on_axes_leave(self, event) -> None:
        if self._anim_timer is not None:
            return
        if self.hovered_node is not None:
            self.hovered_node = None
            self._redraw_hover()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax_axes or event.xdata is None:
            return
        factor = 0.9 if event.step > 0 else 1.1
        cx, cy = event.xdata, event.ydata
        xlim = self.ax_axes.get_xlim()
        ylim = self.ax_axes.get_ylim()
        self.ax_axes.set_xlim(cx + (xlim[0] - cx) * factor, cx + (xlim[1] - cx) * factor)
        self.ax_axes.set_ylim(cy + (ylim[0] - cy) * factor, cy + (ylim[1] - cy) * factor)
        self.fig.canvas.draw_idle()


def draw_space_hierarchy(
    spaces: list,
    labels: dict | None = None,
    title: str = "Space hierarchy",
) -> None:
    """Draw a directed tree of Space objects and their coordinate axes side by side.

    The left panel shows the parent-child hierarchy as a directed graph.
    The right panel draws every space's coordinate frame as seen from the root
    space (or in absolute coordinates when there are multiple roots). Node colours
    are shared across both panels so each space is easy to identify.

    **Interactions on the axes panel (right):**

    - Left-click drag → pan.
    - Scroll wheel → zoom centred on the cursor.

    **Interactions on the hierarchy panel (left):**

    - Left-click a node → set that space as the reference for the axes panel.

    Requires both matplotlib and networkx (``pip install coordinatus[plotting]``).

    Args:
        spaces: All Space objects to include. Every object referenced as a
            parent must also be present in the list.
        labels: Optional mapping ``{id(space): display_label}``. If omitted,
            spaces are labelled "Space 0", "Space 1", … in list order.
        title: Overall figure title.

    Examples:
        >>> from coordinatus import Space2D, create_space
        >>> from coordinatus.visualization import draw_space_hierarchy
        >>> root = Space2D()
        >>> child = create_space(parent=root, tx=1, ty=0, angle_rad=0, sx=1, sy=1)
        >>> draw_space_hierarchy([root, child], title="My hierarchy")
    """
    _check_matplotlib()
    if not _HAS_NETWORKX:  # pragma: no cover
        raise ImportError(
            "networkx is required for draw_space_hierarchy. "
            "Install it with: pip install coordinatus[plotting]"
        )

    import matplotlib.pyplot as plt

    if labels is None:
        labels = {id(s): f"Space {i}" for i, s in enumerate(spaces)}

    data = _build_hierarchy_render_data(spaces, labels)
    panel_w, panel_h = _compute_figure_size(data.pos)

    fig, (ax_graph, ax_axes) = plt.subplots(1, 2, figsize=(panel_w * 2 + 1, panel_h))

    id_to_space = {id(s): s for s in spaces}

    # Keep a reference so the interactor is not garbage-collected.
    # The interactor performs the initial draw in its __init__.
    fig._hierarchy_interactor = _HierarchyInteractor(  # type: ignore[attr-defined]
        fig, ax_graph, ax_axes, data, title, id_to_space
    )
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
