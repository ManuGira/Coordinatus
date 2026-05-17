"""Visualization utilities for spaces and coordinates.

This module provides plotting functions to visualize spaces and points.
Requires matplotlib to be installed. The draw_space_hierarchy function
additionally requires networkx.

Install with: pip install coordinatus[plotting]
"""

from dataclasses import dataclass
from typing import Any, Optional, List, TYPE_CHECKING
import numpy as np

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
    
    # Use absolute space if space is None
    if space is None:
        space = Space2D()
    
    # Get space origin and unit vectors in reference space
    origin = Point(np.array([0, 0]), space=space)
    x_axis = Vector(np.array([1, 0]), space=space)
    y_axis = Vector(np.array([0, 1]), space=space)
    
    # Convert to reference space coordinates
    if reference_space is None:
        origin_coords = origin.to_absolute().coords
        x_axis_coords = x_axis.to_absolute().coords
        y_axis_coords = y_axis.to_absolute().coords
    else:
        origin_coords = origin.relative_to(reference_space).coords
        x_axis_coords = x_axis.relative_to(reference_space).coords
        y_axis_coords = y_axis.relative_to(reference_space).coords

    # Draw origin
    ax.plot(origin_coords[0], origin_coords[1], 'o', 
            color=color, label=f'{label} origin', zorder=5, alpha=alpha)
    
    # Draw x-axis
    head_size = 0.1
    ax.arrow(origin_coords[0], origin_coords[1], 
             x_axis_coords[0] * (1 - head_size), x_axis_coords[1] * (1 - head_size),
             head_width=head_size, head_length=head_size,
             fc=color, ec=color, alpha=alpha)
    
    # Draw y-axis
    ax.arrow(origin_coords[0], origin_coords[1],
             y_axis_coords[0] * (1 - head_size), y_axis_coords[1] * (1 - head_size),
             head_width=head_size, head_length=head_size,
             fc=color, ec=color, alpha=alpha)
    
    # Label axes
    ax.text(origin_coords[0] + x_axis_coords[0] + 0.2,
            origin_coords[1] + x_axis_coords[1],
            f'{label} X', fontsize=9, color=color, fontweight='bold', alpha=alpha)
    ax.text(origin_coords[0] + y_axis_coords[0],
            origin_coords[1] + y_axis_coords[1] + 0.2,
            f'{label} Y', fontsize=9, color=color, fontweight='bold', alpha=alpha)


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


def _draw_hierarchy_subplot(ax, data: _HierarchyRenderData) -> None:
    """Draw the networkx directed graph (parent → child) on *ax*."""
    node_color_list = [data.colors[n] for n in data.graph.nodes]
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
    )
    ax.set_title("Hierarchy")


def _draw_axes_subplot(ax, data: _HierarchyRenderData) -> None:
    """Draw every space's coordinate frame on *ax* using :func:`draw_space_axes`."""
    for space in data.spaces:
        color = data.colors[id(space)]
        label = data.labels[id(space)]
        draw_space_axes(ax, space, reference_space=data.reference_space,
                        color=color, label=label)
    ref_name = (
        data.labels.get(id(data.reference_space), "root")
        if data.reference_space is not None
        else "absolute"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Coordinate axes (in {ref_name} space)")
    ax.legend(fontsize=7, loc="best")


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
    _draw_hierarchy_subplot(ax_graph, data)
    _draw_axes_subplot(ax_axes, data)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
