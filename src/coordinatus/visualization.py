"""Visualization utilities for spaces and coordinates.

This module provides plotting functions to visualize spaces and points.
Requires matplotlib to be installed.

Install with: pip install coordinatus[plotting]
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from matplotlib.axes import Axes as _Axes
    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    _HAS_MATPLOTLIB = False
    _Axes = None  # type: ignore

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
    
    # Use absolute space if reference_space is None
    if reference_space is None:
        reference_space = Space2D()
    
    # Get space origin and unit vectors in reference space
    origin = Point(np.array([0, 0]), space=space)
    x_axis = Vector(np.array([1, 0]), space=space)
    y_axis = Vector(np.array([0, 1]), space=space)
    
    # Convert to reference space coordinates
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
    
    # Use absolute space if reference_space is None
    if reference_space is None:
        reference_space = Space2D()
    
    # Get point coordinates in reference space
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
            