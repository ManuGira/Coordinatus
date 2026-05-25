"""Coordinate package for managing coordinate systems and transformations."""

# Import main classes and functions for convenient access
from .coordinatus_types import CoordinateKind
from .space import Space, Space1D, Space2D, Space3D, Space4D, SpaceND, ProjectionSpace, create_space
from . import transforms  # allows access to `coordinatus.transforms.translate2D(1, 2)``
from .coordinate import Coordinate, Point, Vector, transform_coordinate
from .serializer import to_json, from_json

# Visualization is optional - only available if matplotlib is installed
try:
    from . import visualization2
except ImportError:  # pragma: no cover
    visualization2 = None  # type: ignore[assignment]

# Define what's available when using "from coordinate import *"
__all__ = [
    # Nothing to export explicitly, avoinding namespace conflictions
]


