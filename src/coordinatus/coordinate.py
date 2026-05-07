"""Coordinate representation classes for points and vectors."""

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

from .space import Space
from .types import CoordinateKind


def transform_coordinate(transform: np.ndarray, coordinates: np.ndarray, kind: CoordinateKind) -> np.ndarray:
    """Applies an affine transformation to a coordinate, respecting point vs vector semantics.
    
    Points and vectors transform differently under affine transformations:
    - Points (weight=1): Affected by translation, rotation, and scaling
    - Vectors (weight=0): Affected only by rotation and scaling, NOT translation
    
    This function converts to homogeneous coordinates, applies the transformation,
    and converts back to Cartesian coordinates.
    
    Args:
        transform: 3x3 affine transformation matrix in homogeneous coordinates.
        coordinates: 2D coordinate as numpy array [x, y] or DxN array where D is dimensions
                    and N is the number of points/vectors.
        kind: CoordinateKind.POINT or CoordinateKind.VECTOR.
    
    Returns:
        Transformed 2D coordinate as numpy array [x', y'] or DxN array of transformed coordinates.
    
    Examples:
        >>> # Single point translation
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateKind.POINT)
        array([6., 5.])  # Point moved by (5, 3)
        
        >>> # Single vector translation (no effect)
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateKind.VECTOR)
        array([1., 2.])  # Vector unchanged
        
        >>> # Multiple points translation
        >>> transform_coordinate(translate2D(5, 3), np.array([[1, 2], [2, 4]]), CoordinateKind.POINT)
        array([[6., 7.], [5., 7.]])  # All points moved by (5, 3)
    """
    # Check if we have a single coordinate (1D) - if so, reshape to (D, 1)
    is_single = coordinates.ndim == 1
    if is_single:
        coordinates = coordinates.reshape(-1, 1)
    
    # Handle DxN array where D is dimensions and N is number of points
    D, N = coordinates.shape
    
    # Convert to homogeneous coordinates by adding a row of weights
    weight = 1.0 if kind == CoordinateKind.POINT else 0.0
    weights = np.full((1, N), weight)
    homogeneous_coords = np.vstack([coordinates, weights])  # Shape (D+1, N)
    
    # Apply transformation: (3, 3) @ (3, N) -> (3, N)
    transformed_coords = transform @ homogeneous_coords
    
    # Return to Cartesian coordinates
    # Normalize by the weight (last row) if necessary
    weights = transformed_coords[-1, :]
    # Avoid division by zero - only normalize where weight != 0
    non_zero_mask = weights != 0
    if np.any(non_zero_mask):
        transformed_coords[:, non_zero_mask] /= weights[non_zero_mask]
    
    # Return all dimensions except the last (weight) row
    result = transformed_coords[:D, :]
    
    # If input was 1D, return 1D
    if is_single:
        result = result.flatten()
    
    return result


class Coordinate:
    """Base class for representing coordinates (points or vectors) in a coordinate space.
    
    Coordinates can be defined in any coordinate space and converted between spaces.
    The distinction between points and vectors is crucial:
    - Points: Represent positions, affected by all transformations including translation
    - Vectors: Represent directions/displacements, unaffected by translation
    
    Attributes:
        kind: CoordinateKind.POINT or CoordinateKind.VECTOR
        coords: 2D numpy array [x, y] in the local coordinate space
        space: The coordinate space this coordinate is defined in
    
    Examples:
        >>> space = Space(transform=translate2D(5, 3))
        >>> coord = Coordinate(CoordinateKind.POINT, np.array([1, 2]), space)
        >>> coord = Coordinate(CoordinateKind.POINT, [1, 2], space)  # list also works
        >>> coord = Coordinate(CoordinateKind.POINT, (1, 2), space)  # tuple also works
        >>> absolute_coord = coord.to_absolute()
    """

    def __init__(self, kind: CoordinateKind, coords: ArrayLike, space: Optional[Space] = None):
        """Initialize a coordinate.
        
        Args:
            kind: CoordinateKind.POINT or CoordinateKind.VECTOR
            coords: Array-like (numpy array, list, tuple, etc.) representing coordinates.
                         Can be [x, y] for a single point/vector or [[x1, x2, ...], [y1, y2, ...]]
                         for multiple points/vectors.
            space: Coordinate space this coordinate is defined in.
                   If None, uses absolute/identity space.
        """
        self.kind = kind
        self.coords = np.asarray(coords)
        if space is None:
            D = self.coords.shape[0] if self.coords.ndim > 1 else len(self.coords)
            self.space = Space(np.eye(D + 1))
        else:
            self.space = space

    @property
    def D(self) -> int:
        """Return the dimension D (number of dimensions).
        
        For a single coordinate [x, y], D = 2 (number of elements).
        For multiple coordinates [[x1, x2], [y1, y2]], D = 2 (number of rows).
        
        Returns:
            Number of dimensions.
        """
        if self.coords.ndim == 1:
            return len(self.coords)
        return self.coords.shape[0]

    @property
    def N(self) -> int:
        """Return N (number of points/vectors).
        
        For a single coordinate [x, y], N = 1.
        For multiple coordinates [[x1, x2], [y1, y2]], N = 2 (number of columns).
        
        Returns:
            Number of points/vectors.
        """
        if self.coords.ndim == 1:
            return 1
        return self.coords.shape[1]

    def _make_new(self, coords: np.ndarray, space: Optional[Space] = None) -> 'Coordinate':
        """Create a new coordinate of the same type as self.
        
        Helper method to handle the different constructors between Coordinate and its subclasses.
        Point/Vector don't take 'kind' argument, but Coordinate does.
        
        Args:
            coords: The coordinate values.
            space: The coordinate space. If not provided, uses self.space.
                  Pass Space() explicitly for identity space.
        
        Returns:
            A new instance of the same type as self with the given coords and space.
        """
        if space is None:
            space = self.space
        if type(self) is Coordinate:
            return Coordinate(kind=self.kind, coords=coords, space=space)
        else:
            # Point and Vector constructors don't take 'kind'
            return type(self)(coords=coords, space=space)  # type: ignore[call-arg]

    def __array__(self, dtype=None):
        """Return the underlying numpy array for numpy operations."""
        if dtype is None:
            return self.coords
        return self.coords.astype(dtype)

    def __getitem__(self, key):
        """Support indexing operations like coord[0] or coord[0, 1]."""
        return self.coords[key]

    def __setitem__(self, key, value):
        """Support item assignment like coord[0] = 5."""
        self.coords[key] = value

    def __len__(self):
        """Return length of the coordinate array."""
        return len(self.coords)

    def __repr__(self):
        """String representation of the coordinate."""
        return f"{self.__class__.__name__}(coords={self.coords!r}, space={self.space!r})"

    # Arithmetic operators
    def __add__(self, other):
        """Add coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.space != other.space:
                raise ValueError("Cannot add coordinates from different spaces. Convert to same space first.")
            new_coords = self.coords + other.coords
        else:
            new_coords = self.coords + other
        return self._make_new(new_coords)

    def __radd__(self, other):
        """Right addition."""
        new_coords = self.coords + other
        return self._make_new(new_coords)

    def __sub__(self, other):
        """Subtract coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.space != other.space:
                raise ValueError("Cannot subtract coordinates from different spaces. Convert to same space first.")
            new_coords = self.coords - other.coords
        else:
            new_coords = self.coords - other
        return self._make_new(new_coords)

    def __rsub__(self, other):
        """Right subtraction."""
        new_coords = other - self.coords
        return self._make_new(new_coords)

    def __mul__(self, other):
        """Multiply coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.space != other.space:
                raise ValueError("Cannot multiply coordinates from different spaces. Convert to same space first.")
            new_coords = self.coords * other.coords
        else:
            new_coords = self.coords * other
        return self._make_new(new_coords)

    def __rmul__(self, other):
        """Right multiplication."""
        new_coords = self.coords * other
        return self._make_new(new_coords)

    def __truediv__(self, other):
        """Divide coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.space != other.space:
                raise ValueError("Cannot divide coordinates from different spaces. Convert to same space first.")
            new_coords = self.coords / other.coords
        else:
            new_coords = self.coords / other
        return self._make_new(new_coords)

    def __rtruediv__(self, other):
        """Right division."""
        new_coords = other / self.coords
        return self._make_new(new_coords)

    def __neg__(self):
        """Negate coordinates."""
        new_coords = -self.coords
        return self._make_new(new_coords)

    def __abs__(self):
        """Absolute value of coordinates."""
        new_coords = np.abs(self.coords)
        return self._make_new(new_coords)

    # Comparison operators
    def __eq__(self, other):
        """Check equality."""
        if isinstance(other, Coordinate):
            return np.array_equal(self.coords, other.coords)
        return np.array_equal(self.coords, other)

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def to_absolute(self) -> 'Coordinate':
        """Converts this coordinate to absolute (identity) coordinate space.
        
        Applies the cumulative transformation from this coordinate's space through
        all parent spaces to express the coordinate in absolute space.
        
        Returns:
            New Coordinate with coordinates expressed in absolute space.
        
        Examples:
            >>> root = Space(transform=translate2D(10, 5))
            >>> child = Space(transform=translate2D(3, 2), parent=root)
            >>> point = Point(np.array([1, 1]), space=child)
            >>> absolute_point = point.to_absolute()
            >>> absolute_point.coords  # Should be [14, 8]
        """
        absolute_transform = self.space.compute_absolute_transform()
        absolute_coords = transform_coordinate(absolute_transform, self.coords, self.kind)
        D = absolute_coords.shape[0] if absolute_coords.ndim > 1 else len(absolute_coords)
        return self._make_new(absolute_coords, space=Space(np.eye(D + 1)))
        
    def relative_to(self, target_space: Space) -> 'Coordinate':
        """Converts this coordinate to a different coordinate space.
        
        Transforms the coordinate from its current space to the target space,
        properly handling the coordinate type (point vs vector) semantics.
        
        Args:
            target_space: The destination coordinate space.
        
        Returns:
            New Coordinate with coordinates expressed in the target space.
        
        Examples:
            >>> space_a = Space(transform=translate2D(5, 0))
            >>> space_b = Space(transform=translate2D(0, 3))
            >>> point_in_a = Point(np.array([0, 0]), space=space_a)
            >>> point_in_b = point_in_a.relative_to(space_b)
            >>> point_in_b.coords  # Should be [5, -3]
        """
        # Inverse transform from absolute to target space
        relative_transform = self.space.compute_relative_transform_to(target_space)
        relative_coords = transform_coordinate(relative_transform, self.coords, self.kind)
        return self._make_new(relative_coords, space=target_space)


class Point(Coordinate):
    """Represents a point (position) in a coordinate space.
    
    Points are affected by all transformations including translation, rotation, and scaling.
    Use this class to represent positions in space.
    
    Args:
        coords: Array-like (numpy array, list, tuple) [x, y] representing the point position,
                     or [[x1, x2, ...], [y1, y2, ...]] for multiple points.
        space: Coordinate space this point is defined in. If None, uses absolute space.
    
    Examples:
        >>> # Point at origin in a translated space
        >>> space = Space(transform=translate2D(10, 5))
        >>> point = Point([0, 0], space=space)  # list works
        >>> point = Point((0, 0), space=space)  # tuple works
        >>> point = Point(np.array([0, 0]), space=space)  # numpy array works
        >>> absolute_point = point.to_absolute()
        >>> absolute_point.coords  # [10, 5] - affected by translation
    """
    
    def __init__(self, coords: ArrayLike, space: Optional[Space] = None):
        super().__init__(
            kind=CoordinateKind.POINT,
            coords=coords, 
            space=space)
        

class Vector(Coordinate):
    """Represents a vector (direction/displacement) in a coordinate space.
    
    Vectors are NOT affected by translation, only by rotation and scaling.
    Use this class to represent directions, velocities, or relative displacements.
    
    Args:
        coords: Array-like (numpy array, list, tuple) [x, y] representing the vector components,
                     or [[x1, x2, ...], [y1, y2, ...]] for multiple vectors.
        space: Coordinate space this vector is defined in. If None, uses absolute space.
    
    Examples:
        >>> # Vector in a translated space
        >>> space = Space(transform=translate2D(10, 5))
        >>> vector = Vector([1, 0], space=space)  # list works
        >>> vector = Vector((1, 0), space=space)  # tuple works
        >>> vector = Vector(np.array([1, 0]), space=space)  # numpy array works
        >>> absolute_vector = vector.to_absolute()
        >>> absolute_vector.coords  # Still [1, 0] - unaffected by translation
    """
    
    def __init__(self, coords: ArrayLike, space: Optional[Space] = None):
        super().__init__(
            kind=CoordinateKind.VECTOR,
            coords=coords, 
            space=space)
