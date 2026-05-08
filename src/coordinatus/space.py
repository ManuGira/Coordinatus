"""Coordinate space representation and operations."""

from typing import Optional
import numpy as np

from .transforms import trs2D

class Space:
    """A coordinate space that can be nested within other spaces.
    
    Each space has a position, rotation, and scale relative to its parent space,
    encoded in a transform matrix. Use functions from transforms.py to easily
    create these matrices. Spaces can be organized in a hierarchy, like objects
    in a scene graph.
    
    The dimensionality of the space is determined by the shape of the transform matrix:
    a 2x2 matrix is 1D, a 3x3 matrix is 2D, a 4x4 matrix is 3D, etc.
    Use Space1D, Space2D, or Space3D for conveniently creating identity root spaces.
    
    Attributes:
        transform: MxN affine homogeneous transformation matrix from this space to its parent.
        parent: Optional parent coordinate space. If None, this is a root/absolute space.
    
    Examples:
        >>> # Create a child space translated by (5, 3) relative to a 2D root
        >>> root = Space2D()
        >>> child = Space(transform=translate2D(5, 3), parent=root)
        >>> 
        >>> # Get transformation to absolute space
        >>> absolute_t = child.compute_absolute_transform()
    """
    def __init__(self, transform: np.ndarray, parent: Optional['Space'] = None):
        """Initialize a coordinate space.
        
        Args:
            transform: Affine transformation matrix relative to parent in homogeneous
                      coordinates. A (D+1)x(D+1) matrix for a D-dimensional space.
            parent: Parent coordinate space. If None, this is a root space.
        """
        self.transform = transform
        self.parent = parent
        if transform is None:
            raise ValueError("transform must be a numpy array, not None. Use Space1D(), Space2D(), or Space3D() for identity root spaces.")

    @property
    def D_in(self) -> int:
        """
        Returns the input dimension of this space's coordinate space.
        
        This represents the dimensionality of points and vectors that are
        expressed in this space's local coordinate system (before transformation).
        For a 3x3 transformation matrix, D_in = 2 (2D space).
        For a 4x4 transformation matrix, D_in = 3 (3D space).
        
        Returns:
            The number of dimensions in the space's input space (excludes the
            homogeneous coordinate).
        
        Examples:
            >>> space_2d = Space(transform=np.eye(3))  # 3x3 matrix
            >>> space_2d.D_in
            2
            >>> space_3d = Space(transform=np.eye(4))  # 4x4 matrix
            >>> space_3d.D_in
            3
        """
        return self.transform.shape[1] - 1  # Subtract 1 for homogeneous coordinate
    
    @property
    def D_out(self) -> int:
        """
        Returns the output dimension of the parent's coordinate space.
        
        This represents the dimensionality of the parent space's coordinate
        system (after transformation). For standard transformations, D_out equals
        Din, but dimension-changing transformations (like projections) can have
        D_out ≠ Din.
        
        Returns:
            The number of dimensions in the parent's coordinate space (excludes
            the homogeneous coordinate).
        
        Examples:
            >>> space_2d = Space(transform=np.eye(3))  # 3x3 matrix
            >>> space_2d.D_out
            2
            >>> # For a projection from 3D to 2D (3x4 matrix):
            >>> # D_out would be 2, Din would be 3
        """
        return self.transform.shape[0] - 1  # Subtract 1 for homogeneous coordinate

    def __eq__(self, other):
        """Check if two spaces are equal.
        
        Two spaces are considered equal if:
        1. They are the same object (same reference), OR
        2. Both have no parent and both have identity transforms
        
        This allows coordinates in identity/absolute spaces to be operated on together.
        
        Args:
            other: Another Space object to compare with.
        
        Returns:
            True if spaces are considered equal, False otherwise.
        """
        if not isinstance(other, Space):
            return False
        
        # Same object reference
        if self is other:
            return True
        
        # Both are identity spaces (no parent and identity transform)
        if self.parent is None and other.parent is None:
            n, m = self.transform.shape[0], other.transform.shape[0]
            if n != m:
                return False
            return np.allclose(self.transform, np.eye(n)) and np.allclose(other.transform, np.eye(m))
        
        return False

    def __ne__(self, other):
        """Check if two spaces are not equal."""
        return not self.__eq__(other)

    def get_root(self) -> 'Space':
        """Returns the root (topmost) space in this space's hierarchy.
        
        Walks up the parent chain until reaching a space with no parent.
        
        Returns:
            The root Space of this hierarchy.
        
        Examples:
            >>> root = Space2D()
            >>> child = Space(transform=translate2D(5, 3), parent=root)
            >>> child.get_root() is root
            True
        """
        if self.parent is None:
            return self
        return self.parent.get_root()

    def compute_absolute_transform(self) -> np.ndarray:
        """Computes the cumulative transformation matrix from this space to absolute space.
        
        Recursively multiplies transformation matrices up the hierarchy to compute
        the complete transformation from this coordinate space to the root (absolute)
        coordinate space.
        
        Returns:
            3x3 numpy array representing the transformation from space-relative to absolute coordinates.
        
        Examples:
            >>> root = Space(transform=translate2D(10, 5))
            >>> child = Space(transform=translate2D(3, 2), parent=root)
            >>> absolute_t = child.compute_absolute_transform()
            >>> # absolute_t represents translation by (13, 7)
        """
        if self.parent is None:
            return self.transform
        else:
            return self.parent.compute_absolute_transform() @ self.transform

    def compute_relative_transform_to(self, target_space: 'Space') -> np.ndarray:
        """Computes the transformation matrix to convert coordinates from this space to another.
        
        Calculates the transformation needed to express coordinates defined in this
        coordinate space in the target coordinate space. This is computed by:
        1. Transforming from this space to absolute space
        2. Transforming from absolute space to the target space
        
        Both spaces must belong to the same coordinate hierarchy (share a common root).
        If they do not, a ValueError is raised — converting between unrelated coordinate
        spaces is undefined.
        
        Args:
            target_space: The destination coordinate space.
        
        Returns:
            Transformation matrix that converts coordinates from this space to the target space.
        
        Raises:
            ValueError: If the two spaces do not share a common ancestor.
        
        Examples:
            >>> root = Space2D()
            >>> space_a = Space(transform=translate2D(5, 0), parent=root)
            >>> space_b = Space(transform=translate2D(0, 3), parent=root)
            >>> convert_t = space_a.compute_relative_transform_to(space_b)
            >>> # Use convert_t to express space_a coordinates in space_b
        """
        if self.get_root() is not target_space.get_root():
            raise ValueError(
                "Cannot convert between unrelated coordinate spaces: the two spaces "
                "do not share a common ancestor. Ensure both spaces belong to the "
                "same coordinate hierarchy."
            )
        inv_transform = np.linalg.inv(target_space.compute_absolute_transform())
        return inv_transform @ self.compute_absolute_transform()


class Space1D(Space):
    """A 1D identity coordinate space (root/absolute).
    
    Equivalent to Space(transform=np.eye(2)). Use as the root space for 1D
    coordinate hierarchies.
    
    Args:
        parent: Optional parent coordinate space.
    """
    def __init__(self, parent: Optional[Space] = None):
        super().__init__(transform=np.eye(2), parent=parent)


class Space2D(Space):
    """A 2D identity coordinate space (root/absolute).
    
    Equivalent to Space(transform=np.eye(3)). Use as the root space for 2D
    coordinate hierarchies.
    
    Args:
        parent: Optional parent coordinate space.
    """
    def __init__(self, parent: Optional[Space] = None):
        super().__init__(transform=np.eye(3), parent=parent)


class Space3D(Space):
    """A 3D identity coordinate space (root/absolute).
    
    Equivalent to Space(transform=np.eye(4)). Use as the root space for 3D
    coordinate hierarchies.
    
    Args:
        parent: Optional parent coordinate space.
    """
    def __init__(self, parent: Optional[Space] = None):
        super().__init__(transform=np.eye(4), parent=parent)

class Space4D(Space):
    """A 4D identity coordinate space (root/absolute).
    
    Equivalent to Space(transform=np.eye(5)). Use as the root space for 4D
    coordinate hierarchies.
    
    Args:
        parent: Optional parent coordinate space.
    """
    def __init__(self, parent: Optional[Space] = None):
        super().__init__(transform=np.eye(5), parent=parent)

class SpaceND(Space):
    """A generic N-dimensional identity coordinate space (root/absolute).
    
    Equivalent to Space(transform=np.eye(N+1)). Use as the root space for N-dimensional
    coordinate hierarchies.
    
    Args:
        parent: Optional parent coordinate space.
        N: The number of dimensions for this space.
    """
    def __init__(self, N: int, parent: Optional[Space] = None):
        super().__init__(transform=np.eye(N + 1), parent=parent)


def create_space(parent: Optional[Space]=None, tx: float=0.0, ty: float=0.0, angle_rad: float=0.0, sx: float=1.0, sy: float=1.0) -> Space:
    """Factory function to create a coordinate space using TRS (Translation-Rotation-Scale) parameters.
    
    Convenience function that constructs a coordinate space from intuitive transformation
    parameters instead of requiring a raw transformation matrix. The transformations are
    applied in TRS order: scale first, then rotate, then translate.
    
    Args:
        parent: Parent coordinate space. If None, creates a root space.
        tx: Translation along X-axis (default: 0.0)
        ty: Translation along Y-axis (default: 0.0)
        angle_rad: Rotation angle in radians, counter-clockwise (default: 0.0)
        sx: Scale factor along X-axis (default: 1.0)
        sy: Scale factor along Y-axis (default: 1.0)
    
    Returns:
        A new Space with the specified transformation relative to its parent.
    
    Examples:
        >>> # Create root space at (10, 5) with no rotation or scaling
        >>> root = create_space(None, tx=10, ty=5)
        >>> 
        >>> # Create child rotated 90° and scaled 2x
        >>> child = create_space(root, angle_rad=np.pi/2, sx=2, sy=2)
    """
    transform = trs2D(tx, ty, angle_rad, sx, sy)
    return Space(transform=transform, parent=parent)
