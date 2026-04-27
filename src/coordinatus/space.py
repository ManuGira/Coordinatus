"""Coordinate space representation and operations."""

from typing import Optional
import numpy as np

from .transforms import trs2D

class Space:
    """A coordinate space that can be nested within other spaces.
    
    Each space stores its position, rotation, and scale (TRS) as first-class
    attributes. The ``transform`` property derives the 3x3 affine matrix from
    these attributes on demand, so individual components can be modified
    directly without rebuilding the matrix manually. Spaces can be organized
    in a hierarchy, like objects in a scene graph.
    
    Attributes:
        tx: Translation along the X-axis relative to the parent space.
        ty: Translation along the Y-axis relative to the parent space.
        angle_rad: Rotation angle in radians (counter-clockwise) relative to
                   the parent space.
        sx: Scale factor along the X-axis relative to the parent space.
        sy: Scale factor along the Y-axis relative to the parent space.
        parent: Optional parent coordinate space. If None, this is a root/absolute space.
        transform: Read-only 3x3 affine transformation matrix derived from the
                   TRS attributes. Recomputed each time it is accessed.
    
    Examples:
        >>> # Create a root coordinate space
        >>> root = Space()
        >>> 
        >>> # Create a child space translated by (5, 3) relative to root
        >>> child = Space(parent=root, tx=5, ty=3)
        >>> 
        >>> # Mutate a single TRS component — the transform updates automatically
        >>> child.tx = 10
        >>> 
        >>> # Get transformation to absolute space
        >>> absolute_t = child.compute_absolute_transform()
    """
    def __init__(self, parent: Optional['Space'] = None, tx: float = 0.0, ty: float = 0.0,
                 angle_rad: float = 0.0, sx: float = 1.0, sy: float = 1.0):
        """Initialize a coordinate space with TRS parameters.
        
        Args:
            parent: Parent coordinate space. If None, this is a root space.
            tx: Translation along the X-axis (default: 0.0).
            ty: Translation along the Y-axis (default: 0.0).
            angle_rad: Rotation angle in radians, counter-clockwise (default: 0.0).
            sx: Scale factor along the X-axis (default: 1.0).
            sy: Scale factor along the Y-axis (default: 1.0).
        """
        self.parent = parent
        self.tx = tx
        self.ty = ty
        self.angle_rad = angle_rad
        self.sx = sx
        self.sy = sy

    @property
    def transform(self) -> np.ndarray:
        """3x3 affine transformation matrix derived from the TRS attributes.
        
        Computed from ``tx``, ``ty``, ``angle_rad``, ``sx``, ``sy`` using
        ``trs2D``. Modifying any TRS attribute is immediately reflected the
        next time ``transform`` is accessed.
        
        Returns:
            3x3 numpy array representing the transformation from this space's
            local coordinates to the parent's coordinate system.
        
        Examples:
            >>> space = Space(tx=5, ty=3)
            >>> space.transform  # translation matrix (5, 3)
            >>> space.tx = 10
            >>> space.transform  # translation matrix (10, 3)
        """
        return trs2D(self.tx, self.ty, self.angle_rad, self.sx, self.sy)

    @property
    def D_in(self) -> int:
        """Returns the input dimension of this space's coordinate system.
        
        For the 2D TRS spaces represented by this class the transform is
        always a 3x3 matrix, so ``D_in`` is always 2.
        
        Returns:
            The number of dimensions in the space's input coordinate system
            (excludes the homogeneous coordinate row/column).
        
        Examples:
            >>> space = Space()
            >>> space.D_in
            2
        """
        return self.transform.shape[1] - 1  # Subtract 1 for homogeneous coordinate
    
    @property
    def D_out(self) -> int:
        """Returns the output dimension of the parent's coordinate system.
        
        For the 2D TRS spaces represented by this class the transform is
        always a 3x3 matrix, so ``D_out`` is always 2.
        
        Returns:
            The number of dimensions in the parent's coordinate system
            (excludes the homogeneous coordinate row/column).
        
        Examples:
            >>> space = Space()
            >>> space.D_out
            2
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
        
        # Both are identity spaces (no parent and all TRS values are defaults)
        if self.parent is None and other.parent is None:
            return np.allclose(self.transform, np.eye(3)) and np.allclose(other.transform, np.eye(3))
        
        return False

    def __ne__(self, other):
        """Check if two spaces are not equal."""
        return not self.__eq__(other)

    def compute_absolute_transform(self) -> np.ndarray:
        """Computes the cumulative transformation matrix from this space to absolute space.
        
        Recursively multiplies transformation matrices up the hierarchy to compute
        the complete transformation from this coordinate space to the root (absolute)
        coordinate space.
        
        Returns:
            3x3 numpy array representing the transformation from space-relative to absolute coordinates.
        
        Examples:
            >>> root = Space(tx=10, ty=5)
            >>> child = Space(parent=root, tx=3, ty=2)
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
        
        Args:
            target_space: The destination coordinate space.
        
        Returns:
            3x3 transformation matrix that converts coordinates from this space
            to the target space.
        
        Examples:
            >>> space_a = Space(tx=5)
            >>> space_b = Space(ty=3)
            >>> convert_t = space_a.compute_relative_transform_to(space_b)
            >>> # Use convert_t to express space_a coordinates in space_b
        """
        inv_transform = np.linalg.inv(target_space.compute_absolute_transform())
        return inv_transform @ self.compute_absolute_transform()


def create_space(parent: Optional[Space]=None, tx: float=0.0, ty: float=0.0, angle_rad: float=0.0, sx: float=1.0, sy: float=1.0) -> Space:
    """Factory function to create a coordinate space using TRS (Translation-Rotation-Scale) parameters.
    
    Convenience wrapper around the ``Space`` constructor. Accepts the same TRS
    parameters as ``Space.__init__`` and returns a new ``Space`` instance. The
    transformations are applied in TRS order: scale first, then rotate, then
    translate.
    
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
    return Space(parent=parent, tx=tx, ty=ty, angle_rad=angle_rad, sx=sx, sy=sy)
