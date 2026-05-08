"""Translation transformation utilities."""

import numpy as np
from numpy.typing import ArrayLike

def translate(translation_vector: ArrayLike) -> np.ndarray:
    """
    Creates a translation matrix for an n-dimensional space.
    
    Args:
        translation_vector: Translation offsets for each dimension [tx, ty, ...]
    
    Returns:
        An (n+1)x(n+1) translation matrix in homogeneous coordinates where n is the
        length of translation_vector. The matrix translates points by the specified
        offsets while leaving vectors (w=0) unchanged.
    """
    translation_vector = np.asarray(translation_vector)
    dim = translation_vector.shape[0]
    T = np.eye(dim + 1)
    T[:-1, -1] = translation_vector
    return T

def translate1D(tx: float) -> np.ndarray:
    """
    Creates a 1D translation matrix.
    
    Args:
        tx: Translation offset along the x-axis
    
    Returns:
        A 2x2 translation matrix in homogeneous coordinates:
            [[1, tx]
             [0, 1]]
    """
    return translate([tx])

def translate2D(tx: float, ty: float) -> np.ndarray:
    """
    Creates a 2D translation matrix.
    
    Args:
        tx: Translation offset along the x-axis
        ty: Translation offset along the y-axis
    
    Returns:
        A 3x3 translation matrix in homogeneous coordinates:
            [[1, 0, tx]
             [0, 1, ty]
             [0, 0, 1]]
    """
    return translate([tx, ty])

def translate3D(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Creates a 3D translation matrix.
    
    Args:
        tx: Translation offset along the x-axis
        ty: Translation offset along the y-axis
        tz: Translation offset along the z-axis
    
    Returns:
        A 4x4 translation matrix in homogeneous coordinates:
            [[1, 0, 0, tx]
             [0, 1, 0, ty]
             [0, 0, 1, tz]
             [0, 0, 0, 1]]
    """
    return translate([tx, ty, tz])