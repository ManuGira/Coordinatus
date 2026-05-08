"""Scaling and shearing transformation utilities."""

import numpy as np
from numpy.typing import ArrayLike

def scale(scale_vector: ArrayLike) -> np.ndarray:
    """
    Creates a scaling matrix for an n-dimensional space.
    
    Args:
        scale_vector: Scale factors for each dimension [sx, sy, ...]
    
    Returns:
        An (n+1)x(n+1) scaling matrix in homogeneous coordinates where n is the 
        length of scale_vector. The matrix scales each dimension by the corresponding 
        factor while preserving the homogeneous coordinate.
    """
    scale_vector = np.asarray(scale_vector)
    assert scale_vector.ndim == 1, "scale_vector must be a 1D array"
    homogeneous_scale_vector = np.append(scale_vector, 1)
    return np.diagflat(homogeneous_scale_vector)


def scale1D(sx: float) -> np.ndarray:
    """
    Creates a 1D scaling matrix.

    Args:
        sx: Scale factor along the x-axis

    Returns:
        A 2x2 scaling matrix in homogeneous coordinates:
            [[sx, 0]
             [0,  1]]
    """
    return scale([sx])


def scale2D(sx: float, sy: float) -> np.ndarray:
    """
    Creates a 2D scaling matrix.
    
    Args:
        sx: Scale factor along the x-axis
        sy: Scale factor along the y-axis
    
    Returns:
        A 3x3 scaling matrix in homogeneous coordinates:
            [[sx, 0,  0]
             [0,  sy, 0]
             [0,  0,  1]]
    """
    return scale([sx, sy])

def scale3D(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Creates a 3D scaling matrix.
    
    Args:
        sx: Scale factor along the x-axis
        sy: Scale factor along the y-axis
        sz: Scale factor along the z-axis
    
    Returns:
        A 4x4 scaling matrix in homogeneous coordinates:
            [[sx, 0,  0,  0]
             [0,  sy, 0,  0]
             [0,  0,  sz, 0]
             [0,  0,  0,  1]]
    """
    return scale([sx, sy, sz])


def shear2D(kx: float, ky: float) -> np.ndarray:
    """Creates a 2D shear matrix."""
    return np.array([[1, kx, 0],
                     [ky, 1, 0],
                     [0,  0, 1]])
