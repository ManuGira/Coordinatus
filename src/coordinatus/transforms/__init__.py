"""Transformation matrix utilities for 2D affine transformations."""

import numpy as np

from .translate import translate, translate1D, translate2D, translate3D
from .rotate import rotate2D, rotate3Dx, rotate3Dy, rotate3Dz, rotate3D
from .scale import scale, scale1D, scale2D, scale3D, shear2D
from .dimension import (
    swap_axes,
    reduce_dim,
    augment_dim,
    project_xy_to_x,
    project_xy_to_y,
    project_xyz_to_xy,
    project_xyz_to_xz,
    project_xyz_to_yz,
    project_xyz_to_x,
    project_xyz_to_y,
    project_xyz_to_z,
)

def ts1D(tx: float=0, sx: float=1) -> np.ndarray:
    """Creates a combined translation and scaling matrix for 1D transformations."""
    T = translate([tx])
    S = scale([sx])
    return T @ S

def trs2D(tx: float=0, ty: float=0, angle_rad: float=0, sx: float=1, sy: float=1) -> np.ndarray:
    """Creates a combined translation, rotation, and scaling matrix."""
    T = translate2D(tx, ty)
    R = rotate2D(angle_rad)
    S = scale2D(sx, sy)
    return T @ R @ S


def trks2D(tx: float=0, ty: float=0, angle_rad: float=0, kx: float=0, ky: float=0, sx: float=1, sy: float=1) -> np.ndarray:
    """Creates a combined translation, rotation, shear, and scaling matrix."""
    T = translate2D(tx, ty)
    R = rotate2D(angle_rad)
    K = shear2D(kx, ky)
    S = scale2D(sx, sy)
    return T @ R @ K @ S


__all__ = [
    # Nothing to export explicitly, avoiding namespace conflictss
]
