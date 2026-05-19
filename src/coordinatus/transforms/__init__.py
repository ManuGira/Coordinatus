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


def _decompose_trks2d(
    M: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Decompose a 3x3 2D affine matrix into (tx, ty, angle_rad, kx, sx, sy).

    Uses a QR-based convention (upper-triangular shear) so the result is unique
    for any invertible matrix, including those built with :func:`trks2D`.
    The ``ky`` component is always 0 by convention.

    Given M = T @ R @ [[1, kx], [0, 1]] @ [[sx, 0], [0, sy]]:

    - sx = ||col0||  →  sqrt(M00^2 + M10^2)
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
    """Interpolate between two 3x3 2D affine matrices with smoothstep easing at *t* in [0, 1].

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


__all__ = [
    # Nothing to export explicitly, avoiding namespace conflictss
]
