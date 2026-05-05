"""Rotation transformation utilities."""

import numpy as np


def rotate2D(angle_rad: float) -> np.ndarray:
    """
    Creates a 2D rotation matrix.
    
    Args:
        angle_rad: Rotation angle in radians (counter-clockwise)
    
    Returns:
        A 3x3 rotation matrix in homogeneous coordinates:
            [[cos(θ), -sin(θ), 0]
             [sin(θ),  cos(θ), 0]
             [0,       0,      1]]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def rotate3Dx(angle_rad: float) -> np.ndarray:
    """
    Creates a 3D rotation matrix around the X-axis.
    
    Args:
        angle_rad: Rotation angle in radians (counter-clockwise when looking from positive X)
    
    Returns:
        A 4x4 rotation matrix in homogeneous coordinates:
            [[1, 0,       0,       0]
             [0, cos(θ), -sin(θ),  0]
             [0, sin(θ),  cos(θ),  0]
             [0, 0,       0,       1]]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

def rotate3Dy(angle_rad: float) -> np.ndarray:
    """
    Creates a 3D rotation matrix around the Y-axis.
    
    Args:
        angle_rad: Rotation angle in radians (counter-clockwise when looking from positive Y)
    
    Returns:
        A 4x4 rotation matrix in homogeneous coordinates:
            [[cos(θ),  0, sin(θ), 0]
             [0,       1, 0,      0]
             [-sin(θ), 0, cos(θ), 0]
             [0,       0, 0,      1]]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate3Dz(angle_rad: float) -> np.ndarray:
    """
    Creates a 3D rotation matrix around the Z-axis.
    
    Args:
        angle_rad: Rotation angle in radians (counter-clockwise when looking from positive Z)
    
    Returns:
        A 4x4 rotation matrix in homogeneous coordinates:
            [[cos(θ), -sin(θ), 0, 0]
             [sin(θ),  cos(θ), 0, 0]
             [0,       0,      1, 0]
             [0,       0,      0, 1]]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])

def rotate3D(angle_x_rad: float, angle_y_rad: float, angle_z_rad: float) -> np.ndarray:
    """
    Creates a combined 3D rotation matrix from 3 angles.
        
    Args:
        angle_x_rad: Rotation angle around X-axis in radians
        angle_y_rad: Rotation angle around Y-axis in radians
        angle_z_rad: Rotation angle around Z-axis in radians
    
    Returns:
        A 4x4 rotation matrix in homogeneous coordinates representing the combined rotation.
    """
    Rx = rotate3Dx(angle_x_rad)
    Ry = rotate3Dy(angle_y_rad)
    Rz = rotate3Dz(angle_z_rad)
    # Combined rotation matrix. Will be applied from right to left, around extrinsic axes
    return Rz @ Ry @ Rx

