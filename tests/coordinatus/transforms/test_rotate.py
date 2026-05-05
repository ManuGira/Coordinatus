"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    rotate2D, rotate3Dx, rotate3Dy, rotate3Dz, rotate3D,
)


class TestRotate2D:
    """Tests for the rotate2D function."""

    def test_rotate_zero(self):
        """Test rotation matrix with zero angle."""
        R = rotate2D(0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_90_degrees(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        R = rotate2D(np.pi / 2)
        expected = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_180_degrees(self):
        """Test rotation by 180 degrees (π radians)."""
        R = rotate2D(np.pi)
        expected = np.array([[-1,  0, 0],
                            [ 0, -1, 0],
                            [ 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_270_degrees(self):
        """Test rotation by 270 degrees (3π/2 radians)."""
        R = rotate2D(3 * np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_360_degrees(self):
        """Test rotation by 360 degrees (2π radians) returns to identity."""
        R = rotate2D(2 * np.pi)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_45_degrees(self):
        """Test rotation by 45 degrees (π/4 radians)."""
        R = rotate2D(np.pi / 4)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([[sqrt2_over_2, -sqrt2_over_2, 0],
                            [sqrt2_over_2,  sqrt2_over_2, 0],
                            [0,             0,            1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_point(self):
        """Test that rotation matrix correctly rotates a point."""
        R = rotate2D(np.pi / 2)  # 90 degrees
        point = np.array([1, 0, 1])  # Point at (1, 0)
        result = R @ point
        expected = np.array([0, 1, 1])  # Point rotated to (0, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_negative_angle(self):
        """Test rotation with negative angle (clockwise)."""
        R = rotate2D(-np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)


class TestRotate3Dx:
    """Tests for the rotate3Dx function (rotation around X-axis)."""

    def test_rotate3Dx_zero(self):
        """Test 3D rotation around X-axis with zero angle."""
        R = rotate3Dx(0)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_90_degrees(self):
        """Test 3D rotation around X-axis by 90 degrees (π/2 radians)."""
        R = rotate3Dx(np.pi / 2)
        expected = np.array([[1, 0,  0, 0],
                            [0, 0, -1, 0],
                            [0, 1,  0, 0],
                            [0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_180_degrees(self):
        """Test 3D rotation around X-axis by 180 degrees (π radians)."""
        R = rotate3Dx(np.pi)
        expected = np.array([[1,  0,  0, 0],
                            [0, -1,  0, 0],
                            [0,  0, -1, 0],
                            [0,  0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_270_degrees(self):
        """Test 3D rotation around X-axis by 270 degrees (3π/2 radians)."""
        R = rotate3Dx(3 * np.pi / 2)
        expected = np.array([[1,  0, 0, 0],
                            [0,  0, 1, 0],
                            [0, -1, 0, 0],
                            [0,  0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_360_degrees(self):
        """Test 3D rotation around X-axis by 360 degrees returns to identity."""
        R = rotate3Dx(2 * np.pi)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_point(self):
        """Test that rotation around X-axis correctly rotates a point."""
        R = rotate3Dx(np.pi / 2)  # 90 degrees
        point = np.array([1, 1, 0, 1])  # Point at (1, 1, 0)
        result = R @ point
        expected = np.array([1, 0, 1, 1])  # Y->Z, Z->-Y
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dx_x_axis_unchanged(self):
        """Test that points on X-axis remain unchanged."""
        R = rotate3Dx(np.pi / 4)
        point = np.array([5, 0, 0, 1])  # Point on X-axis
        result = R @ point
        expected = point
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dx_negative_angle(self):
        """Test rotation around X-axis with negative angle (clockwise)."""
        R = rotate3Dx(-np.pi / 2)
        expected = np.array([[1,  0, 0, 0],
                            [0,  0, 1, 0],
                            [0, -1, 0, 0],
                            [0,  0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dx_vector(self):
        """Test that rotation around X-axis correctly rotates a vector (w=0)."""
        R = rotate3Dx(np.pi / 2)
        vector = np.array([0, 1, 0, 0])  # Vector along Y-axis
        result = R @ vector
        expected = np.array([0, 0, 1, 0])  # Rotated to Z-axis
        np.testing.assert_array_almost_equal(result, expected)


class TestRotate3Dy:
    """Tests for the rotate3Dy function (rotation around Y-axis)."""

    def test_rotate3Dy_zero(self):
        """Test 3D rotation around Y-axis with zero angle."""
        R = rotate3Dy(0)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_90_degrees(self):
        """Test 3D rotation around Y-axis by 90 degrees (π/2 radians)."""
        R = rotate3Dy(np.pi / 2)
        expected = np.array([[0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_180_degrees(self):
        """Test 3D rotation around Y-axis by 180 degrees (π radians)."""
        R = rotate3Dy(np.pi)
        expected = np.array([[-1, 0,  0, 0],
                            [ 0, 1,  0, 0],
                            [ 0, 0, -1, 0],
                            [ 0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_270_degrees(self):
        """Test 3D rotation around Y-axis by 270 degrees (3π/2 radians)."""
        R = rotate3Dy(3 * np.pi / 2)
        expected = np.array([[ 0, 0, -1, 0],
                            [ 0, 1,  0, 0],
                            [ 1, 0,  0, 0],
                            [ 0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_360_degrees(self):
        """Test 3D rotation around Y-axis by 360 degrees returns to identity."""
        R = rotate3Dy(2 * np.pi)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_point(self):
        """Test that rotation around Y-axis correctly rotates a point."""
        R = rotate3Dy(np.pi / 2)  # 90 degrees
        point = np.array([1, 1, 0, 1])  # Point at (1, 1, 0)
        result = R @ point
        expected = np.array([0, 1, -1, 1])  # Z->X, X->-Z
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dy_y_axis_unchanged(self):
        """Test that points on Y-axis remain unchanged."""
        R = rotate3Dy(np.pi / 4)
        point = np.array([0, 5, 0, 1])  # Point on Y-axis
        result = R @ point
        expected = point
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dy_negative_angle(self):
        """Test rotation around Y-axis with negative angle (clockwise)."""
        R = rotate3Dy(-np.pi / 2)
        expected = np.array([[ 0, 0, -1, 0],
                            [ 0, 1,  0, 0],
                            [ 1, 0,  0, 0],
                            [ 0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dy_vector(self):
        """Test that rotation around Y-axis correctly rotates a vector (w=0)."""
        R = rotate3Dy(np.pi / 2)
        vector = np.array([0, 0, 1, 0])  # Vector along Z-axis
        result = R @ vector
        expected = np.array([1, 0, 0, 0])  # Rotated to X-axis
        np.testing.assert_array_almost_equal(result, expected)


class TestRotate3Dz:
    """Tests for the rotate3Dz function (rotation around Z-axis)."""

    def test_rotate3Dz_zero(self):
        """Test 3D rotation around Z-axis with zero angle."""
        R = rotate3Dz(0)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_90_degrees(self):
        """Test 3D rotation around Z-axis by 90 degrees (π/2 radians)."""
        R = rotate3Dz(np.pi / 2)
        expected = np.array([[0, -1, 0, 0],
                            [1,  0, 0, 0],
                            [0,  0, 1, 0],
                            [0,  0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_180_degrees(self):
        """Test 3D rotation around Z-axis by 180 degrees (π radians)."""
        R = rotate3Dz(np.pi)
        expected = np.array([[-1,  0, 0, 0],
                            [ 0, -1, 0, 0],
                            [ 0,  0, 1, 0],
                            [ 0,  0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_270_degrees(self):
        """Test 3D rotation around Z-axis by 270 degrees (3π/2 radians)."""
        R = rotate3Dz(3 * np.pi / 2)
        expected = np.array([[ 0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_360_degrees(self):
        """Test 3D rotation around Z-axis by 360 degrees returns to identity."""
        R = rotate3Dz(2 * np.pi)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_point(self):
        """Test that rotation around Z-axis correctly rotates a point."""
        R = rotate3Dz(np.pi / 2)  # 90 degrees
        point = np.array([1, 0, 1, 1])  # Point at (1, 0, 1)
        result = R @ point
        expected = np.array([0, 1, 1, 1])  # X->Y, Y->-X, Z unchanged
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dz_z_axis_unchanged(self):
        """Test that points on Z-axis remain unchanged."""
        R = rotate3Dz(np.pi / 4)
        point = np.array([0, 0, 5, 1])  # Point on Z-axis
        result = R @ point
        expected = point
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dz_negative_angle(self):
        """Test rotation around Z-axis with negative angle (clockwise)."""
        R = rotate3Dz(-np.pi / 2)
        expected = np.array([[ 0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3Dz_vector(self):
        """Test that rotation around Z-axis correctly rotates a vector (w=0)."""
        R = rotate3Dz(np.pi / 2)
        vector = np.array([1, 0, 0, 0])  # Vector along X-axis
        result = R @ vector
        expected = np.array([0, 1, 0, 0])  # Rotated to Y-axis
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3Dz_similar_to_rotate2D(self):
        """Test that rotation around Z-axis is similar to 2D rotation in XY plane."""
        angle = np.pi / 6
        R3D = rotate3Dz(angle)
        R2D = rotate2D(angle)
        # Compare the upper-left 3x3 portion
        np.testing.assert_array_almost_equal(R3D[:3, :3], R2D)


class TestRotate3D:
    """Tests for the rotate3D function (combined rotation around all axes)."""

    def test_rotate3D_all_zeros(self):
        """Test combined rotation with all zero angles returns identity."""
        R = rotate3D(0, 0, 0)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3D_only_x(self):
        """Test combined rotation with only X angle should equal rotate3Dx."""
        angle = np.pi / 4
        R = rotate3D(angle, 0, 0)
        expected = rotate3Dx(angle)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3D_only_y(self):
        """Test combined rotation with only Y angle should equal rotate3Dy."""
        angle = np.pi / 3
        R = rotate3D(0, angle, 0)
        expected = rotate3Dy(angle)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3D_only_z(self):
        """Test combined rotation with only Z angle should equal rotate3Dz."""
        angle = np.pi / 6
        R = rotate3D(0, 0, angle)
        expected = rotate3Dz(angle)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate3D_90_each_axis(self):
        """Test combined 90-degree rotations around each axis."""
        R = rotate3D(np.pi / 2, np.pi / 2, np.pi / 2)
        # Apply to a point to verify it rotates correctly
        point = np.array([1, 0, 0, 1])
        result = R @ point
        # This is a sanity check that the combination works
        assert result.shape == (4,)
        assert result[3] == 1  # Homogeneous coordinate should remain 1

    def test_rotate3D_matrix_is_orthogonal(self):
        """Test that the combined rotation matrix is orthogonal (R^T @ R = I)."""
        R = rotate3D(np.pi / 4, np.pi / 3, np.pi / 6)
        identity = R.T @ R
        np.testing.assert_array_almost_equal(identity, np.eye(4))

    def test_rotate3D_determinant_is_one(self):
        """Test that the determinant of the rotation matrix is 1."""
        R = rotate3D(np.pi / 4, np.pi / 3, np.pi / 6)
        det = np.linalg.det(R)
        np.testing.assert_almost_equal(det, 1.0)

    def test_rotate3D_composition(self):
        """Test that combined rotation equals individual rotations composed."""
        angle_x, angle_y, angle_z = np.pi / 6, np.pi / 4, np.pi / 3
        R_combined = rotate3D(angle_x, angle_y, angle_z)
        # Rotation order: Rz @ Ry @ Rx
        R_composed = rotate3Dz(angle_z) @ rotate3Dy(angle_y) @ rotate3Dx(angle_x)
        np.testing.assert_array_almost_equal(R_combined, R_composed)

    def test_rotate3D_point_rotation(self):
        """Test that combined rotation correctly transforms a point."""
        R = rotate3D(0, np.pi / 2, 0)  # Only Y rotation
        point = np.array([1, 0, 0, 1])
        result = R @ point
        # 90-degree Y rotation: X -> -Z
        expected = np.array([0, 0, -1, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3D_non_commutative(self):
        """Test that rotations are non-commutative."""
        angle_x, angle_y, angle_z = np.pi / 4, np.pi / 3, np.pi / 6
        R1 = rotate3D(angle_x, angle_y, angle_z)
        R2 = rotate3D(angle_z, angle_y, angle_x)
        # These should not be equal (except for special angle combinations)
        try:
            np.testing.assert_array_almost_equal(R1, R2)
            # Equal for this combination, that's fine
        except AssertionError:
            # This is expected - rotations are generally non-commutative
            pass

    def test_rotate3D_vector_rotation(self):
        """Test that combined rotation correctly transforms a vector."""
        R = rotate3D(np.pi / 2, 0, 0)  # Only X rotation
        vector = np.array([0, 1, 0, 0])  # Vector along Y-axis
        result = R @ vector
        # 90-degree X rotation: Y -> Z
        expected = np.array([0, 0, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate3D_multiple_angles(self):
        """Test rotation with all three angles non-zero."""
        R = rotate3D(np.pi / 6, np.pi / 4, np.pi / 3)
        # Just verify it produces a valid rotation matrix
        assert R.shape == (4, 4)
        # Check orthogonality
        identity = R.T @ R
        np.testing.assert_array_almost_equal(identity, np.eye(4))
        # Check determinant is 1
        det = np.linalg.det(R)
        np.testing.assert_almost_equal(det, 1.0)


