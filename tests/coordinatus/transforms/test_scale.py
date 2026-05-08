"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    scale, scale1D, scale2D, scale3D, shear2D,
)

class TestScale1D:
    """Tests for the scale1D function."""

    def test_scale_identity(self):
        """Test scaling matrix with unit scale."""
        S = scale1D(1)
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_uniform(self):
        """Test uniform scaling."""
        S = scale1D(3)
        expected = np.array([[3, 0],
                             [0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_fractional(self):
        """Test scaling with a fractional value."""
        S = scale1D(0.5)
        expected = np.array([[0.5, 0],
                             [0,   1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_zero(self):
        """Test scaling to zero."""
        S = scale1D(0)
        expected = np.array([[0, 0],
                             [0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_negative(self):
        """Test negative scaling (reflection)."""
        S = scale1D(-2)
        expected = np.array([[-2, 0],
                              [0,  1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_point(self):
        """Test that the matrix correctly scales a point."""
        S = scale1D(4)
        point = np.array([3, 1])  # x=3, w=1
        result = S @ point
        expected = np.array([12, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_vector(self):
        """Test that the matrix correctly scales a vector (w=0)."""
        S = scale1D(5)
        vector = np.array([2, 0])  # x=2, w=0
        result = S @ vector
        expected = np.array([10, 0])
        np.testing.assert_array_almost_equal(result, expected)


class TestScale2D:
    """Tests for the scale2D function."""

    def test_scale_identity(self):
        """Test scaling matrix with unit scale."""
        S = scale2D(1, 1)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_uniform(self):
        """Test uniform scaling."""
        S = scale2D(2, 2)
        expected = np.array([[2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        S = scale2D(3, 2)
        expected = np.array([[3, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_zero(self):
        """Test scaling to zero."""
        S = scale2D(0, 0)
        expected = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_negative(self):
        """Test negative scaling (reflection)."""
        S = scale2D(-1, 1)
        expected = np.array([[-1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_point(self):
        """Test that scaling matrix correctly scales a point."""
        S = scale2D(2, 3)
        point = np.array([4, 5, 1])
        result = S @ point
        expected = np.array([8, 15, 1])  # Point scaled by (2, 3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_fractional(self):
        """Test scaling with fractional values."""
        S = scale2D(0.5, 0.25)
        expected = np.array([[0.5,  0,    0],
                            [0,    0.25, 0],
                            [0,    0,    1]])
        np.testing.assert_array_almost_equal(S, expected)


class TestShear2D:
    """Tests for the shear2D function."""

    def test_shear_identity(self):
        """Test shear matrix with zero shear."""
        K = shear2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_x_only(self):
        """Test shear in x direction only."""
        K = shear2D(0.5, 0)
        expected = np.array([[1, 0.5, 0],
                            [0, 1,   0],
                            [0, 0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_y_only(self):
        """Test shear in y direction only."""
        K = shear2D(0, 0.5)
        expected = np.array([[1,   0, 0],
                            [0.5, 1, 0],
                            [0,   0, 1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_both_axes(self):
        """Test shear in both x and y directions."""
        K = shear2D(0.3, 0.7)
        expected = np.array([[1,   0.3, 0],
                            [0.7, 1,   0],
                            [0,   0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_negative(self):
        """Test shear with negative values."""
        K = shear2D(-0.5, -0.25)
        expected = np.array([[1,     -0.5, 0],
                            [-0.25, 1,    0],
                            [0,     0,    1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_point_x(self):
        """Test that shear matrix correctly shears a point in x direction."""
        K = shear2D(1, 0)  # Shear x by y amount
        point = np.array([0, 2, 1])  # Point at (0, 2)
        result = K @ point
        expected = np.array([2, 2, 1])  # x shifted by y*kx = 2*1 = 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_point_y(self):
        """Test that shear matrix correctly shears a point in y direction."""
        K = shear2D(0, 1)  # Shear y by x amount
        point = np.array([3, 0, 1])  # Point at (3, 0)
        result = K @ point
        expected = np.array([3, 3, 1])  # y shifted by x*ky = 3*1 = 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_vector(self):
        """Test that shear matrix correctly shears a vector."""
        K = shear2D(0.5, 0.5)
        vector = np.array([2, 2, 0])  # Vector (w=0)
        result = K @ vector
        expected = np.array([3, 3, 0])  # x += y*0.5, y += x*0.5
        np.testing.assert_array_almost_equal(result, expected)


class TestScale:
    """Tests for the general n-dimensional scale function."""

    def test_scale_1D(self):
        """Test scaling matrix in 1D space."""
        S = scale([2])
        expected = np.array([[2, 0],
                            [0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_2D(self):
        """Test scaling matrix in 2D space."""
        S = scale([2, 3])
        expected = np.array([[2, 0, 0],
                            [0, 3, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_3D(self):
        """Test scaling matrix in 3D space."""
        S = scale([2, 3, 4])
        expected = np.array([[2, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 4, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_4D(self):
        """Test scaling matrix in 4D space."""
        S = scale([1, 2, 3, 4])
        expected = np.array([[1, 0, 0, 0, 0],
                            [0, 2, 0, 0, 0],
                            [0, 0, 3, 0, 0],
                            [0, 0, 0, 4, 0],
                            [0, 0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_uniform(self):
        """Test uniform scaling in 3D."""
        S = scale([2, 2, 2])
        expected = np.array([[2, 0, 0, 0],
                            [0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_identity(self):
        """Test identity scaling."""
        S = scale([1, 1, 1])
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_zero(self):
        """Test scaling to zero."""
        S = scale([0, 0, 0])
        expected = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_negative(self):
        """Test negative scaling (reflection)."""
        S = scale([-1, 1, -1])
        expected = np.array([[-1, 0,  0, 0],
                            [ 0, 1,  0, 0],
                            [ 0, 0, -1, 0],
                            [ 0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_fractional(self):
        """Test scaling with fractional values."""
        S = scale([0.5, 0.25, 0.1])
        expected = np.array([[0.5,  0,    0,   0],
                            [0,    0.25, 0,   0],
                            [0,    0,    0.1, 0],
                            [0,    0,    0,   1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_point_3D(self):
        """Test that scaling matrix correctly scales a 3D point."""
        S = scale([2, 3, 4])
        point = np.array([1, 2, 3, 1])  # Homogeneous coordinates
        result = S @ point
        expected = np.array([2, 6, 12, 1])  # Point scaled by (2, 3, 4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_vector_3D(self):
        """Test that scaling matrix correctly scales a 3D vector (w=0)."""
        S = scale([2, 3, 4])
        vector = np.array([1, 2, 3, 0])  # Homogeneous coordinates for vector
        result = S @ vector
        expected = np.array([2, 6, 12, 0])  # Vector scaled by (2, 3, 4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_accepts_list(self):
        """Test that scale accepts a list as input."""
        S = scale([2, 3])
        expected = np.array([[2, 0, 0],
                            [0, 3, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_accepts_numpy_array(self):
        """Test that scale accepts a numpy array as input."""
        S = scale(np.array([2, 3]))
        expected = np.array([[2, 0, 0],
                            [0, 3, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)


class TestScale3D:
    """Tests for the scale3D function."""

    def test_scale3D_identity(self):
        """Test 3D scaling matrix with unit scale."""
        S = scale3D(1, 1, 1)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_uniform(self):
        """Test uniform 3D scaling."""
        S = scale3D(2, 2, 2)
        expected = np.array([[2, 0, 0, 0],
                            [0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_non_uniform(self):
        """Test non-uniform 3D scaling."""
        S = scale3D(2, 3, 4)
        expected = np.array([[2, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 4, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_zero(self):
        """Test 3D scaling to zero."""
        S = scale3D(0, 0, 0)
        expected = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_negative(self):
        """Test negative 3D scaling (reflection)."""
        S = scale3D(-1, 1, -1)
        expected = np.array([[-1, 0,  0, 0],
                            [ 0, 1,  0, 0],
                            [ 0, 0, -1, 0],
                            [ 0, 0,  0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_fractional(self):
        """Test 3D scaling with fractional values."""
        S = scale3D(0.5, 0.25, 0.1)
        expected = np.array([[0.5,  0,    0,   0],
                            [0,    0.25, 0,   0],
                            [0,    0,    0.1, 0],
                            [0,    0,    0,   1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_point(self):
        """Test that 3D scaling matrix correctly scales a point."""
        S = scale3D(2, 3, 4)
        point = np.array([5, 6, 7, 1])  # Homogeneous coordinates
        result = S @ point
        expected = np.array([10, 18, 28, 1])  # Point scaled by (2, 3, 4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale3D_vector(self):
        """Test that 3D scaling matrix correctly scales a vector (w=0)."""
        S = scale3D(2, 3, 4)
        vector = np.array([1, 2, 3, 0])  # Homogeneous coordinates for vector
        result = S @ vector
        expected = np.array([2, 6, 12, 0])  # Vector scaled by (2, 3, 4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale3D_origin(self):
        """Test that 3D scaling leaves the origin unchanged."""
        S = scale3D(5, 10, 15)
        origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates
        result = S @ origin
        expected = np.array([0, 0, 0, 1])  # Origin unchanged
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale3D_mixed_values(self):
        """Test 3D scaling with mixed positive, negative, and fractional values."""
        S = scale3D(2, -0.5, 1.5)
        expected = np.array([[2,    0,    0,   0],
                            [0,   -0.5,  0,   0],
                            [0,    0,    1.5, 0],
                            [0,    0,    0,   1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale3D_consistency_with_scale(self):
        """Test that scale3D produces the same result as scale([sx, sy, sz])."""
        sx, sy, sz = 2, 3, 4
        S1 = scale3D(sx, sy, sz)
        S2 = scale([sx, sy, sz])
        np.testing.assert_array_almost_equal(S1, S2)


