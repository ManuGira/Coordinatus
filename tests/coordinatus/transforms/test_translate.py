"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    translate,
    translate1D,
    translate2D,
    translate3D
)

class TestTranslate2D:
    """Tests for the translate2D function."""

    def test_translate_zero(self):
        """Test translation matrix with zero translation."""
        T = translate2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_positive(self):
        """Test translation with positive values."""
        T = translate2D(3, 5)
        expected = np.array([[1, 0, 3],
                            [0, 1, 5],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_negative(self):
        """Test translation with negative values."""
        T = translate2D(-2, -4)
        expected = np.array([[1, 0, -2],
                            [0, 1, -4],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_point(self):
        """Test that translation matrix correctly transforms a point."""
        T = translate2D(3, 2)
        point = np.array([1, 1, 1])  # Homogeneous coordinates
        result = T @ point
        expected = np.array([4, 3, 1])  # Point moved by (3, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_vector(self):
        """Test that translation matrix doesn't affect vectors (w=0)."""
        T = translate2D(3, 2)
        vector = np.array([1, 1, 0])  # Homogeneous coordinates for vector
        result = T @ vector
        expected = np.array([1, 1, 0])  # Vector unchanged
        np.testing.assert_array_almost_equal(result, expected)


class TestTranslate:
    """Tests for the general n-dimensional translate function."""

    def test_translate_1D(self):
        """Test translation matrix in 1D space."""
        T = translate([5])
        expected = np.array([[1, 5],
                            [0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_2D(self):
        """Test translation matrix in 2D space."""
        T = translate([3, 4])
        expected = np.array([[1, 0, 3],
                            [0, 1, 4],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_3D(self):
        """Test translation matrix in 3D space."""
        T = translate([2, 3, 4])
        expected = np.array([[1, 0, 0, 2],
                            [0, 1, 0, 3],
                            [0, 0, 1, 4],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_4D(self):
        """Test translation matrix in 4D space."""
        T = translate([1, 2, 3, 4])
        expected = np.array([[1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 2],
                            [0, 0, 1, 0, 3],
                            [0, 0, 0, 1, 4],
                            [0, 0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_zero_vector(self):
        """Test translation with zero vector."""
        T = translate([0, 0, 0])
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_negative_values(self):
        """Test translation with negative values."""
        T = translate([-1, -2, -3])
        expected = np.array([[1, 0, 0, -1],
                            [0, 1, 0, -2],
                            [0, 0, 1, -3],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_point_3D(self):
        """Test that translation matrix correctly transforms a 3D point."""
        T = translate([5, 10, 15])
        point = np.array([1, 2, 3, 1])  # Homogeneous coordinates
        result = T @ point
        expected = np.array([6, 12, 18, 1])  # Point moved by (5, 10, 15)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_vector_3D(self):
        """Test that translation matrix doesn't affect 3D vectors (w=0)."""
        T = translate([5, 10, 15])
        vector = np.array([1, 2, 3, 0])  # Homogeneous coordinates for vector
        result = T @ vector
        expected = np.array([1, 2, 3, 0])  # Vector unchanged
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_accepts_list(self):
        """Test that translate accepts a list as input."""
        T = translate([1, 2])
        expected = np.array([[1, 0, 1],
                            [0, 1, 2],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_accepts_numpy_array(self):
        """Test that translate accepts a numpy array as input."""
        T = translate(np.array([1, 2]))
        expected = np.array([[1, 0, 1],
                            [0, 1, 2],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)


class TestTranslate3D:
    """Tests for the translate3D function."""

    def test_translate3D_zero(self):
        """Test 3D translation matrix with zero translation."""
        T = translate3D(0, 0, 0)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate3D_positive(self):
        """Test 3D translation with positive values."""
        T = translate3D(3, 5, 7)
        expected = np.array([[1, 0, 0, 3],
                            [0, 1, 0, 5],
                            [0, 0, 1, 7],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate3D_negative(self):
        """Test 3D translation with negative values."""
        T = translate3D(-2, -4, -6)
        expected = np.array([[1, 0, 0, -2],
                            [0, 1, 0, -4],
                            [0, 0, 1, -6],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate3D_mixed_values(self):
        """Test 3D translation with mixed positive and negative values."""
        T = translate3D(1, -2, 3)
        expected = np.array([[1, 0, 0, 1],
                            [0, 1, 0, -2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate3D_point(self):
        """Test that 3D translation matrix correctly transforms a point."""
        T = translate3D(10, 20, 30)
        point = np.array([1, 2, 3, 1])  # Homogeneous coordinates
        result = T @ point
        expected = np.array([11, 22, 33, 1])  # Point moved by (10, 20, 30)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate3D_vector(self):
        """Test that 3D translation matrix doesn't affect vectors (w=0)."""
        T = translate3D(10, 20, 30)
        vector = np.array([1, 2, 3, 0])  # Homogeneous coordinates for vector
        result = T @ vector
        expected = np.array([1, 2, 3, 0])  # Vector unchanged
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate3D_origin(self):
        """Test that 3D translation moves the origin correctly."""
        T = translate3D(5, 10, 15)
        origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates
        result = T @ origin
        expected = np.array([5, 10, 15, 1])  # Origin moved to (5, 10, 15)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate3D_float_values(self):
        """Test 3D translation with floating point values."""
        T = translate3D(1.5, 2.5, 3.5)
        expected = np.array([[1, 0, 0, 1.5],
                            [0, 1, 0, 2.5],
                            [0, 0, 1, 3.5],
                            [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate3D_consistency_with_translate(self):
        """Test that translate3D produces the same result as translate([tx, ty, tz])."""
        tx, ty, tz = 3, 5, 7
        T1 = translate3D(tx, ty, tz)
        T2 = translate([tx, ty, tz])
        np.testing.assert_array_almost_equal(T1, T2)


class TestTranslate1D:
    """Tests for the translate1D function."""

    def test_translate1D_returns_correct_matrix(self):
        """Test that translate1D returns a 2x2 homogeneous translation matrix."""
        T = translate1D(5)
        expected = np.array([[1, 5],
                             [0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

