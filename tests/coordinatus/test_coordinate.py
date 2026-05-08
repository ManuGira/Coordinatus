"""Unit tests for the Coordinate, Point, and Vector classes."""

import numpy as np
from coordinatus.coordinate import Coordinate, Point, Vector, transform_coordinate
from coordinatus.space import Space
from coordinatus.types import CoordinateKind
from coordinatus.transforms import translate2D, rotate2D, scale2D
from coordinatus.transforms import project_xyz_to_xy, project_xy_to_x


class TestTransformCoordinate:
    """Tests for the transform_coordinate utility function."""

    def test_transform_point_translation(self):
        """Test transforming a point with translation."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # Point should be translated
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_translation(self):
        """Test that translation doesn't affect vectors."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should be unchanged by translation
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_rotation(self):
        """Test transforming a point with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_rotation(self):
        """Test transforming a vector with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should also rotate
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_scale(self):
        """Test transforming a point with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_scale(self):
        """Test transforming a vector with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should also be scaled
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)


class TestTransformCoordinateProjection:
    """Tests for transform_coordinate with dimension-reducing (non-square) projection matrices.

    These tests reveal a bug: transform_coordinate uses D_in to slice the output but should
    use D_out = transform.shape[0] - 1 for non-square projection matrices.
    """

    def test_projection_3d_to_2d_single_point_output_shape(self):
        """Projecting a single 3D point through a 3x4 matrix must yield a 2D result, not 3D."""
        proj = project_xyz_to_xy()  # 3x4 matrix: 3D -> 2D
        coords = np.array([3.0, 5.0, 7.0])
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        assert result.shape == (2,), (
            f"Expected shape (2,) for 3D->2D projection, got {result.shape}. "
            "transform_coordinate is using D_in instead of D_out to slice the result."
        )

    def test_projection_3d_to_2d_single_point_values(self):
        """Projecting a 3D point through project_xyz_to_xy must return its x, y components."""
        proj = project_xyz_to_xy()  # drops z, keeps x and y
        coords = np.array([3.0, 5.0, 7.0])
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        np.testing.assert_array_almost_equal(result, np.array([3.0, 5.0]))

    def test_projection_3d_to_2d_batch_output_shape(self):
        """Projecting a batch of 3D points through a 3x4 matrix must yield a (2, N) array."""
        proj = project_xyz_to_xy()
        coords = np.array([[1.0, 2.0, 3.0],   # x values
                           [4.0, 5.0, 6.0],   # y values
                           [7.0, 8.0, 9.0]])  # z values, shape (3, 3)
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        assert result.shape == (2, 3), (
            f"Expected shape (2, 3) for 3D->2D projection of 3 points, got {result.shape}."
        )

    def test_projection_3d_to_2d_batch_values(self):
        """Batch 3D->2D projection must return x and y columns only."""
        proj = project_xyz_to_xy()
        coords = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        expected = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_projection_2d_to_1d_single_point_output_shape(self):
        """Projecting a single 2D point through a 2x3 matrix must yield a 1D result."""
        proj = project_xy_to_x()  # 2x3 matrix: 2D -> 1D
        coords = np.array([3.0, 5.0])
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        assert result.shape == (1,), (
            f"Expected shape (1,) for 2D->1D projection, got {result.shape}."
        )

    def test_projection_2d_to_1d_single_point_values(self):
        """Projecting a 2D point through project_xy_to_x must return only the x component."""
        proj = project_xy_to_x()
        coords = np.array([3.0, 5.0])
        result = transform_coordinate(proj, coords, CoordinateKind.POINT)
        np.testing.assert_array_almost_equal(result, np.array([3.0]))

    def test_projection_3d_to_2d_output_shape_via_to_absolute(self):
        """Point.to_absolute() through a projection Space must return a coordinate with D_out dims.

        A Space whose transform is the 3x4 projection matrix has D_in=3 and D_out=2.
        A 3D Point in that space converted to_absolute() must yield a 2D coordinate.
        """
        proj = project_xyz_to_xy()  # 3x4
        space = Space(transform=proj)  # root space, D_in=3, D_out=2
        point = Point(np.array([3.0, 5.0, 7.0]), space=space)
        result = point.to_absolute()
        assert result.D == 2, (
            f"Expected to_absolute() to yield a 2D coordinate, got D={result.D}."
        )
        np.testing.assert_array_almost_equal(result.coords, np.array([3.0, 5.0]))


class TestCoordinateInit:
    """Tests for Coordinate initialization."""

    def test_coordinate_init_no_system(self):
        """Test creating a coordinate without a space creates identity space."""
        coord = Coordinate(
            kind=CoordinateKind.POINT,
            coords=np.array([1, 2]),
            space=None
        )
        
        assert coord.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(coord.coords, [1, 2])
        assert coord.space is not None
        assert coord.space.parent is None
        np.testing.assert_array_equal(coord.space.transform, np.eye(3))

    def test_coordinate_init_with_system(self):
        """Test creating a coordinate with a custom space."""
        space = Space(transform=translate2D(5, 3), parent=None)
        coord = Coordinate(
            kind=CoordinateKind.VECTOR,
            coords=np.array([2, 3]),
            space=space
        )
        
        assert coord.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(coord.coords, [2, 3])
        assert coord.space is space


class TestPointInit:
    """Tests for Point initialization."""

    def test_point_init_basic(self):
        """Test creating a basic point."""
        point = Point(coords=np.array([3, 4]))
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert point.space is not None

    def test_point_init_with_system(self):
        """Test creating a point with a custom space."""
        space = Space(transform=translate2D(10, 20), parent=None)
        point = Point(coords=np.array([1, 1]), space=space)
        
        assert point.kind == CoordinateKind.POINT
        assert point.space is space

    def test_point_init_with_list(self):
        """Test creating a point with a list."""
        point = Point(coords=[3, 4])
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert isinstance(point.coords, np.ndarray)

    def test_point_init_with_tuple(self):
        """Test creating a point with a tuple."""
        point = Point(coords=(3, 4))
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert isinstance(point.coords, np.ndarray)


class TestVectorInit:
    """Tests for Vector initialization."""

    def test_vector_init_basic(self):
        """Test creating a basic vector."""
        vector = Vector(coords=np.array([1, 0]))
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert vector.space is not None

    def test_vector_init_with_system(self):
        """Test creating a vector with a custom space."""
        space = Space(transform=rotate2D(np.pi / 4), parent=None)
        vector = Vector(coords=np.array([1, 1]), space=space)
        
        assert vector.kind == CoordinateKind.VECTOR
        assert vector.space is space

    def test_vector_init_with_list(self):
        """Test creating a vector with a list."""
        vector = Vector(coords=[1, 0])
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert isinstance(vector.coords, np.ndarray)

    def test_vector_init_with_tuple(self):
        """Test creating a vector with a tuple."""
        vector = Vector(coords=(1, 0))
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert isinstance(vector.coords, np.ndarray)


class TestCoordinateToAbsolute:
    """Tests for converting coordinates to absolute space."""

    def test_coordinate_base_class_to_absolute(self):
        """Test to_absolute with base Coordinate class (for coverage)."""
        space = Space(transform=translate2D(5, 3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([1, 2]), space=space)
        
        result = coord.to_absolute()
        
        # Should work the same as Point
        np.testing.assert_array_almost_equal(result.coords, [6, 5])
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT

    def test_to_global_no_parent(self):
        """Test to_absolute when space has no parent (root space)."""
        space = Space(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = point.to_absolute()
        
        # Point at (1, 2) in space with translation (5, 3)
        # Absolute coordinates should be (6, 5)
        np.testing.assert_array_almost_equal(result.coords, [6, 5])
        # Result should be in identity/absolute space
        assert result.space.parent is None
        np.testing.assert_array_almost_equal(result.space.transform, np.eye(3))

    def test_to_global_one_level(self):
        """Test to_absolute with one parent level."""
        parent = Space(transform=translate2D(10, 5), parent=None)
        child_space = Space(transform=translate2D(3, 2), parent=parent)
        point = Point(coords=np.array([1, 1]), space=child_space)
        
        result = point.to_absolute()
        
        # Point at (1,1) in child
        # Child is at (3,2) relative to parent
        # So point is at (4,3) in parent
        # Parent is at (10,5) in absolute
        # So point is at (14,8) in absolute
        np.testing.assert_array_almost_equal(result.coords, [14, 8])
        # Result should be in absolute/identity space
        assert result.space.parent is None
        np.testing.assert_array_almost_equal(result.space.transform, np.eye(3))

    def test_to_global_multiple_levels(self):
        """Test to_absolute with nested hierarchy."""
        root = Space(transform=translate2D(100, 100), parent=None)
        middle = Space(transform=translate2D(10, 10), parent=root)
        leaf = Space(transform=translate2D(1, 1), parent=middle)
        
        point = Point(coords=np.array([0, 0]), space=leaf)
        result = point.to_absolute()
        
        # Point at origin in leaf space
        # Leaf is at (1, 1) in middle, middle is at (10, 10) in root, root is at (100, 100) in absolute
        # So point is at (111, 111) in absolute coordinates
        np.testing.assert_array_almost_equal(result.coords, [111, 111])
        # Result should be in absolute/identity space
        assert result.space.parent is None
        np.testing.assert_array_almost_equal(result.space.transform, np.eye(3))

    def test_to_global_with_rotation(self):
        """Test to_absolute with rotated coordinate space."""
        parent = Space(transform=np.eye(3), parent=None)
        # Child rotated 90 degrees
        child = Space(transform=rotate2D(np.pi / 2), parent=parent)
        
        # Point at (1, 0) in child space
        point = Point(coords=np.array([1, 0]), space=child)
        result = point.to_absolute()
        
        # After 90 degree rotation, (1, 0) becomes (0, 1) in absolute
        np.testing.assert_array_almost_equal(result.coords, [0, 1])
        # Result should be in absolute/identity space
        assert result.space.parent is None
        np.testing.assert_array_almost_equal(result.space.transform, np.eye(3))

    def test_to_global_vector_ignores_translation(self):
        """Test that vectors ignore translation when going to absolute."""
        parent = Space(transform=translate2D(10, 20), parent=None)
        child = Space(transform=translate2D(5, 5), parent=parent)
        
        vector = Vector(coords=np.array([1, 0]), space=child)
        result = vector.to_absolute()
        
        # Vector should maintain direction, translation ignored (weight=0)
        # Vector (1, 0) should remain (1, 0) in absolute since translations don't affect vectors
        np.testing.assert_array_almost_equal(result.coords, [1, 0])
        # Result should be in absolute/identity space
        assert result.space.parent is None
        np.testing.assert_array_almost_equal(result.space.transform, np.eye(3))


class TestCoordinateToSpace:
    """Tests for converting coordinates between spaces."""

    def test_coordinate_base_class_relative_to(self):
        """Test relative_to with base Coordinate class (for coverage)."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=root)
        space_b = Space(transform=translate2D(0, 3), parent=root)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([0, 0]), space=space_a)
        
        result = coord.relative_to(space_b)
        
        # Should work the same as Point
        np.testing.assert_array_almost_equal(result.coords, [5, -3])
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT

    def test_to_system_same_system(self):
        """Test converting to the same space."""
        space = Space(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = point.relative_to(space)
        
        # Should have same coordinates in same space
        np.testing.assert_array_almost_equal(result.coords, [1, 2])
        assert result.space is space

    def test_to_system_siblings(self):
        """Test converting between sibling coordinate spaces."""
        parent = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=parent)
        space_b = Space(transform=translate2D(0, 3), parent=parent)
        
        # Point at (0, 0) in space A
        point = Point(coords=np.array([0, 0]), space=space_a)
        
        # Convert to space B
        result = point.relative_to(space_b)
        
        # Point is at (5, 0) globally
        # In space B coords (which is at (0, 3)), that's (5, -3)
        np.testing.assert_array_almost_equal(result.coords, [5, -3])
        assert result.space is space_b

    def test_to_system_parent_to_child(self):
        """Test converting from parent to child space."""
        parent = Space(transform=translate2D(10, 5), parent=None)
        child = Space(transform=translate2D(3, 2), parent=parent)
        
        point = Point(coords=np.array([0, 0]), space=parent)
        result = point.relative_to(child)
        
        # Point at parent origin is at (10, 5) globally
        # Child origin is at (13, 7) globally
        # So parent origin in child coords is (-3, -2)
        np.testing.assert_array_almost_equal(result.coords, [-3, -2])
        assert result.space is child

    def test_to_system_child_to_parent(self):
        """Test converting from child to parent space."""
        parent = Space(transform=translate2D(10, 5), parent=None)
        child = Space(transform=translate2D(3, 2), parent=parent)
        
        point = Point(coords=np.array([0, 0]), space=child)
        result = point.relative_to(parent)
        
        # Point at child origin (13, 7 globally) is at (3, 2) in parent
        np.testing.assert_array_almost_equal(result.coords, [3, 2])
        assert result.space is parent

    def test_to_system_with_rotation(self):
        """Test converting between rotated spaces."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=np.eye(3), parent=root)
        space_b = Space(transform=rotate2D(np.pi / 2), parent=root)
        
        # Point at (1, 0) in space A
        point = Point(coords=np.array([1, 0]), space=space_a)
        
        # Convert to space B (rotated 90 degrees)
        result = point.relative_to(space_b)
        
        # (1, 0) in A becomes (0, -1) in B (inverse rotation)
        np.testing.assert_array_almost_equal(result.coords, [0, -1])

    def test_to_system_vector_translation(self):
        """Test that vector conversion ignores translation."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(10, 5), parent=root)
        space_b = Space(transform=translate2D(20, 15), parent=root)
        
        vector = Vector(coords=np.array([1, 0]), space=space_a)
        result = vector.relative_to(space_b)
        
        # Vector direction should be same regardless of translation
        np.testing.assert_array_almost_equal(result.coords, [1, 0])
        assert result.space is space_b

    def test_to_system_vector_rotation(self):
        """Test that vector conversion respects rotation."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=np.eye(3), parent=root)
        space_b = Space(transform=rotate2D(np.pi / 2), parent=root)
        
        vector = Vector(coords=np.array([1, 0]), space=space_a)
        result = vector.relative_to(space_b)
        
        # Vector should rotate
        np.testing.assert_array_almost_equal(result.coords, [0, -1])

    def test_relative_to_no_common_ancestor_raises(self):
        """Test that converting between unrelated spaces raises ValueError."""
        import pytest
        root_a = Space(transform=np.eye(3), parent=None)
        root_b = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=root_a)
        space_b = Space(transform=translate2D(0, 3), parent=root_b)
        point = Point(coords=np.array([1, 0]), space=space_a)
        
        with pytest.raises(ValueError, match="common ancestor"):
            point.relative_to(space_b)

    def test_to_system_complex_hierarchy(self):
        """Test conversion in complex hierarchy."""
        root = Space(transform=np.eye(3), parent=None)
        branch_a = Space(transform=translate2D(10, 0), parent=root)
        leaf_a = Space(transform=rotate2D(np.pi / 4), parent=branch_a)
        
        branch_b = Space(transform=translate2D(0, 10), parent=root)
        
        point = Point(coords=np.array([1, 0]), space=leaf_a)
        result = point.relative_to(branch_b)
        
        # Point goes through: leaf_a -> branch_a -> root -> branch_b
        # In leaf_a: (1, 0)
        # After rotation (π/4): (√2/2, √2/2)
        # After translate (10, 0): (10 + √2/2, √2/2)
        # In branch_b (0, 10): (10 + √2/2, √2/2 - 10)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([10 + sqrt2_over_2, sqrt2_over_2 - 10])
        np.testing.assert_array_almost_equal(result.coords, expected)


class TestDxNArraySupport:
    """Tests for DxN array support - working with multiple points/vectors."""

    def test_transform_multiple_points_translation(self):
        """Test transforming multiple points with translation."""
        transform = translate2D(5, 3)
        # 2D array with 3 points: [x1, x2, x3] and [y1, y2, y3]
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # shape (2, 3)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # All points should be translated
        expected = np.array([[6, 7, 8], [5, 7, 9]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_vectors_translation(self):
        """Test that translation doesn't affect multiple vectors."""
        transform = translate2D(5, 3)
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # shape (2, 3)
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vectors should be unchanged by translation
        expected = np.array([[1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_points_rotation(self):
        """Test transforming multiple points with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([[1, 0, 1], [0, 1, 1]])  # 3 points: (1,0), (0,1), (1,1)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # After 90 degree rotation: (1,0)->(0,1), (0,1)->(-1,0), (1,1)->(-1,1)
        expected = np.array([[0, -1, -1], [1, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_vectors_rotation(self):
        """Test transforming multiple vectors with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([[1, 0], [0, 1]])  # 2 vectors: (1,0) and (0,1)
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vectors should also rotate
        expected = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_points_scale(self):
        """Test transforming multiple points with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([[1, 2], [3, 4]])  # 2 points: (1,3) and (2,4)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([[2, 4], [9, 12]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_3d_points_translation(self):
        """Test transforming 3D points (though transform is 2D embedded in 3D)."""
        # For 3D we'd need a 4x4 transform, but let's test the concept
        # This test documents current 2D-only behavior
        transform = translate2D(5, 3)
        coords = np.array([[1, 2], [3, 4]])  # 2 points in 2D
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([[6, 7], [6, 7]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_with_multiple_coordinates(self):
        """Test Point class with multiple coordinates."""
        space = Space(transform=translate2D(5, 3), parent=None)
        # Create a point with multiple coordinates
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # 3 points
        point = Point(coords=coords, space=space)
        
        result = point.to_absolute()
        
        # All points should be translated
        expected = np.array([[6, 7, 8], [5, 7, 9]])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_with_multiple_coordinates(self):
        """Test Vector class with multiple coordinates."""
        space = Space(transform=translate2D(5, 3), parent=None)
        # Create a vector with multiple coordinates
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # 3 vectors
        vector = Vector(coords=coords, space=space)
        
        result = vector.to_absolute()
        
        # Vectors should be unchanged by translation
        expected = np.array([[1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_single_point_still_works(self):
        """Test that single point (2,) still works after DxN implementation."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])  # shape (2,)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result, expected)


class TestCoordinateOperators:
    """Tests for mathematical operators on Coordinate objects."""

    def test_coordinate_addition(self):
        """Test adding two coordinates."""
        point1 = Point([1, 2])
        point2 = Point([3, 4])
        
        result = point1 + point2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])
        assert result.kind == CoordinateKind.POINT
        assert result.space is point1.space

    def test_coordinate_subtraction(self):
        """Test subtracting two coordinates."""
        point1 = Point([5, 7])
        point2 = Point([2, 3])
        
        result = point1 - point2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])
        assert result.kind == CoordinateKind.POINT

    def test_coordinate_scalar_multiplication(self):
        """Test multiplying coordinate by scalar."""
        point = Point([2, 3])
        
        result = point * 2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])
        assert result.kind == CoordinateKind.POINT

    def test_coordinate_scalar_multiplication_reverse(self):
        """Test multiplying scalar by coordinate."""
        point = Point([2, 3])
        
        result = 2 * point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])

    def test_coordinate_division(self):
        """Test dividing coordinate by scalar."""
        point = Point([4, 6])
        
        result = point / 2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [2, 3])

    def test_coordinate_negation(self):
        """Test negating a coordinate."""
        point = Point([1, -2])
        
        result = -point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [-1, 2])

    def test_coordinate_indexing(self):
        """Test indexing into a coordinate."""
        point = Point([5, 10])
        
        assert point[0] == 5
        assert point[1] == 10

    def test_coordinate_item_assignment(self):
        """Test assigning to coordinate items."""
        point = Point([1, 2])
        point[0] = 5
        
        assert point[0] == 5
        np.testing.assert_array_equal(point.coords, [5, 2])

    def test_coordinate_length(self):
        """Test getting length of coordinate."""
        point = Point([1, 2])
        
        assert len(point) == 2

    def test_coordinate_as_numpy_array(self):
        """Test using coordinate in numpy operations."""
        point = Point([1, 2])
        
        # Should work with numpy functions via __array__
        result = np.sin(point)
        expected = np.sin(np.array([1, 2]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_coordinate_equality(self):
        """Test coordinate equality comparison."""
        point1 = Point([1, 2])
        point2 = Point([1, 2])
        point3 = Point([3, 4])
        
        assert point1 == point2
        assert point1 != point3

    def test_coordinate_absolute_value(self):
        """Test absolute value of coordinate."""
        point = Point([-3, 4])
        
        result = abs(point)
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])

    def test_multiple_coordinates_indexing(self):
        """Test indexing with multiple coordinates."""
        point = Point([[1, 2, 3], [4, 5, 6]])  # DxN array
        
        # Index first dimension
        np.testing.assert_array_equal(point[0], [1, 2, 3])
        # Index specific element
        assert point[1, 2] == 6

    def test_operations_preserve_space(self):
        """Test that operations preserve the coordinate space."""
        space = Space(transform=translate2D(5, 3))
        point = Point([1, 2], space=space)
        
        result = point * 2
        assert result.space is space


class TestCoordinateBaseClassOperations:
    """Tests for arithmetic operations on the base Coordinate class."""

    def test_coordinate_base_class_addition(self):
        """Test addition with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([1, 2]), space=space)
        
        result = coord + np.array([3, 4])
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [4, 6])

    def test_coordinate_base_class_subtraction(self):
        """Test subtraction with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([5, 7]), space=space)
        
        result = coord - np.array([1, 2])
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [4, 5])

    def test_coordinate_base_class_multiplication(self):
        """Test multiplication with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.VECTOR, coords=np.array([2, 3]), space=space)
        
        result = coord * 2
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.VECTOR
        np.testing.assert_array_almost_equal(result.coords, [4, 6])

    def test_coordinate_base_class_division(self):
        """Test division with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([4, 6]), space=space)
        
        result = coord / 2
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [2, 3])

    def test_coordinate_base_class_negation(self):
        """Test negation with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.VECTOR, coords=np.array([1, -2]), space=space)
        
        result = -coord
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.VECTOR
        np.testing.assert_array_almost_equal(result.coords, [-1, 2])

    def test_coordinate_base_class_abs(self):
        """Test absolute value with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([-3, 4]), space=space)
        
        result = abs(coord)
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [3, 4])

    def test_coordinate_base_class_radd(self):
        """Test right addition with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([1, 2]), space=space)
        
        # When left operand is a Python scalar (not numpy), __radd__ is called
        result = 5 + coord
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [6, 7])

    def test_coordinate_base_class_rsub(self):
        """Test right subtraction with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([1, 2]), space=space)
        
        # When left operand is a Python scalar (not numpy), __rsub__ is called
        result = 10 - coord
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [9, 8])

    def test_coordinate_base_class_rmul(self):
        """Test right multiplication with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.VECTOR, coords=np.array([2, 3]), space=space)
        
        result = 2 * coord
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.VECTOR
        np.testing.assert_array_almost_equal(result.coords, [4, 6])

    def test_coordinate_base_class_rtruediv(self):
        """Test right division with base Coordinate class."""
        space = Space(transform=np.eye(3), parent=None)
        coord = Coordinate(kind=CoordinateKind.POINT, coords=np.array([2, 4]), space=space)
        
        # When left operand is a Python scalar (not numpy), __rtruediv__ is called
        result = 8 / coord
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [4, 2])

    def test_coordinate_base_class_coord_addition(self):
        """Test adding two base Coordinate instances."""
        space = Space(transform=np.eye(3), parent=None)
        coord1 = Coordinate(kind=CoordinateKind.POINT, coords=np.array([1, 2]), space=space)
        coord2 = Coordinate(kind=CoordinateKind.POINT, coords=np.array([3, 4]), space=space)
        
        result = coord1 + coord2
        
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.POINT
        np.testing.assert_array_almost_equal(result.coords, [4, 6])


    def test_vector_operations(self):
        """Test operations on vectors preserve vector type."""
        vector = Vector([1, 0])
        
        result = vector * 3
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(result.coords, [3, 0])

    def test_different_spaces_addition_raises_error(self):
        """Test that adding coordinates from different spaces raises an error."""
        space1 = Space(transform=translate2D(5, 0))
        space2 = Space(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], space=space1)
        point2 = Point([3, 4], space=space2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 + point2
        assert "different spaces" in str(exc_info.exception).lower()

    def test_different_spaces_subtraction_raises_error(self):
        """Test that subtracting coordinates from different spaces raises an error."""
        space1 = Space(transform=translate2D(5, 0))
        space2 = Space(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], space=space1)
        point2 = Point([3, 4], space=space2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 - point2
        assert "different spaces" in str(exc_info.exception).lower()

    def test_different_spaces_multiplication_raises_error(self):
        """Test that multiplying coordinates from different spaces raises an error."""
        space1 = Space(transform=translate2D(5, 0))
        space2 = Space(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], space=space1)
        point2 = Point([3, 4], space=space2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 * point2
        assert "different spaces" in str(exc_info.exception).lower()

    def test_different_spaces_division_raises_error(self):
        """Test that dividing coordinates from different spaces raises an error."""
        space1 = Space(transform=translate2D(5, 0))
        space2 = Space(transform=translate2D(0, 5))
        
        point1 = Point([4, 6], space=space1)
        point2 = Point([2, 3], space=space2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 / point2
        assert "different spaces" in str(exc_info.exception).lower()

    def test_same_space_operations_work(self):
        """Test that operations between coordinates in the same space work correctly."""
        space = Space(transform=translate2D(5, 0))
        
        point1 = Point([1, 2], space=space)
        point2 = Point([3, 4], space=space)
        
        # All these should work
        result_add = point1 + point2
        result_sub = point1 - point2
        result_mul = point1 * point2
        result_div = point2 / point1
        
        np.testing.assert_array_equal(result_add.coords, [4, 6])
        np.testing.assert_array_equal(result_sub.coords, [-2, -2])
        np.testing.assert_array_equal(result_mul.coords, [3, 8])
        np.testing.assert_array_equal(result_div.coords, [3, 2])

    def test_coordinate_with_array_addition(self):
        """Test adding a coordinate with a plain array."""
        point = Point([1, 2])
        array = np.array([3, 4])
        
        result = point + array
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])

    def test_coordinate_with_array_subtraction(self):
        """Test subtracting a plain array from coordinate."""
        point = Point([5, 7])
        array = np.array([2, 3])
        
        result = point - array
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])

    def test_right_addition_with_array(self):
        """Test right addition (array + coordinate)."""
        point = Point([1, 2])
        array = np.array([3, 4])
        
        result = array + point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 6])

    def test_right_subtraction_with_array(self):
        """Test right subtraction (array - coordinate)."""
        point = Point([1, 2])
        array = np.array([5, 7])
        
        result = array - point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 5])

    def test_right_division_with_array(self):
        """Test right division (array / coordinate)."""
        point = Point([2, 4])
        array = np.array([8, 12])
        
        result = array / point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 3])

    def test_array_conversion_with_dtype(self):
        """Test __array__ method with dtype specification."""
        point = Point([1.5, 2.7])
        
        # Convert to integer array
        int_array = np.asarray(point, dtype=int)
        np.testing.assert_array_equal(int_array, [1, 2])

    def test_repr(self):
        """Test string representation of coordinate."""
        point = Point([1, 2])
        repr_str = repr(point)
        
        assert "Point" in repr_str
        assert "coords" in repr_str

    def test_equality_with_array(self):
        """Test equality comparison with plain array."""
        point = Point([1, 2])
        array = np.array([1, 2])
        
        assert point == array

    def test_radd_with_scalar(self):
        """Test right addition with scalar (scalar + coordinate)."""
        point = Point([1, 2])
        
        # This should trigger __radd__
        result = 5 + point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [6, 7])

    def test_rsub_with_scalar(self):
        """Test right subtraction with scalar (scalar - coordinate)."""
        point = Point([1, 2])
        
        # This should trigger __rsub__
        result = 10 - point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [9, 8])

    def test_rtruediv_with_scalar(self):
        """Test right division with scalar (scalar / coordinate)."""
        point = Point([2, 4])
        
        # This should trigger __rtruediv__
        result = 8 / point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 2])


class TestPointAndVectorBehavior:
    """Tests to verify Point and Vector behave differently with transformations."""

    def test_point_affected_by_translation(self):
        """Test that points are affected by translation."""
        space = Space(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        # When converting to absolute space, point should be translated
        result = point.to_absolute()
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_unaffected_by_translation(self):
        """Test that vectors are unaffected by translation."""
        space = Space(transform=translate2D(5, 3), parent=None)
        vector = Vector(coords=np.array([1, 2]), space=space)
        
        # When converting to absolute space, vector should not be translated
        result = vector.to_absolute()
        
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_point_scaled_from_origin(self):
        """Test that points are scaled from origin."""
        space = Space(transform=scale2D(2, 2), parent=None)
        point = Point(coords=np.array([3, 4]), space=space)
        
        result = point.to_absolute()
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_scaled(self):
        """Test that vectors are also scaled."""
        space = Space(transform=scale2D(2, 2), parent=None)
        vector = Vector(coords=np.array([3, 4]), space=space)
        
        result = vector.to_absolute()
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_point_and_vector_both_rotate(self):
        """Test that both points and vectors rotate the same way."""
        space = Space(transform=rotate2D(np.pi / 2), parent=None)
        
        point = Point(coords=np.array([1, 0]), space=space)
        vector = Vector(coords=np.array([1, 0]), space=space)
        
        point_result = point.to_absolute()
        vector_result = vector.to_absolute()
        
        # Both should rotate the same
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(point_result.coords, expected)
        np.testing.assert_array_almost_equal(vector_result.coords, expected)


class TestTypePreservation:
    """Tests to ensure Point and Vector types are preserved through operations."""

    def test_point_to_absolute_preserves_type(self):
        """Test that Point.to_absolute() returns a Point instance."""
        space = Space(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = point.to_absolute()
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [6, 5])

    def test_vector_to_absolute_preserves_type(self):
        """Test that Vector.to_absolute() returns a Vector instance."""
        space = Space(transform=translate2D(5, 3), parent=None)
        vector = Vector(coords=np.array([1, 0]), space=space)
        
        result = vector.to_absolute()
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [1, 0])

    def test_point_relative_to_preserves_type(self):
        """Test that Point.relative_to() returns a Point instance."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=root)
        space_b = Space(transform=translate2D(0, 3), parent=root)
        point = Point(coords=np.array([0, 0]), space=space_a)
        
        result = point.relative_to(space_b)
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [5, -3])

    def test_vector_relative_to_preserves_type(self):
        """Test that Vector.relative_to() returns a Vector instance."""
        root = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=root)
        space_b = Space(transform=translate2D(0, 3), parent=root)
        vector = Vector(coords=np.array([1, 0]), space=space_a)
        
        result = vector.relative_to(space_b)
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [1, 0])

    def test_point_addition_preserves_type(self):
        """Test that Point + value returns a Point instance."""
        space = Space(transform=np.eye(3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = point + np.array([3, 4])
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [4, 6])

    def test_vector_addition_preserves_type(self):
        """Test that Vector + value returns a Vector instance."""
        space = Space(transform=np.eye(3), parent=None)
        vector = Vector(coords=np.array([1, 2]), space=space)
        
        result = vector + np.array([3, 4])
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [4, 6])

    def test_point_multiplication_preserves_type(self):
        """Test that Point * scalar returns a Point instance."""
        space = Space(transform=np.eye(3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = point * 2
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [2, 4])

    def test_vector_multiplication_preserves_type(self):
        """Test that Vector * scalar returns a Vector instance."""
        space = Space(transform=np.eye(3), parent=None)
        vector = Vector(coords=np.array([1, 2]), space=space)
        
        result = vector * 2
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [2, 4])

    def test_point_negation_preserves_type(self):
        """Test that -Point returns a Point instance."""
        space = Space(transform=np.eye(3), parent=None)
        point = Point(coords=np.array([1, 2]), space=space)
        
        result = -point
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [-1, -2])

    def test_vector_negation_preserves_type(self):
        """Test that -Vector returns a Vector instance."""
        space = Space(transform=np.eye(3), parent=None)
        vector = Vector(coords=np.array([1, 2]), space=space)
        
        result = -vector
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [-1, -2])

    def test_point_division_preserves_type(self):
        """Test that Point / scalar returns a Point instance."""
        space = Space(transform=np.eye(3), parent=None)
        point = Point(coords=np.array([4, 6]), space=space)
        
        result = point / 2
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [2, 3])

    def test_vector_division_preserves_type(self):
        """Test that Vector / scalar returns a Vector instance."""
        space = Space(transform=np.eye(3), parent=None)
        vector = Vector(coords=np.array([4, 6]), space=space)
        
        result = vector / 2
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [2, 3])

    def test_point_abs_preserves_type(self):
        """Test that abs(Point) returns a Point instance."""
        space = Space(transform=np.eye(3), parent=None)
        point = Point(coords=np.array([-1, -2]), space=space)
        
        result = abs(point)
        
        assert isinstance(result, Point), f"Expected Point but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [1, 2])

    def test_vector_abs_preserves_type(self):
        """Test that abs(Vector) returns a Vector instance."""
        space = Space(transform=np.eye(3), parent=None)
        vector = Vector(coords=np.array([-1, -2]), space=space)
        
        result = abs(vector)
        
        assert isinstance(result, Vector), f"Expected Vector but got {type(result)}"
        np.testing.assert_array_almost_equal(result.coords, [1, 2])


class TestCoordinateDimensionProperties:
    """Tests for the D and N properties of Coordinate."""

    def test_D_single_2d_coordinate(self):
        """Test D property for single 2D coordinate."""
        coord = Coordinate(CoordinateKind.POINT, np.array([1, 2]))
        assert coord.D == 2

    def test_N_single_coordinate(self):
        """Test N property for single coordinate."""
        coord = Coordinate(CoordinateKind.POINT, np.array([1, 2]))
        assert coord.N == 1

    def test_D_multiple_coordinates(self):
        """Test D property for multiple coordinates in DxN format."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array (2 dimensions, 3 points)
        coord = Coordinate(CoordinateKind.POINT, coords)
        assert coord.D == 2

    def test_N_multiple_coordinates(self):
        """Test N property for multiple coordinates in DxN format."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array (2 dimensions, 3 points)
        coord = Coordinate(CoordinateKind.POINT, coords)
        assert coord.N == 3

    def test_D_N_with_point(self):
        """Test D and N properties with Point subclass."""
        point = Point(np.array([3, 4]))
        assert point.D == 2
        assert point.N == 1

    def test_D_N_with_vector(self):
        """Test D and N properties with Vector subclass."""
        vector = Vector(np.array([5, 6]))
        assert vector.D == 2
        assert vector.N == 1

    def test_D_N_with_multiple_points(self):
        """Test D and N properties with multiple points."""
        coords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4 array
        point = Point(coords)
        assert point.D == 2
        assert point.N == 4

    def test_D_N_with_multiple_vectors(self):
        """Test D and N properties with multiple vectors."""
        coords = np.array([[1, 2], [3, 4]])  # 2x2 array
        vector = Vector(coords)
        assert vector.D == 2
        assert vector.N == 2

    def test_D_with_list_input(self):
        """Test D property when coordinate is initialized with a list."""
        coord = Coordinate(CoordinateKind.POINT, [7, 8])
        assert coord.D == 2

    def test_N_with_list_input(self):
        """Test N property when coordinate is initialized with a list."""
        coord = Coordinate(CoordinateKind.POINT, [7, 8])
        assert coord.N == 1

    def test_D_with_tuple_input(self):
        """Test D property when coordinate is initialized with a tuple."""
        coord = Coordinate(CoordinateKind.VECTOR, (9, 10))
        assert coord.D == 2

    def test_N_with_tuple_input(self):
        """Test N property when coordinate is initialized with a tuple."""
        coord = Coordinate(CoordinateKind.VECTOR, (9, 10))
        assert coord.N == 1

