"""Unit tests for the Space class."""

import numpy as np
from coordinatus.space import Space, Space1D, Space2D, Space3D, Space4D, SpaceND, create_space
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D


class TestSpaceInit:
    """Tests for Space initialization."""

    def test_space_init_no_parent(self):
        """Test creating a space without a parent."""
        transform = np.eye(3)
        space = Space(transform=transform, parent=None)
        
        assert space.parent is None
        np.testing.assert_array_equal(space.transform, transform)

    def test_space_init_with_parent(self):
        """Test creating a space with a parent."""
        parent_transform = translate2D(5, 3)
        parent = Space(transform=parent_transform, parent=None)
        
        child_transform = rotate2D(np.pi / 4)
        child = Space(transform=child_transform, parent=parent)
        
        assert child.parent is parent
        np.testing.assert_array_equal(child.transform, child_transform)
        np.testing.assert_array_equal(child.parent.transform, parent_transform)

    def test_space_init_none_transform_raises(self):
        """Test that passing None as transform raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="transform must be a numpy array"):
            Space(transform=None)  # type: ignore[arg-type]


class TestSpaceDimensions:
    """Tests for Space.D_in and Space.D_out properties."""

    def test_D_in_2D_space(self):
        """Test D_in for a 2D space (3x3 transformation matrix)."""
        space = Space(transform=np.eye(3))
        assert space.D_in == 2

    def test_D_out_2D_space(self):
        """Test D_out for a 2D space (3x3 transformation matrix)."""
        space = Space(transform=np.eye(3))
        assert space.D_out == 2

    def test_D_in_3D_space(self):
        """Test D_in for a 3D space (4x4 transformation matrix)."""
        space = Space(transform=np.eye(4))
        assert space.D_in == 3

    def test_D_out_3D_space(self):
        """Test D_out for a 3D space (4x4 transformation matrix)."""
        space = Space(transform=np.eye(4))
        assert space.D_out == 3

    def test_D_in_with_translation_2D(self):
        """Test D_in remains correct with translated 2D space."""
        space = Space(transform=translate2D(5, 10))
        assert space.D_in == 2

    def test_D_out_with_translation_2D(self):
        """Test D_out remains correct with translated 2D space."""
        space = Space(transform=translate2D(5, 10))
        assert space.D_out == 2

    def test_D_in_with_rotation_2D(self):
        """Test D_in remains correct with rotated 2D space."""
        space = Space(transform=rotate2D(np.pi / 4))
        assert space.D_in == 2

    def test_D_out_with_rotation_2D(self):
        """Test D_out remains correct with rotated 2D space."""
        space = Space(transform=rotate2D(np.pi / 4))
        assert space.D_out == 2

    def test_D_in_with_parent(self):
        """Test D_in is independent of parent space."""
        parent = Space(transform=translate2D(10, 5))
        child = Space(transform=rotate2D(np.pi / 2), parent=parent)
        assert child.D_in == 2
        assert parent.D_in == 2

    def test_D_out_with_parent(self):
        """Test D_out is independent of parent space."""
        parent = Space(transform=translate2D(10, 5))
        child = Space(transform=rotate2D(np.pi / 2), parent=parent)
        assert child.D_out == 2
        assert parent.D_out == 2

    def test_D_in_D_out_equal_for_standard_transforms(self):
        """Test that D_in equals D_out for standard (non-projection) transformations."""
        spaces = [
            Space(transform=np.eye(3)),
            Space(transform=translate2D(3, 4)),
            Space(transform=rotate2D(np.pi / 3)),
            Space(transform=scale2D(2, 3)),
            Space(transform=trs2D(5, 10, np.pi / 4, 2, 2)),
        ]
        
        for space in spaces:
            assert space.D_in == space.D_out, "D_in and D_out should be equal for standard transforms"

    def test_D_in_D_out_different_for_projection(self):
        """Test that D_in != D_out for dimension-changing transformations (projections)."""
        # Create a 3x4 projection matrix (projects 3D to 2D)
        projection_3d_to_2d = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]])
        
        space = Space(transform=projection_3d_to_2d)
        assert space.D_in == 3  # Input is 3D
        assert space.D_out == 2  # Output is 2D
        assert space.D_in != space.D_out

    def test_D_in_1D_space(self):
        """Test D_in for a 1D space (2x2 transformation matrix)."""
        space = Space(transform=np.eye(2))
        assert space.D_in == 1

    def test_D_out_1D_space(self):
        """Test D_out for a 1D space (2x2 transformation matrix)."""
        space = Space(transform=np.eye(2))
        assert space.D_out == 1



class TestSpaceEquality:
    """Tests for Space equality comparison."""

    def test_same_reference_equal(self):
        """Test that same space object is equal to itself."""
        space = Space(transform=translate2D(5, 3))
        
        assert space == space
        assert not (space != space)

    def test_different_spaces_not_equal(self):
        """Test that different space objects are not equal by default."""
        space1 = Space(transform=translate2D(5, 3))
        space2 = Space(transform=translate2D(5, 3))
        
        # Different objects, not the same reference
        assert space1 is not space2
        assert space1 != space2

    def test_identity_spaces_equal(self):
        """Test that two identity spaces (no parent, identity transform) are equal."""
        space1 = Space2D()  # 2D identity
        space2 = Space2D()  # Another 2D identity
        
        assert space1 == space2
        assert not (space1 != space2)

    def test_identity_spaces_with_explicit_identity_equal(self):
        """Test identity spaces created explicitly."""
        space1 = Space(transform=np.eye(3), parent=None)
        space2 = Space(transform=np.eye(3), parent=None)
        
        assert space1 == space2

    def test_identity_and_non_identity_not_equal(self):
        """Test that identity space is not equal to non-identity space."""
        identity_space = Space2D()
        translated_space = Space(transform=translate2D(5, 3))
        
        assert identity_space != translated_space

    def test_spaces_with_parents_not_equal(self):
        """Test that spaces with parents are not equal (even if transforms are same)."""
        parent = Space2D()
        space1 = Space(transform=translate2D(5, 3), parent=parent)
        space2 = Space(transform=translate2D(5, 3), parent=parent)
        
        # Even though they have same transform and parent, they're different objects
        assert space1 != space2

    def test_space_not_equal_to_non_space(self):
        """Test that space is not equal to non-Space object."""
        space = Space2D()
        
        assert space is not None
        assert space != 42
        assert space != "space"
        assert space != np.eye(3)


class TestGetRoot:
    """Tests for the get_root method."""

    def test_get_root_no_parent(self):
        """A space with no parent is its own root."""
        space = Space(transform=np.eye(3), parent=None)
        assert space.get_root() is space

    def test_get_root_one_level(self):
        """Root of a child space is its parent."""
        root = Space(transform=np.eye(3), parent=None)
        child = Space(transform=translate2D(5, 3), parent=root)
        assert child.get_root() is root

    def test_get_root_nested(self):
        """Root is the topmost ancestor in a deep hierarchy."""
        root = Space(transform=np.eye(3), parent=None)
        middle = Space(transform=translate2D(5, 0), parent=root)
        leaf = Space(transform=translate2D(0, 5), parent=middle)
        assert leaf.get_root() is root
        assert middle.get_root() is root

    def test_separate_hierarchies_different_roots(self):
        """Spaces in different hierarchies have different roots."""
        root_a = Space(transform=np.eye(3), parent=None)
        root_b = Space(transform=np.eye(3), parent=None)
        child_a = Space(transform=translate2D(1, 0), parent=root_a)
        child_b = Space(transform=translate2D(0, 1), parent=root_b)
        assert child_a.get_root() is not child_b.get_root()


class TestComputeAbsoluteTransform:
    """Tests for the compute_absolute_transform method."""

    def test_global_transform_no_parent(self):
        """Test absolute transform when there's no parent (should return own transform)."""
        transform = translate2D(3, 2)
        space = Space(transform=transform, parent=None)
        
        result = space.compute_absolute_transform()
        expected = transform
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_one_parent(self):
        """Test absolute transform with one parent level."""
        # Parent translates by (10, 5)
        parent = Space(transform=translate2D(10, 5), parent=None)
        
        # Child translates by (3, 2) relative to parent
        child = Space(transform=translate2D(3, 2), parent=parent)
        
        result = child.compute_absolute_transform()
        # Expected: parent @ child
        expected = translate2D(10, 5) @ translate2D(3, 2)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Verify a point transforms correctly
        point = np.array([0, 0, 1])
        transformed = result @ point
        # Point should be at (13, 7) in absolute space
        np.testing.assert_array_almost_equal(transformed, [13, 7, 1])

    def test_global_transform_nested_hierarchy(self):
        """Test absolute transform with multiple nested parents."""
        # Grandparent: translate (10, 0)
        grandparent = Space(transform=translate2D(10, 0), parent=None)
        
        # Parent: translate (5, 0) relative to grandparent
        parent = Space(transform=translate2D(5, 0), parent=grandparent)
        
        # Child: translate (2, 0) relative to parent
        child = Space(transform=translate2D(2, 0), parent=parent)
        
        result = child.compute_absolute_transform()
        
        # Expected: grandparent @ parent @ child
        expected = translate2D(10, 0) @ translate2D(5, 0) @ translate2D(2, 0)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Point at origin should end up at (17, 0)
        point = np.array([0, 0, 1])
        transformed = result @ point
        np.testing.assert_array_almost_equal(transformed, [17, 0, 1])

    def test_global_transform_with_rotation_and_scale(self):
        """Test absolute transform with rotation and scaling."""
        # Parent: scale by 2
        parent = Space(transform=scale2D(2, 2), parent=None)
        
        # Child: rotate 90 degrees
        child = Space(transform=rotate2D(np.pi / 2), parent=parent)
        
        result = child.compute_absolute_transform()
        expected = scale2D(2, 2) @ rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_complex_hierarchy(self):
        """Test absolute transform with complex transformations at each level."""
        # Root: translate and rotate
        root = Space(transform=trs2D(10, 5, np.pi / 4, 1, 1), parent=None)
        
        # Middle: scale
        middle = Space(transform=scale2D(2, 2), parent=root)
        
        # Leaf: translate
        leaf = Space(transform=translate2D(3, 0), parent=middle)
        
        result = leaf.compute_absolute_transform()
        expected = trs2D(10, 5, np.pi / 4, 1, 1) @ scale2D(2, 2) @ translate2D(3, 0)
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeRelativeTransformTo:
    """Tests for the compute_relative_transform_to method."""

    def test_convert_transform_same_space(self):
        """Test conversion from a space to itself (should be identity)."""
        space = Space(transform=translate2D(5, 3), parent=None)
        
        result = space.compute_relative_transform_to(space)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_transform_siblings(self):
        """Test conversion between sibling spaces."""
        parent = Space(transform=np.eye(3), parent=None)
        
        # Space A: translate by (5, 0)
        space_a = Space(transform=translate2D(5, 0), parent=parent)
        
        # Space B: translate by (0, 3)
        space_b = Space(transform=translate2D(0, 3), parent=parent)
        
        # Convert from A to B
        result = space_a.compute_relative_transform_to(space_b)
        
        # To go from A to B: go to absolute, then to B
        # Absolute of A: (5, 0)
        # Inverse of B: (-0, -3)
        # So point at (0,0) in A is at (5, 0) in absolute, which is (5, -3) in B
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [5, -3, 1])

    def test_convert_transform_parent_to_child(self):
        """Test conversion from parent to child coordinate space."""
        parent = Space(transform=translate2D(10, 5), parent=None)
        child = Space(transform=translate2D(3, 2), parent=parent)
        
        # Convert from parent to child
        result = parent.compute_relative_transform_to(child)
        
        # Point at (0, 0) in parent is at (0, 0) in absolute (since parent has absolute (10,5))
        # In child coordinates, we need to invert the child's absolute transform
        point_in_parent = np.array([0, 0, 1])
        point_in_child = result @ point_in_parent
        
        # Parent's origin is at (10, 5) in absolute
        # Child's origin is at (13, 7) in absolute
        # So parent origin in child coords is at (-3, -2)
        np.testing.assert_array_almost_equal(point_in_child, [-3, -2, 1])

    def test_convert_transform_child_to_parent(self):
        """Test conversion from child to parent coordinate space."""
        parent = Space(transform=translate2D(10, 5), parent=None)
        child = Space(transform=translate2D(3, 2), parent=parent)
        
        # Convert from child to parent
        result = child.compute_relative_transform_to(parent)
        
        # Point at (0, 0) in child is at (13, 7) in absolute
        # In parent coordinates, that's (3, 2)
        point_in_child = np.array([0, 0, 1])
        point_in_parent = result @ point_in_child
        np.testing.assert_array_almost_equal(point_in_parent, [3, 2, 1])

    def test_convert_transform_with_rotation(self):
        """Test conversion with rotated coordinate spaces."""
        root = Space(transform=np.eye(3), parent=None)

        # Space A: no transformation
        space_a = Space(transform=np.eye(3), parent=root)
        
        # Space B: rotated 90 degrees
        space_b = Space(transform=rotate2D(np.pi / 2), parent=root)
        
        # Convert from A to B
        result = space_a.compute_relative_transform_to(space_b)
        
        # Point (1, 0) in A should be (0, -1) in B (rotated -90 degrees)
        point_in_a = np.array([1, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [0, -1, 1])

    def test_convert_transform_no_common_ancestor_raises(self):
        """Test that converting between unrelated spaces raises ValueError."""
        import pytest
        root_a = Space(transform=np.eye(3), parent=None)
        root_b = Space(transform=np.eye(3), parent=None)
        space_a = Space(transform=translate2D(5, 0), parent=root_a)
        space_b = Space(transform=translate2D(0, 3), parent=root_b)
        
        with pytest.raises(ValueError, match="common ancestor"):
            space_a.compute_relative_transform_to(space_b)

    def test_convert_transform_nested_spaces(self):
        """Test conversion between spaces in different branches of hierarchy."""
        root = Space(transform=np.eye(3), parent=None)
        
        # Branch A
        branch_a = Space(transform=translate2D(10, 0), parent=root)
        
        # Branch B
        branch_b = Space(transform=translate2D(0, 10), parent=root)
        
        # Convert from branch_a to branch_b
        result = branch_a.compute_relative_transform_to(branch_b)
        
        # Point at (0, 0) in branch_a is at (10, 0) in absolute
        # In branch_b coords, that's (10, -10)
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [10, -10, 1])


class TestCreateSpace:
    """Tests for the create_space function."""

    def test_factory_identity(self):
        """Test factory with identity transformation."""
        space = create_space(parent=None, tx=0, ty=0, angle_rad=0, sx=1, sy=1)
        
        np.testing.assert_array_almost_equal(space.transform, np.eye(3))
        assert space.parent is None

    def test_factory_translation(self):
        """Test factory with translation."""
        space = create_space(parent=None, tx=5, ty=3)
        
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(space.transform, expected)

    def test_factory_rotation(self):
        """Test factory with rotation."""
        space = create_space(parent=None, angle_rad=np.pi / 2)
        
        expected = rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(space.transform, expected)

    def test_factory_scale(self):
        """Test factory with scaling."""
        space = create_space(parent=None, sx=2, sy=3)
        
        expected = scale2D(2, 3)
        np.testing.assert_array_almost_equal(space.transform, expected)

    def test_factory_full_trs(self):
        """Test factory with all TRS parameters."""
        space = create_space(
            parent=None,
            tx=10, ty=5,
            angle_rad=np.pi / 4,
            sx=2, sy=1.5
        )
        
        expected = trs2D(10, 5, np.pi / 4, 2, 1.5)
        np.testing.assert_array_almost_equal(space.transform, expected)

    def test_factory_with_parent(self):
        """Test factory with a parent space."""
        parent = Space(transform=translate2D(100, 100), parent=None)
        child = create_space(parent=parent, tx=5, ty=3)
        
        assert child.parent is parent
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(child.transform, expected)

    def test_factory_default_parameters(self):
        """Test factory with default parameters."""
        space = create_space(parent=None)
        
        # Should create identity transform with default params
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(space.transform, expected)

    def test_factory_builds_hierarchy(self):
        """Test using factory to build a space hierarchy."""
        root = create_space(parent=None, tx=10, ty=10)
        child = create_space(parent=root, tx=5, ty=0, angle_rad=np.pi / 2)
        grandchild = create_space(parent=child, sx=2, sy=2)
        
        # Test hierarchy is connected
        assert child.parent is root
        assert grandchild.parent is child
        
        # Test absolute transform of grandchild
        absolute_t = grandchild.compute_absolute_transform()
        expected = trs2D(10, 10, 0, 1, 1) @ trs2D(5, 0, np.pi / 2, 1, 1) @ trs2D(0, 0, 0, 2, 2)
        np.testing.assert_array_almost_equal(absolute_t, expected)


class TestSpaceSubclasses:
    """Tests for Space1D, Space2D, Space3D, Space4D, SpaceND subclasses."""

    def test_space4d_has_identity_transform(self):
        """Test that Space4D initialises with a 5x5 identity matrix."""
        space = Space4D()
        np.testing.assert_array_equal(space.transform, np.eye(5))

    def test_space4d_dimensionality(self):
        """Test that Space4D reports D_in == D_out == 4."""
        space = Space4D()
        assert space.D_in == 4
        assert space.D_out == 4

    def test_space4d_no_parent_by_default(self):
        """Test that Space4D has no parent by default."""
        space = Space4D()
        assert space.parent is None

    def test_space4d_with_parent(self):
        """Test that Space4D accepts an optional parent."""
        parent = Space4D()
        child = Space4D(parent=parent)
        assert child.parent is parent

    def test_space4d_equality(self):
        """Test that two independent Space4D instances are equal (both identity)."""
        assert Space4D() == Space4D()

    def test_space4d_is_space_instance(self):
        """Test that Space4D is a subclass of Space."""
        assert isinstance(Space4D(), Space)

    def test_spacend_has_correct_identity_transform(self):
        """Test that SpaceND(N) initialises with a (N+1)x(N+1) identity matrix."""
        for n in [1, 2, 3, 4, 5, 10]:
            space = SpaceND(n)
            np.testing.assert_array_equal(space.transform, np.eye(n + 1))

    def test_spacend_dimensionality(self):
        """Test that SpaceND(N) reports D_in == D_out == N."""
        for n in [1, 2, 3, 4, 5]:
            space = SpaceND(n)
            assert space.D_in == n
            assert space.D_out == n

    def test_spacend_no_parent_by_default(self):
        """Test that SpaceND has no parent by default."""
        assert SpaceND(3).parent is None

    def test_spacend_with_parent(self):
        """Test that SpaceND accepts an optional parent."""
        parent = SpaceND(3)
        child = SpaceND(3, parent=parent)
        assert child.parent is parent

    def test_spacend_equality(self):
        """Test that two independent SpaceND(N) instances with the same N are equal."""
        assert SpaceND(3) == SpaceND(3)

    def test_spacend_different_dims_not_equal(self):
        """Test that SpaceND instances with different N are not equal."""
        assert SpaceND(2) != SpaceND(3)

    def test_spacend_is_space_instance(self):
        """Test that SpaceND is a subclass of Space."""
        assert isinstance(SpaceND(2), Space)

    def test_spacend_2_equivalent_to_space2d(self):
        """Test that SpaceND(2) is equal to Space2D()."""
        assert SpaceND(2) == Space2D()

    def test_spacend_4_equivalent_to_space4d(self):
        """Test that SpaceND(4) is equal to Space4D()."""
        assert SpaceND(4) == Space4D()

    def test_space1d_has_identity_transform(self):
        """Test that Space1D initialises with a 2x2 identity matrix."""
        space = Space1D()
        np.testing.assert_array_equal(space.transform, np.eye(2))
        assert space.D_in == 1
        assert space.parent is None

    def test_space3d_has_identity_transform(self):
        """Test that Space3D initialises with a 4x4 identity matrix."""
        space = Space3D()
        np.testing.assert_array_equal(space.transform, np.eye(4))
        assert space.D_in == 3
        assert space.parent is None
