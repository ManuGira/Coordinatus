"""Unit tests for the Space class."""

import numpy as np
from coordinatus.space import Space, create_space
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D


class TestSpaceInit:
    """Tests for Space initialization."""

    def test_space_init_default(self):
        """Test creating a space with default (identity) parameters."""
        space = Space()

        assert space.parent is None
        assert space.tx == 0.0
        assert space.ty == 0.0
        assert space.angle_rad == 0.0
        assert space.sx == 1.0
        assert space.sy == 1.0
        np.testing.assert_array_almost_equal(space.transform, np.eye(3))

    def test_space_init_with_trs(self):
        """Test creating a space with explicit TRS parameters."""
        space = Space(tx=5.0, ty=3.0, angle_rad=np.pi / 4, sx=2.0, sy=1.5)

        assert space.tx == 5.0
        assert space.ty == 3.0
        assert space.angle_rad == np.pi / 4
        assert space.sx == 2.0
        assert space.sy == 1.5
        np.testing.assert_array_almost_equal(
            space.transform, trs2D(5.0, 3.0, np.pi / 4, 2.0, 1.5)
        )

    def test_space_init_with_parent(self):
        """Test creating a space with a parent."""
        parent = Space(tx=5, ty=3)

        child = Space(parent=parent, angle_rad=np.pi / 4)

        assert child.parent is parent
        np.testing.assert_array_almost_equal(child.transform, rotate2D(np.pi / 4))
        np.testing.assert_array_almost_equal(child.parent.transform, translate2D(5, 3))


class TestSpaceTRSMutation:
    """Tests for mutable TRS attributes on Space (key feature of Option A)."""

    def test_set_tx_updates_transform(self):
        """Test that mutating tx immediately updates the transform."""
        space = Space(tx=5, ty=3)
        space.tx = 10
        np.testing.assert_array_almost_equal(space.transform, translate2D(10, 3))

    def test_set_ty_updates_transform(self):
        """Test that mutating ty immediately updates the transform."""
        space = Space(tx=5, ty=3)
        space.ty = 10
        np.testing.assert_array_almost_equal(space.transform, translate2D(5, 10))

    def test_set_angle_rad_updates_transform(self):
        """Test that mutating angle_rad immediately updates the transform."""
        space = Space()
        space.angle_rad = np.pi / 4
        np.testing.assert_array_almost_equal(space.transform, rotate2D(np.pi / 4))

    def test_set_sx_updates_transform(self):
        """Test that mutating sx immediately updates the transform."""
        space = Space()
        space.sx = 2.0
        np.testing.assert_array_almost_equal(space.transform, scale2D(2.0, 1.0))

    def test_set_sy_updates_transform(self):
        """Test that mutating sy immediately updates the transform."""
        space = Space()
        space.sy = 3.0
        np.testing.assert_array_almost_equal(space.transform, scale2D(1.0, 3.0))

    def test_multiple_trs_mutations(self):
        """Test mutating multiple TRS components in sequence."""
        space = Space()
        space.tx = 5
        space.ty = 3
        space.angle_rad = np.pi / 2
        space.sx = 2.0
        space.sy = 0.5
        np.testing.assert_array_almost_equal(
            space.transform, trs2D(5, 3, np.pi / 2, 2.0, 0.5)
        )

    def test_trs_attributes_readable(self):
        """Test that TRS attributes are readable after construction."""
        space = Space(tx=5, ty=3, angle_rad=np.pi / 4, sx=2, sy=1.5)
        assert space.tx == 5
        assert space.ty == 3
        assert space.angle_rad == np.pi / 4
        assert space.sx == 2
        assert space.sy == 1.5


class TestSpaceDimensions:
    """Tests for Space.D_in and Space.D_out properties."""

    def test_D_in_2D_space(self):
        """Test D_in for a 2D TRS space (always 2)."""
        space = Space()
        assert space.D_in == 2

    def test_D_out_2D_space(self):
        """Test D_out for a 2D TRS space (always 2)."""
        space = Space()
        assert space.D_out == 2

    def test_D_in_with_translation_2D(self):
        """Test D_in remains correct with translated 2D space."""
        space = Space(tx=5, ty=10)
        assert space.D_in == 2

    def test_D_out_with_translation_2D(self):
        """Test D_out remains correct with translated 2D space."""
        space = Space(tx=5, ty=10)
        assert space.D_out == 2

    def test_D_in_with_rotation_2D(self):
        """Test D_in remains correct with rotated 2D space."""
        space = Space(angle_rad=np.pi / 4)
        assert space.D_in == 2

    def test_D_out_with_rotation_2D(self):
        """Test D_out remains correct with rotated 2D space."""
        space = Space(angle_rad=np.pi / 4)
        assert space.D_out == 2

    def test_D_in_with_parent(self):
        """Test D_in is independent of parent space."""
        parent = Space(tx=10, ty=5)
        child = Space(parent=parent, angle_rad=np.pi / 2)
        assert child.D_in == 2
        assert parent.D_in == 2

    def test_D_out_with_parent(self):
        """Test D_out is independent of parent space."""
        parent = Space(tx=10, ty=5)
        child = Space(parent=parent, angle_rad=np.pi / 2)
        assert child.D_out == 2
        assert parent.D_out == 2

    def test_D_in_D_out_equal_for_standard_transforms(self):
        """Test that D_in equals D_out for all 2D TRS spaces."""
        spaces = [
            Space(),
            Space(tx=3, ty=4),
            Space(angle_rad=np.pi / 3),
            Space(sx=2, sy=3),
            Space(tx=5, ty=10, angle_rad=np.pi / 4, sx=2, sy=2),
        ]

        for space in spaces:
            assert space.D_in == space.D_out == 2, "D_in and D_out should both be 2 for 2D TRS spaces"


class TestSpaceEquality:
    """Tests for Space equality comparison."""

    def test_same_reference_equal(self):
        """Test that same space object is equal to itself."""
        space = Space(tx=5, ty=3)

        assert space == space
        assert not (space != space)

    def test_different_spaces_not_equal(self):
        """Test that different space objects are not equal by default."""
        space1 = Space(tx=5, ty=3)
        space2 = Space(tx=5, ty=3)

        # Different objects, not the same reference
        assert space1 is not space2
        assert space1 != space2

    def test_identity_spaces_equal(self):
        """Test that two identity spaces (no parent, default TRS) are equal."""
        space1 = Space()  # Default is identity
        space2 = Space()  # Another identity

        assert space1 == space2
        assert not (space1 != space2)

    def test_identity_spaces_with_explicit_values_equal(self):
        """Test identity spaces created with explicit identity values."""
        space1 = Space(tx=0, ty=0, angle_rad=0, sx=1, sy=1)
        space2 = Space(tx=0, ty=0, angle_rad=0, sx=1, sy=1)

        assert space1 == space2

    def test_identity_and_non_identity_not_equal(self):
        """Test that identity space is not equal to non-identity space."""
        identity_space = Space()
        translated_space = Space(tx=5, ty=3)

        assert identity_space != translated_space

    def test_spaces_with_parents_not_equal(self):
        """Test that spaces with parents are not equal (even if transforms are same)."""
        parent = Space()
        space1 = Space(parent=parent, tx=5, ty=3)
        space2 = Space(parent=parent, tx=5, ty=3)

        # Even though they have same TRS and parent, they're different objects
        assert space1 != space2

    def test_space_not_equal_to_non_space(self):
        """Test that space is not equal to non-Space object."""
        space = Space()

        assert space is not None
        assert space != 42
        assert space != "space"
        assert space != np.eye(3)


class TestComputeAbsoluteTransform:
    """Tests for the compute_absolute_transform method."""

    def test_global_transform_no_parent(self):
        """Test absolute transform when there's no parent (should return own transform)."""
        space = Space(tx=3, ty=2)

        result = space.compute_absolute_transform()
        expected = translate2D(3, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_one_parent(self):
        """Test absolute transform with one parent level."""
        # Parent translates by (10, 5)
        parent = Space(tx=10, ty=5)

        # Child translates by (3, 2) relative to parent
        child = Space(parent=parent, tx=3, ty=2)

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
        grandparent = Space(tx=10)

        # Parent: translate (5, 0) relative to grandparent
        parent = Space(parent=grandparent, tx=5)

        # Child: translate (2, 0) relative to parent
        child = Space(parent=parent, tx=2)

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
        parent = Space(sx=2, sy=2)

        # Child: rotate 90 degrees
        child = Space(parent=parent, angle_rad=np.pi / 2)

        result = child.compute_absolute_transform()
        expected = scale2D(2, 2) @ rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_complex_hierarchy(self):
        """Test absolute transform with complex transformations at each level."""
        # Root: translate and rotate
        root = Space(tx=10, ty=5, angle_rad=np.pi / 4)

        # Middle: scale
        middle = Space(parent=root, sx=2, sy=2)

        # Leaf: translate
        leaf = Space(parent=middle, tx=3)

        result = leaf.compute_absolute_transform()
        expected = trs2D(10, 5, np.pi / 4, 1, 1) @ scale2D(2, 2) @ translate2D(3, 0)
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeRelativeTransformTo:
    """Tests for the compute_relative_transform_to method."""

    def test_convert_transform_same_space(self):
        """Test conversion from a space to itself (should be identity)."""
        space = Space(tx=5, ty=3)

        result = space.compute_relative_transform_to(space)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_transform_siblings(self):
        """Test conversion between sibling spaces."""
        parent = Space()

        # Space A: translate by (5, 0)
        space_a = Space(parent=parent, tx=5)

        # Space B: translate by (0, 3)
        space_b = Space(parent=parent, ty=3)

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
        parent = Space(tx=10, ty=5)
        child = Space(parent=parent, tx=3, ty=2)

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
        parent = Space(tx=10, ty=5)
        child = Space(parent=parent, tx=3, ty=2)

        # Convert from child to parent
        result = child.compute_relative_transform_to(parent)

        # Point at (0, 0) in child is at (13, 7) in absolute
        # In parent coordinates, that's (3, 2)
        point_in_child = np.array([0, 0, 1])
        point_in_parent = result @ point_in_child
        np.testing.assert_array_almost_equal(point_in_parent, [3, 2, 1])

    def test_convert_transform_with_rotation(self):
        """Test conversion with rotated coordinate spaces."""
        # Space A: no transformation
        space_a = Space()

        # Space B: rotated 90 degrees
        space_b = Space(angle_rad=np.pi / 2)

        # Convert from A to B
        result = space_a.compute_relative_transform_to(space_b)

        # Point (1, 0) in A should be (0, -1) in B (rotated -90 degrees)
        point_in_a = np.array([1, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [0, -1, 1])

    def test_convert_transform_nested_spaces(self):
        """Test conversion between spaces in different branches of hierarchy."""
        root = Space()

        # Branch A
        branch_a = Space(parent=root, tx=10)

        # Branch B
        branch_b = Space(parent=root, ty=10)

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
        parent = Space(tx=100, ty=100)
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

    def test_factory_returns_space_with_trs_attributes(self):
        """Test that create_space returns a Space with accessible TRS attributes."""
        space = create_space(parent=None, tx=5, ty=3, angle_rad=np.pi / 4, sx=2, sy=1.5)

        assert space.tx == 5
        assert space.ty == 3
        assert space.angle_rad == np.pi / 4
        assert space.sx == 2
        assert space.sy == 1.5

