"""Unit tests for the Space class."""

import numpy as np
import pytest
from coordinatus.space import (
    Space, Space1D, Space2D, Space3D, Space4D, SpaceND, create_space,
    ProjectionSpace, _find_lca, _path_from, _invert_step,
)
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D
from coordinatus.transforms.dimension import project_xyz_to_xy


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


class TestFindLCA:
    """Tests for the _find_lca helper."""

    def test_lca_two_siblings(self):
        """LCA of two siblings is their direct parent."""
        parent = Space2D()
        a = Space(transform=translate2D(1, 0), parent=parent)
        b = Space(transform=translate2D(0, 1), parent=parent)
        assert _find_lca(a, b) is parent

    def test_lca_ancestor_and_descendant(self):
        """LCA when one node is a direct ancestor of the other is the ancestor."""
        root = Space2D()
        child = Space(transform=translate2D(1, 0), parent=root)
        assert _find_lca(root, child) is root
        assert _find_lca(child, root) is root

    def test_lca_deeper_ancestor(self):
        """LCA when ancestor is two levels up."""
        root = Space2D()
        mid = Space(transform=translate2D(1, 0), parent=root)
        leaf = Space(transform=translate2D(2, 0), parent=mid)
        assert _find_lca(root, leaf) is root
        assert _find_lca(leaf, root) is root

    def test_lca_node_with_itself(self):
        """LCA of a node with itself is the node."""
        node = Space2D()
        assert _find_lca(node, node) is node

    def test_lca_cousins(self):
        """LCA of two cousins (sharing a grandparent) is the grandparent."""
        root = Space2D()
        branch_a = Space(transform=translate2D(5, 0), parent=root)
        branch_b = Space(transform=translate2D(0, 5), parent=root)
        leaf_a = Space(transform=translate2D(1, 0), parent=branch_a)
        leaf_b = Space(transform=translate2D(0, 1), parent=branch_b)
        assert _find_lca(leaf_a, leaf_b) is root

    def test_lca_unrelated_raises(self):
        """_find_lca raises ValueError for unrelated spaces."""
        root_a = Space2D()
        root_b = Space2D()
        with pytest.raises(ValueError):
            _find_lca(root_a, root_b)


class TestPathFrom:
    """Tests for the _path_from helper."""

    def test_path_direct_child(self):
        """Path from parent to direct child is [child]."""
        parent = Space2D()
        child = Space(transform=translate2D(1, 0), parent=parent)
        assert _path_from(parent, child) == [child]

    def test_path_two_steps(self):
        """Path from grandparent to grandchild is [child, grandchild]."""
        root = Space2D()
        child = Space(transform=translate2D(1, 0), parent=root)
        grandchild = Space(transform=translate2D(2, 0), parent=child)
        assert _path_from(root, grandchild) == [child, grandchild]

    def test_path_to_self_is_empty(self):
        """Path from a node to itself is an empty list."""
        node = Space2D()
        assert _path_from(node, node) == []

    def test_path_order_is_top_down(self):
        """Path is ordered from child-of-ancestor down to descendant."""
        root = Space2D()
        a = Space(transform=np.eye(3), parent=root)
        b = Space(transform=np.eye(3), parent=a)
        c = Space(transform=np.eye(3), parent=b)
        path = _path_from(root, c)
        assert path == [a, b, c]


class TestInvertStep:
    """Tests for the _invert_step helper."""

    def test_square_transform_returns_inverse(self):
        """Standard square transform returns the matrix inverse."""
        t = translate2D(3, 4)
        node = Space(transform=t, parent=Space2D())
        result = _invert_step(node)
        expected = np.linalg.inv(t)
        np.testing.assert_array_almost_equal(result, expected)

    def test_projection_space_returns_projection_matrix(self):
        """ProjectionSpace returns projection_matrix, not pseudo-inverse."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        result = _invert_step(screen)
        np.testing.assert_array_almost_equal(result, proj)

    def test_non_square_non_projection_raises(self):
        """Non-square transform on a plain Space raises ValueError."""
        non_square = project_xyz_to_xy()  # 3x4
        node = Space(transform=non_square, parent=Space3D())
        with pytest.raises(ValueError):
            _invert_step(node)


class TestProjectionSpace:
    """Tests for the ProjectionSpace class."""

    def test_stores_projection_matrix(self):
        """Constructor stores projection_matrix attribute."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        np.testing.assert_array_almost_equal(screen.projection_matrix, proj)

    def test_d_in_is_child_dimension(self):
        """D_in is the child (screen) dimension (2 for 3D→2D)."""
        proj = project_xyz_to_xy()  # 3x4: parent=3D, child=2D
        screen = ProjectionSpace(projection_matrix=proj, parent=Space3D())
        assert screen.D_in == 2

    def test_d_out_is_parent_dimension(self):
        """D_out is the parent dimension (3 for 3D→2D)."""
        proj = project_xyz_to_xy()
        screen = ProjectionSpace(projection_matrix=proj, parent=Space3D())
        assert screen.D_out == 3

    def test_parent_is_set(self):
        """Parent is correctly stored."""
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=project_xyz_to_xy(), parent=world)
        assert screen.parent is world

    def test_is_space_instance(self):
        """ProjectionSpace is a subclass of Space."""
        screen = ProjectionSpace(projection_matrix=project_xyz_to_xy(), parent=Space3D())
        assert isinstance(screen, Space)

    def test_compute_absolute_transform_from_child(self):
        """compute_absolute_transform() chains pseudo-inverse upward correctly."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        pseudo_inv = np.linalg.pinv(proj)
        # world is identity, so absolute = world.transform @ pseudo_inv = I @ pseudo_inv
        expected = np.eye(4) @ pseudo_inv
        np.testing.assert_array_almost_equal(screen.compute_absolute_transform(), expected)

    def test_transform_is_pseudo_inverse(self):
        """The stored transform is the pseudo-inverse of projection_matrix."""
        proj = project_xyz_to_xy()
        screen = ProjectionSpace(projection_matrix=proj, parent=Space3D())
        expected_pinv = np.linalg.pinv(proj)
        np.testing.assert_array_almost_equal(screen.transform, expected_pinv)


class TestSpaceUid:
    """Tests for the uid attribute on Space and its subclasses."""

    def test_uid_auto_generated_when_none(self):
        """uid is auto-generated when not provided."""
        space = Space(transform=np.eye(3))
        assert space.uid is not None
        assert isinstance(space.uid, str)

    def test_uid_auto_generated_format(self):
        """Auto-generated uid has the format 'Space_<id>'."""
        space = Space(transform=np.eye(3))
        assert space.uid == f"Space_{id(space)}"

    def test_uid_custom_value_stored(self):
        """Custom uid is stored as provided."""
        space = Space(transform=np.eye(3), uid="my_space")
        assert space.uid == "my_space"

    def test_uid_distinct_across_instances(self):
        """Two spaces without explicit uid have different uid values."""
        a = Space(transform=np.eye(3))
        b = Space(transform=np.eye(3))
        assert a.uid != b.uid

    def test_uid_space1d_default(self):
        """Space1D auto-generates a uid when none is provided."""
        space = Space1D()
        assert space.uid == f"Space_{id(space)}"

    def test_uid_space1d_custom(self):
        """Space1D stores a custom uid."""
        space = Space1D(uid="root_1d")
        assert space.uid == "root_1d"

    def test_uid_space2d_default(self):
        """Space2D auto-generates a uid when none is provided."""
        space = Space2D()
        assert space.uid == f"Space_{id(space)}"

    def test_uid_space2d_custom(self):
        """Space2D stores a custom uid."""
        space = Space2D(uid="world")
        assert space.uid == "world"

    def test_uid_space3d_custom(self):
        """Space3D stores a custom uid."""
        space = Space3D(uid="scene_root")
        assert space.uid == "scene_root"

    def test_uid_space4d_custom(self):
        """Space4D stores a custom uid."""
        space = Space4D(uid="hyper_root")
        assert space.uid == "hyper_root"

    def test_uid_spacend_custom(self):
        """SpaceND stores a custom uid."""
        space = SpaceND(5, uid="nd_root")
        assert space.uid == "nd_root"

    def test_uid_spacend_default(self):
        """SpaceND auto-generates a uid when none is provided."""
        space = SpaceND(3)
        assert space.uid == f"Space_{id(space)}"

    def test_uid_create_space_default(self):
        """create_space auto-generates a uid when none is provided."""
        space = create_space()
        assert space.uid == f"Space_{id(space)}"

    def test_uid_create_space_custom(self):
        """create_space stores a custom uid."""
        space = create_space(uid="factory_space")
        assert space.uid == "factory_space"

    def test_uid_projection_space_custom(self):
        """ProjectionSpace stores a custom uid."""
        from coordinatus.transforms.dimension import project_xyz_to_xy
        screen = ProjectionSpace(
            projection_matrix=project_xyz_to_xy(),
            parent=Space3D(),
            uid="screen",
        )
        assert screen.uid == "screen"

    def test_uid_not_used_in_equality(self):
        """uid does not affect space equality — two identity spaces with different uid are equal."""
        a = Space2D(uid="a")
        b = Space2D(uid="b")
        assert a == b


class TestComputeRelativeTransformToWithProjection:
    """Tests for compute_relative_transform_to with ProjectionSpace in the path."""

    def test_siblings_regression(self):
        """Two siblings give the same result as the old inv(absolute) @ absolute formula."""
        root = Space2D()
        a = Space(transform=translate2D(5, 0), parent=root)
        b = Space(transform=translate2D(0, 3), parent=root)
        result = a.compute_relative_transform_to(b)
        # Old formula reference
        expected = np.linalg.inv(b.compute_absolute_transform()) @ a.compute_absolute_transform()
        np.testing.assert_array_almost_equal(result, expected)

    def test_3d_world_to_projection_space(self):
        """world.compute_relative_transform_to(screen) applies the projection matrix."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        result = world.compute_relative_transform_to(screen)
        # Going from world (=root) to screen: just apply _invert_step(screen) = proj
        np.testing.assert_array_almost_equal(result, proj)

    def test_projection_space_to_3d_world(self):
        """screen.compute_relative_transform_to(world) applies the pseudo-inverse."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        result = screen.compute_relative_transform_to(world)
        expected = np.linalg.pinv(proj)
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_projected_through_scene_graph(self):
        """A 3D point in world space projected through the scene graph gives xy coords."""
        proj = project_xyz_to_xy()
        world = Space3D()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        T = world.compute_relative_transform_to(screen)
        pt_homogeneous = np.array([3.0, 5.0, 7.0, 1.0])
        result = T @ pt_homogeneous
        # result is 3-vector (2D homogeneous); normalize and strip weight
        result_cart = result[:2] / result[2]
        np.testing.assert_array_almost_equal(result_cart, [3.0, 5.0])

    def test_sibling_of_projection_space(self):
        """Convert between two siblings where one is a ProjectionSpace."""
        world = Space3D()
        proj = project_xyz_to_xy()
        screen = ProjectionSpace(projection_matrix=proj, parent=world)
        camera = Space(transform=np.eye(4), parent=world)
        # camera → screen: go up to world, then down into screen
        T = camera.compute_relative_transform_to(screen)
        # camera has identity transform, world is root, so camera coords == world coords
        # => result should equal project_xyz_to_xy() applied directly
        np.testing.assert_array_almost_equal(T, proj)
