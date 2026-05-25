"""Unit tests for the serializer module (to_json / from_json)."""

import numpy as np
import pytest
from coordinatus.space import Space, Space2D, Space3D
from coordinatus.coordinate import Coordinate, Point, Vector
from coordinatus.coordinatus_types import CoordinateKind
from coordinatus.serializer import to_json, from_json
from coordinatus.transforms import translate2D, rotate2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_scene():
    """Return a minimal scene: one root Space2D and one Point."""
    root = Space2D(uid="root")
    pt = Point(coords=np.array([1.0, 2.0]), space=root)
    return [root], [pt]


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_returns_dict_with_expected_keys(self):
        spaces, coords = _make_simple_scene()
        data = to_json(spaces, coords)
        assert set(data.keys()) == {"spaces", "points"}

    def test_space_fields(self):
        spaces, coords = _make_simple_scene()
        data = to_json(spaces, coords)
        assert len(data["spaces"]) == 1
        s = data["spaces"][0]
        assert s["uid"] == "root"
        assert s["parent_uid"] == ""
        assert isinstance(s["transform"], list)

    def test_space_transform_values(self):
        spaces, coords = _make_simple_scene()
        data = to_json(spaces, coords)
        transform = np.array(data["spaces"][0]["transform"])
        np.testing.assert_array_almost_equal(transform, np.eye(3))

    def test_coordinate_fields(self):
        spaces, coords = _make_simple_scene()
        data = to_json(spaces, coords)
        assert len(data["points"]) == 1
        c = data["points"][0]
        assert c["kind"] == CoordinateKind.POINT.value
        assert c["coords"] == [1.0, 2.0]
        assert c["space_uid"] == "root"

    def test_vector_kind_serialized(self):
        root = Space2D(uid="r")
        vec = Vector(coords=np.array([3.0, 4.0]), space=root)
        data = to_json([root], [vec])
        assert data["points"][0]["kind"] == CoordinateKind.VECTOR.value

    def test_parent_uid_populated(self):
        root = Space2D(uid="root")
        child = Space(transform=translate2D(1, 2), parent=root, uid="child")
        data = to_json([root, child], [])
        child_dict = next(s for s in data["spaces"] if s["uid"] == "child")
        assert child_dict["parent_uid"] == "root"

    def test_empty_inputs(self):
        data = to_json([], [])
        assert data == {"spaces": [], "points": []}

    def test_multiple_coordinates(self):
        root = Space2D(uid="r")
        pts = [Point(coords=np.array([float(i), float(i)]), space=root) for i in range(3)]
        data = to_json([root], pts)
        assert len(data["points"]) == 3

    def test_3d_space_serialized(self):
        root = Space3D(uid="r3d")
        pt = Point(coords=np.array([1.0, 2.0, 3.0]), space=root)
        data = to_json([root], [pt])
        transform = np.array(data["spaces"][0]["transform"])
        np.testing.assert_array_almost_equal(transform, np.eye(4))


# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------

class TestFromJson:
    def test_roundtrip_single_space_no_parent(self):
        root = Space2D(uid="root")
        pt = Point(coords=np.array([1.0, 2.0]), space=root)
        data = to_json([root], [pt])
        spaces, coords = from_json(data)
        assert len(spaces) == 1
        assert len(coords) == 1

    def test_roundtrip_space_uid_preserved(self):
        root = Space2D(uid="my-root")
        data = to_json([root], [])
        spaces, _ = from_json(data)
        assert spaces[0].uid == "my-root"

    def test_roundtrip_transform_preserved(self):
        t = translate2D(5.0, 3.0)
        root = Space(transform=t, uid="s")
        data = to_json([root], [])
        spaces, _ = from_json(data)
        np.testing.assert_array_almost_equal(spaces[0].transform, t)

    def test_roundtrip_parent_linked(self):
        root = Space2D(uid="root")
        child = Space(transform=translate2D(1, 0), parent=root, uid="child")
        data = to_json([root, child], [])
        spaces, _ = from_json(data)
        uid_map = {s.uid: s for s in spaces}
        assert uid_map["child"].parent is uid_map["root"]

    def test_roundtrip_root_has_no_parent(self):
        root = Space2D(uid="root")
        data = to_json([root], [])
        spaces, _ = from_json(data)
        assert spaces[0].parent is None

    def test_roundtrip_coordinate_coords(self):
        root = Space2D(uid="r")
        pt = Point(coords=np.array([7.0, -3.5]), space=root)
        data = to_json([root], [pt])
        _, coords = from_json(data)
        np.testing.assert_array_almost_equal(coords[0].coords, np.array([7.0, -3.5]))

    def test_roundtrip_coordinate_kind_point(self):
        root = Space2D(uid="r")
        pt = Point(coords=np.array([1.0, 0.0]), space=root)
        data = to_json([root], [pt])
        _, coords = from_json(data)
        assert coords[0].kind == CoordinateKind.POINT

    def test_roundtrip_coordinate_kind_vector(self):
        root = Space2D(uid="r")
        vec = Vector(coords=np.array([0.0, 1.0]), space=root)
        data = to_json([root], [vec])
        _, coords = from_json(data)
        assert coords[0].kind == CoordinateKind.VECTOR

    def test_roundtrip_coordinate_space_linked(self):
        root = Space2D(uid="r")
        pt = Point(coords=np.array([1.0, 2.0]), space=root)
        data = to_json([root], [pt])
        spaces, coords = from_json(data)
        assert coords[0].space is spaces[0]

    def test_roundtrip_empty(self):
        data = to_json([], [])
        spaces, coords = from_json(data)
        assert spaces == []
        assert coords == []

    def test_roundtrip_multiple_spaces_and_coordinates(self):
        root = Space2D(uid="root")
        child = Space(transform=translate2D(2, 3), parent=root, uid="child")
        pts = [
            Point(coords=np.array([1.0, 0.0]), space=root),
            Vector(coords=np.array([0.0, 1.0]), space=child),
        ]
        data = to_json([root, child], pts)
        spaces, coords = from_json(data)
        assert len(spaces) == 2
        assert len(coords) == 2

    def test_roundtrip_3d(self):
        root = Space3D(uid="r3d")
        pt = Point(coords=np.array([1.0, 2.0, 3.0]), space=root)
        data = to_json([root], [pt])
        spaces, coords = from_json(data)
        np.testing.assert_array_almost_equal(coords[0].coords, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(spaces[0].transform, np.eye(4))
