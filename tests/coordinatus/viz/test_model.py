"""Unit tests for coordinatus.viz.model — no Qt, no pyqtgraph."""

from __future__ import annotations

import math

import numpy as np
import pytest

from coordinatus.coordinate import Point
from coordinatus.space import Space, Space2D
from coordinatus.transforms import translate2D
from coordinatus.viz.model import VisualizerModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _space_def(space_id: str, tx: float = 0.0, ty: float = 0.0,
               parent_id: str | None = None) -> dict:
    """Return a single space-definition dict for inclusion in a 'spaces' list."""
    return {
        "id": space_id,
        "parent_id": parent_id,
        "transform": translate2D(tx, ty).tolist(),
    }


def _state(spaces: list | None = None, points: list | None = None) -> dict:
    """Build a full-state message."""
    msg: dict = {}
    if spaces is not None:
        msg["spaces"] = spaces
    if points is not None:
        msg["points"] = points
    return msg


def _pts(channel: str, space_id: str, coords: list) -> dict:
    """Build a points sub-message for inclusion in a 'points' list."""
    return {"channel": channel, "space_id": space_id, "coords": coords}


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #


class TestInit:
    def test_empty_domain_data(self):
        m = VisualizerModel()
        assert m.spaces == {}
        assert m.point_channels == {}

    def test_view_space_parent_is_implicit_root(self):
        m = VisualizerModel()
        assert m.view_space.parent is m._implicit_root

    def test_view_space_is_identity(self):
        m = VisualizerModel()
        assert np.allclose(m.view_space.transform, np.eye(3))

    def test_interaction_defaults(self):
        m = VisualizerModel()
        assert m.hovered_node_id is None
        assert m.selected_node_id is None
        assert m.mouse_x == 0.0
        assert m.mouse_y == 0.0

    def test_implicit_root_is_space2d(self):
        m = VisualizerModel()
        assert isinstance(m._implicit_root, Space2D)


# --------------------------------------------------------------------------- #
# apply_message — space
# --------------------------------------------------------------------------- #


class TestApplyMessageSpace:
    def test_root_space_added(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        assert "world" in m.spaces

    def test_root_space_parent_is_implicit_root(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        assert m.spaces["world"].parent is m._implicit_root

    def test_root_space_transform(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world", tx=3.0, ty=4.0)]))
        assert np.allclose(m.spaces["world"].transform, translate2D(3.0, 4.0))

    def test_child_space_parent_set(self):
        m = VisualizerModel()
        m.apply_message(_state([
            _space_def("world"),
            _space_def("sensor", tx=1.0, ty=2.0, parent_id="world"),
        ]))
        assert m.spaces["sensor"].parent is m.spaces["world"]

    def test_out_of_order_child_before_parent(self):
        m = VisualizerModel()
        # child listed before parent in the same message
        m.apply_message(_state([
            _space_def("sensor", tx=1.0, parent_id="world"),
            _space_def("world"),
        ]))
        assert "world" in m.spaces
        assert "sensor" in m.spaces
        assert m.spaces["sensor"].parent is m.spaces["world"]

    def test_out_of_order_chain(self):
        m = VisualizerModel()
        # grandchild → child → root (all in one message, reversed order)
        m.apply_message(_state([
            _space_def("gc", tx=0.1, parent_id="child"),
            _space_def("child", parent_id="root"),
            _space_def("root"),
        ]))
        assert "root" in m.spaces
        assert "child" in m.spaces
        assert "gc" in m.spaces
        assert m.spaces["gc"].parent is m.spaces["child"]

    def test_new_message_replaces_old_spaces(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world", tx=1.0)]))
        m.apply_message(_state([_space_def("world", tx=5.0)]))
        assert np.allclose(m.spaces["world"].transform, translate2D(5.0, 0.0))
        assert len(m.spaces) == 1

    def test_spaces_absent_from_new_message_are_removed(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world"), _space_def("sensor", parent_id="world")]))
        m.apply_message(_state([_space_def("world")]))
        assert "sensor" not in m.spaces


# --------------------------------------------------------------------------- #
# apply_message — points
# --------------------------------------------------------------------------- #


class TestApplyMessagePoints:
    def test_points_added_to_channel(self):
        m = VisualizerModel()
        m.apply_message(_state(
            spaces=[_space_def("world")],
            points=[_pts("lidar", "world", [[1.0, 2.0], [3.0, 4.0]])],
        ))
        assert "lidar" in m.point_channels
        assert len(m.point_channels["lidar"]) == 2

    def test_points_in_correct_space(self):
        m = VisualizerModel()
        m.apply_message(_state(
            spaces=[_space_def("world")],
            points=[_pts("ch", "world", [[7.0, 8.0]])],
        ))
        pt = m.point_channels["ch"][0]
        assert pt.space is m.spaces["world"]
        assert np.allclose(pt.coords, [7.0, 8.0])

    def test_unknown_space_drops_silently(self):
        m = VisualizerModel()
        m.apply_message(_state(points=[_pts("ch", "missing", [[1.0, 2.0]])]))
        assert "ch" not in m.point_channels

    def test_new_message_replaces_old_points(self):
        m = VisualizerModel()
        m.apply_message(_state(
            spaces=[_space_def("world")],
            points=[_pts("ch", "world", [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])],
        ))
        m.apply_message(_state(
            spaces=[_space_def("world")],
            points=[_pts("ch", "world", [[9.0, 0.0]])],
        ))
        assert len(m.point_channels["ch"]) == 1

    def test_two_channels_in_one_message(self):
        m = VisualizerModel()
        m.apply_message(_state(
            spaces=[_space_def("world")],
            points=[
                _pts("a", "world", [[1.0, 0.0]]),
                _pts("b", "world", [[2.0, 0.0], [3.0, 0.0]]),
            ],
        ))
        assert len(m.point_channels["a"]) == 1
        assert len(m.point_channels["b"]) == 2


# --------------------------------------------------------------------------- #
# apply_message — set_display_space
# --------------------------------------------------------------------------- #


class TestApplyMessageSetDisplaySpace:
    def test_select_space_called(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.apply_message({"type": "set_display_space", "space_id": "world"})
        assert m.selected_node_id == "world"
        assert m.view_space.parent is m.spaces["world"]


# --------------------------------------------------------------------------- #
# select_space
# --------------------------------------------------------------------------- #


class TestSelectSpace:
    def test_select_known_space(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.select_space("world")
        assert m.selected_node_id == "world"
        assert m.view_space.parent is m.spaces["world"]
        assert np.allclose(m.view_space.transform, np.eye(3))

    def test_select_none_reverts_to_implicit_root(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.select_space("world")
        m.select_space(None)
        assert m.selected_node_id is None
        assert m.view_space.parent is m._implicit_root

    def test_select_unknown_id_reverts_to_implicit_root(self):
        m = VisualizerModel()
        m.select_space("nonexistent")
        assert m.view_space.parent is m._implicit_root

    def test_view_space_resets_to_identity_on_select(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.pan(5.0, 3.0)  # dirty the view_space transform
        m.select_space("world")
        assert np.allclose(m.view_space.transform, np.eye(3))

    def test_view_space_reanchored_after_spaces_rebuilt(self):
        """When a new state message arrives, view_space re-anchors to the new Space object."""
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.select_space("world")
        m.pan(2.0, 1.0)
        saved_transform = m.view_space.transform.copy()

        # New state message — rebuilds all Space objects
        m.apply_message(_state([_space_def("world", tx=1.0)]))

        assert m.view_space.parent is m.spaces["world"]          # re-anchored
        assert np.allclose(m.view_space.transform, saved_transform)  # pan preserved

    def test_selected_space_removed_from_new_state(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world"), _space_def("sensor", parent_id="world")]))
        m.select_space("sensor")
        # New state without "sensor"
        m.apply_message(_state([_space_def("world")]))
        assert m.selected_node_id is None
        assert m.view_space.parent is m._implicit_root


# --------------------------------------------------------------------------- #
# set_interaction
# --------------------------------------------------------------------------- #


class TestSetInteraction:
    def test_hovered_node_id(self):
        m = VisualizerModel()
        m.set_interaction(hovered_node_id="abc")
        assert m.hovered_node_id == "abc"

    def test_clear_hover(self):
        m = VisualizerModel()
        m.set_interaction(hovered_node_id="abc")
        m.set_interaction(hovered_node_id=None)
        assert m.hovered_node_id is None

    def test_mouse_xy_stored(self):
        m = VisualizerModel()
        m.set_interaction(mouse_x=0.5, mouse_y=-0.3)
        assert m.mouse_x == pytest.approx(0.5)
        assert m.mouse_y == pytest.approx(-0.3)

    def test_does_not_touch_view_space(self):
        m = VisualizerModel()
        original = m.view_space
        m.set_interaction(mouse_x=1.0, mouse_y=1.0, hovered_node_id="x")
        assert m.view_space is original


# --------------------------------------------------------------------------- #
# pan
# --------------------------------------------------------------------------- #


class TestPan:
    def _view(self, m: VisualizerModel, wx: float, wy: float) -> tuple[float, float]:
        """Place a Point at (wx, wy) in implicit_root and return its view coords."""
        pt = Point([wx, wy], space=m._implicit_root)
        pv = pt.relative_to(m.view_space)
        return float(pv.coords[0]), float(pv.coords[1])

    def test_pan_shifts_view_coords_positively(self):
        m = VisualizerModel()
        # Before pan: world (1,0) → view (1,0)
        vx0, vy0 = self._view(m, 1.0, 0.0)
        assert vx0 == pytest.approx(1.0)
        m.pan(1.0, 0.0)
        # After pan right by 1: world (1,0) → view (2,0)
        vx1, vy1 = self._view(m, 1.0, 0.0)
        assert vx1 == pytest.approx(2.0)
        assert vy1 == pytest.approx(0.0)

    def test_pan_creates_new_space_object(self):
        m = VisualizerModel()
        old = m.view_space
        m.pan(1.0, 0.0)
        assert m.view_space is not old

    def test_pan_preserves_parent(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.select_space("world")
        parent_before = m.view_space.parent
        m.pan(2.0, 3.0)
        assert m.view_space.parent is parent_before

    def test_pan_y(self):
        m = VisualizerModel()
        m.pan(0.0, 2.0)
        _, vy = self._view(m, 0.0, 1.0)
        assert vy == pytest.approx(3.0)

    def test_pan_accumulates(self):
        m = VisualizerModel()
        m.pan(1.0, 0.0)
        m.pan(1.0, 0.0)
        vx, _ = self._view(m, 0.0, 0.0)
        assert vx == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# zoom
# --------------------------------------------------------------------------- #


class TestZoom:
    def _view(self, m: VisualizerModel, wx: float, wy: float) -> tuple[float, float]:
        pt = Point([wx, wy], space=m._implicit_root)
        pv = pt.relative_to(m.view_space)
        return float(pv.coords[0]), float(pv.coords[1])

    def test_zoom_in_from_origin(self):
        m = VisualizerModel()
        m.zoom(2.0, 0.0, 0.0)
        vx, vy = self._view(m, 1.0, 0.0)
        assert vx == pytest.approx(2.0)
        assert vy == pytest.approx(0.0)

    def test_zoom_out_from_origin(self):
        m = VisualizerModel()
        m.zoom(0.5, 0.0, 0.0)
        vx, _ = self._view(m, 1.0, 0.0)
        assert vx == pytest.approx(0.5)

    def test_zoom_center_stays_fixed(self):
        m = VisualizerModel()
        cx, cy = 3.0, 2.0
        m.zoom(3.0, cx, cy)
        vx, vy = self._view(m, cx, cy)
        assert vx == pytest.approx(cx)
        assert vy == pytest.approx(cy)

    def test_zoom_creates_new_space_object(self):
        m = VisualizerModel()
        old = m.view_space
        m.zoom(2.0, 0.0, 0.0)
        assert m.view_space is not old

    def test_zoom_preserves_parent(self):
        m = VisualizerModel()
        m.zoom(2.0, 0.0, 0.0)
        assert m.view_space.parent is m._implicit_root


# --------------------------------------------------------------------------- #
# to_scene — empty model
# --------------------------------------------------------------------------- #


class TestToSceneEmpty:
    def test_returns_scene(self):
        from coordinatus.viz.scene import Scene
        m = VisualizerModel()
        scene = m.to_scene()
        assert isinstance(scene, Scene)

    def test_empty_graph(self):
        m = VisualizerModel()
        scene = m.to_scene()
        assert scene.graph.scatter == []
        assert scene.graph.curves == []
        assert scene.graph.arrows == []
        assert scene.graph.labels == []

    def test_empty_plot(self):
        m = VisualizerModel()
        scene = m.to_scene()
        assert scene.plot.scatter == []
        assert scene.plot.curves == []


# --------------------------------------------------------------------------- #
# to_scene — graph panel (space nodes + edges)
# --------------------------------------------------------------------------- #


class TestToSceneGraph:
    def test_space_node_appears_in_scatter(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        scene = m.to_scene()
        assert len(scene.graph.scatter) == 1
        sc = scene.graph.scatter[0]
        assert "world" in sc.ids

    def test_space_origin_position(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world", tx=3.0, ty=4.0)]))
        scene = m.to_scene()
        sc = scene.graph.scatter[0]
        idx = sc.ids.index("world")
        pos = sc.positions[idx]
        assert np.allclose(pos, [3.0, 4.0])

    def test_label_created_per_space(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        scene = m.to_scene()
        labels = scene.graph.labels
        assert len(labels) == 1
        assert labels[0].text == "world"

    def test_hierarchy_edge_curve(self):
        m = VisualizerModel()
        m.apply_message(_state([
            _space_def("world"),
            _space_def("sensor", tx=2.0, parent_id="world"),
        ]))
        scene = m.to_scene()
        assert len(scene.graph.curves) == 1
        curve = scene.graph.curves[0]
        assert np.allclose(curve.points[0], [0.0, 0.0])
        assert np.allclose(curve.points[1], [2.0, 0.0])

    def test_hierarchy_arrow_at_70_percent(self):
        m = VisualizerModel()
        m.apply_message(_state([
            _space_def("world"),
            _space_def("sensor", tx=10.0, parent_id="world"),
        ]))
        scene = m.to_scene()
        assert len(scene.graph.arrows) == 1
        arrow = scene.graph.arrows[0]
        assert arrow.x == pytest.approx(7.0)
        assert arrow.y == pytest.approx(0.0)

    def test_no_edge_for_root_space(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        scene = m.to_scene()
        assert scene.graph.curves == []
        assert scene.graph.arrows == []

    def test_hovered_node_color(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.set_interaction(hovered_node_id="world")
        scene = m.to_scene()
        sc = scene.graph.scatter[0]
        idx = sc.ids.index("world")
        assert sc.colors[idx] == "#4488ff"

    def test_selected_node_color(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        m.select_space("world")
        scene = m.to_scene()
        sc = scene.graph.scatter[0]
        idx = sc.ids.index("world")
        assert sc.colors[idx] == "#aa44ff"

    def test_default_node_color(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        scene = m.to_scene()
        sc = scene.graph.scatter[0]
        assert sc.colors[0] == "#808080"

    def test_multiple_spaces_in_single_scatter_spec(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("a"), _space_def("b", tx=1.0)]))
        scene = m.to_scene()
        assert len(scene.graph.scatter) == 1
        assert set(scene.graph.scatter[0].ids) == {"a", "b"}


# --------------------------------------------------------------------------- #
# to_scene — label rotation
# --------------------------------------------------------------------------- #


class TestLabelRotation:
    def test_label_rotation_zero_for_identity_space(self):
        m = VisualizerModel()
        m.apply_message(_state([_space_def("world")]))
        scene = m.to_scene()
        label = scene.graph.labels[0]
        assert label.rotation == pytest.approx(0.0)

    def test_label_rotation_for_rotated_space(self):
        from coordinatus.transforms import rotate2D
        m = VisualizerModel()
        angle = math.pi / 2  # 90 degrees CCW
        m.apply_message({
            "spaces": [{"id": "rotated", "parent_id": None, "transform": rotate2D(angle).tolist()}]
        })
        scene = m.to_scene()
        label = scene.graph.labels[0]
        assert label.rotation == pytest.approx(90.0, abs=1e-6)
