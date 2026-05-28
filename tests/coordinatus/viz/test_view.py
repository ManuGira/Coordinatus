"""Tests for viz/view.py — non-Qt portions.

All tests here avoid creating QApplication / QWidget instances so that they
run safely in headless CI environments.  The Qt widget classes themselves are
exercised through the running application in manual / integration testing.
"""

from __future__ import annotations

import numpy as np

from coordinatus.viz.scene import (
    CurveSpec,
    GraphScene,
    PlotScene,
    ScatterSpec,
    Scene,
)
from coordinatus.viz.view import ViewEventHandler


# ── ViewEventHandler protocol ──────────────────────────────────────────────────


class _FullHandler:
    """Minimal class that satisfies the ViewEventHandler protocol."""

    def on_node_hovered(self, node_id: str | None) -> None:
        pass

    def on_node_clicked(self, node_id: str) -> None:
        pass

    def on_plot_pan(self, dx: float, dy: float) -> None:
        pass

    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None:
        pass

    def on_mouse_moved(self, norm_x: float, norm_y: float) -> None:
        pass


class _MissingOneMethod:
    """A class that is missing on_mouse_moved."""

    def on_node_hovered(self, node_id: str | None) -> None:
        pass

    def on_node_clicked(self, node_id: str) -> None:
        pass

    def on_plot_pan(self, dx: float, dy: float) -> None:
        pass

    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None:
        pass


class TestViewEventHandlerProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        assert isinstance(_FullHandler(), ViewEventHandler)

    def test_incomplete_class_fails_isinstance(self) -> None:
        assert not isinstance(_MissingOneMethod(), ViewEventHandler)

    def test_empty_class_fails_isinstance(self) -> None:
        class Empty:
            pass

        assert not isinstance(Empty(), ViewEventHandler)


# ── Topology diff logic ────────────────────────────────────────────────────────


def _make_scatter(ids: list[str]) -> ScatterSpec:
    n = len(ids)
    return ScatterSpec(
        points=np.zeros((n, 2)),
        colors=["#ffffff"] * n,
        sizes=[10.0] * n,
        ids=ids,
    )


def _make_curve() -> CurveSpec:
    return CurveSpec(points=np.array([[0.0, 0.0], [1.0, 1.0]]), color="#ff0000", width=1.0)


def _prev_ids(scene: Scene) -> frozenset[str]:
    return frozenset(nid for s in scene.graph.scatter for nid in s.ids)


def _curr_ids(scene: Scene) -> frozenset[str]:
    return _prev_ids(scene)


def _needs_rebuild(prev: Scene, curr: Scene) -> bool:
    p_ids = frozenset(nid for s in prev.graph.scatter for nid in s.ids)
    c_ids = frozenset(nid for s in curr.graph.scatter for nid in s.ids)
    return c_ids != p_ids or len(curr.graph.curves) != len(prev.graph.curves)


class TestTopologyDiff:
    def test_same_nodes_same_edges_no_rebuild(self) -> None:
        prev = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a", "b"])], curves=[_make_curve()]),
            plot=PlotScene(),
        )
        curr = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a", "b"])], curves=[_make_curve()]),
            plot=PlotScene(),
        )
        assert not _needs_rebuild(prev, curr)

    def test_added_node_triggers_rebuild(self) -> None:
        prev = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a"])]),
            plot=PlotScene(),
        )
        curr = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a", "b"])]),
            plot=PlotScene(),
        )
        assert _needs_rebuild(prev, curr)

    def test_removed_node_triggers_rebuild(self) -> None:
        prev = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a", "b"])]),
            plot=PlotScene(),
        )
        curr = Scene(
            graph=GraphScene(scatter=[_make_scatter(["a"])]),
            plot=PlotScene(),
        )
        assert _needs_rebuild(prev, curr)

    def test_different_edge_count_triggers_rebuild(self) -> None:
        prev = Scene(
            graph=GraphScene(
                scatter=[_make_scatter(["a", "b"])],
                curves=[_make_curve()],
            ),
            plot=PlotScene(),
        )
        curr = Scene(
            graph=GraphScene(
                scatter=[_make_scatter(["a", "b"])],
                curves=[],
            ),
            plot=PlotScene(),
        )
        assert _needs_rebuild(prev, curr)

    def test_empty_prev_empty_curr_no_rebuild(self) -> None:
        prev = Scene(graph=GraphScene(), plot=PlotScene())
        curr = Scene(graph=GraphScene(), plot=PlotScene())
        assert not _needs_rebuild(prev, curr)

    def test_first_render_vs_empty_prev_triggers_rebuild(self) -> None:
        """Simulates the first render: prev is empty, curr has nodes."""
        prev = Scene(graph=GraphScene(), plot=PlotScene())
        curr = Scene(
            graph=GraphScene(scatter=[_make_scatter(["world", "sensor"])]),
            plot=PlotScene(),
        )
        assert _needs_rebuild(prev, curr)
