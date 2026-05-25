"""VisualizerView — Qt renderer for the MVP visualizer.

Renders Scene snapshots passed from the Presenter via ``render(scene)``.
All domain logic lives in the Model; this module contains only rendering
and event-dispatch code.

Requires pyqtgraph and PySide6 (dev dependencies):
    uv add pyqtgraph PySide6 --dev
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

try:
    import pyqtgraph as pg
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtWidgets import QMainWindow, QSplitter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv add pyqtgraph PySide6 --dev"
    ) from exc

from coordinatus.viz.scene import (
    ArrowSpec,
    CurveSpec,
    FilledPolygonSpec,
    GraphScene,
    LabelSpec,
    PlotScene,
    ScatterSpec,
    Scene,
)

# ── Visual constants ───────────────────────────────────────────────────────────

_BG = "#1e1e2e"
_FG = "#cdd6f4"

# Default ArrowItem dimensions (pixel space; pyqtgraph ArrowItem uses pixels)
_ARROW_HEAD_LEN = 14
_ARROW_TAIL_LEN = 0

HOST = "127.0.0.1"
PORT = 9876


# ── ViewEventHandler protocol ──────────────────────────────────────────────────


@runtime_checkable
class ViewEventHandler(Protocol):
    """Protocol the Presenter implements; fired by the View on user events."""

    def on_node_hovered(self, node_id: str | None) -> None: ...
    def on_node_clicked(self, node_id: str) -> None: ...
    def on_plot_pan(self, dx: float, dy: float) -> None: ...
    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None: ...
    def on_mouse_moved(self, norm_x: float, norm_y: float) -> None: ...


# ── Left panel: directed space-graph ──────────────────────────────────────────


class _DirectedGraphPanel(pg.PlotWidget):
    """Left panel — space-origin nodes and parent→child hierarchy edges.

    Native ViewBox pan/zoom is **enabled**.  A one-shot ``enableAutoRange()``
    is called on the first ``render()`` so all nodes are visible initially;
    after that the ViewBox is self-managing.

    Hover and click events are forwarded to a ``ViewEventHandler``.
    """

    def __init__(self) -> None:
        super().__init__(background=_BG)
        self.setTitle("Space Graph", color=_FG, size="11pt")
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.getViewBox().setMouseEnabled(x=True, y=True)

        self._handler: ViewEventHandler | None = None
        self._scatter_item: pg.ScatterPlotItem | None = None
        self._edge_items: list[pg.PlotDataItem] = []
        self._arrow_items: list[pg.ArrowItem] = []
        self._label_items: list[pg.TextItem] = []
        self._first_render = True

    # ── Public interface ───────────────────────────────────────────────────

    def set_handler(self, handler: ViewEventHandler) -> None:
        self._handler = handler

    def display(self, scene: GraphScene, needs_rebuild: bool) -> None:
        """Render the graph scene.

        Full rebuild when *needs_rebuild* is True; style-only update otherwise.
        One-shot auto-range fires on the very first call.
        """
        if needs_rebuild:
            self._full_rebuild(scene)
        else:
            self._style_update(scene)

        if self._first_render:
            self.getViewBox().enableAutoRange()
            self._first_render = False

    # ── Full rebuild ───────────────────────────────────────────────────────

    def _full_rebuild(self, scene: GraphScene) -> None:
        self._clear_all()
        for spec in scene.curves:
            self._add_curve(spec)
        for spec in scene.arrows:
            self._add_arrow(spec)
        if scene.scatter:
            self._scatter_item = pg.ScatterPlotItem(hoverable=True)
            self._scatter_item.sigClicked.connect(self._on_clicked)
            self._scatter_item.sigHovered.connect(self._on_hovered)
            self.addItem(self._scatter_item)
            self._apply_scatter(scene.scatter)
        for spec in scene.labels:
            self._add_label(spec)

    def _clear_all(self) -> None:
        for item in self._edge_items:
            self.removeItem(item)
        for item in self._arrow_items:
            self.removeItem(item)
        if self._scatter_item is not None:
            self.removeItem(self._scatter_item)
        for item in self._label_items:
            self.removeItem(item)
        self._edge_items.clear()
        self._arrow_items.clear()
        self._scatter_item = None
        self._label_items.clear()

    # ── Style-only update ─────────────────────────────────────────────────

    def _style_update(self, scene: GraphScene) -> None:
        """Update scatter colors/sizes and labels — topology unchanged."""
        if scene.scatter and self._scatter_item is not None:
            self._apply_scatter(scene.scatter)
        # Labels are cheap; re-add them to pick up color changes from hover/select.
        for item in self._label_items:
            self.removeItem(item)
        self._label_items.clear()
        for spec in scene.labels:
            self._add_label(spec)

    # ── Item constructors ─────────────────────────────────────────────────

    def _add_curve(self, spec: CurveSpec) -> None:
        item = self.plot(
            spec.points[:, 0],
            spec.points[:, 1],
            pen=pg.mkPen(spec.color, width=spec.width),
        )
        self._edge_items.append(item)

    def _add_arrow(self, spec: ArrowSpec) -> None:
        # ArrowItem.angle is clockwise from +x (Qt convention);
        # ArrowSpec.angle is CCW from +x — negate to convert.
        qt_angle = -spec.angle
        arrow = pg.ArrowItem(
            pos=(spec.x, spec.y),
            angle=qt_angle,
            headLen=_ARROW_HEAD_LEN,
            tailLen=_ARROW_TAIL_LEN,
            tipAngle=28,
            baseAngle=12,
            brush=pg.mkBrush(spec.color),
            pen=pg.mkPen(spec.border_color) if spec.border_color else pg.mkPen(None),
        )
        self.addItem(arrow)
        self._arrow_items.append(arrow)

    def _apply_scatter(self, scatter_list: list[ScatterSpec]) -> None:
        """Set spots on the shared scatter item from all ScatterSpecs."""
        spots = []
        for spec in scatter_list:
            for i in range(len(spec.ids)):
                pos = spec.positions[i]
                spots.append(
                    {
                        "pos": (float(pos[0]), float(pos[1])),
                        "brush": pg.mkBrush(spec.colors[i]),
                        "pen": pg.mkPen(_FG, width=1.5),
                        "size": spec.sizes[i],
                        "data": spec.ids[i],
                    }
                )
        if self._scatter_item is not None:
            self._scatter_item.setData(spots)

    def _add_label(self, spec: LabelSpec) -> None:
        lbl = pg.TextItem(text=spec.text, color=spec.color, anchor=spec.anchor)
        # TextItem.setAngle takes clockwise degrees; spec.rotation is CCW — negate.
        lbl.setAngle(-spec.rotation)
        lbl.setPos(spec.x, spec.y)
        lbl.setZValue(10)
        self.addItem(lbl)
        self._label_items.append(lbl)

    # ── Interaction callbacks ─────────────────────────────────────────────

    def _on_hovered(self, _scatter, spots, _ev) -> None:
        if self._handler is None:
            return
        node_id: str | None = spots[0].data() if len(spots) > 0 else None
        self._handler.on_node_hovered(node_id)

    def _on_clicked(self, _scatter, spots, _ev) -> None:
        if self._handler is None or len(spots) == 0:
            return
        self._handler.on_node_clicked(str(spots[0].data()))


# ── Right panel: view-space plot ───────────────────────────────────────────────


class _PlotPanel(pg.PlotWidget):
    """Right panel — point channels and space-axis overlays in view-space coords.

    Native ViewBox pan/zoom is **disabled**.
    Left-button drag fires ``on_plot_pan``; scroll wheel fires ``on_plot_zoom``.
    Mouse moves fire ``on_mouse_moved`` (normalised [-1, 1]).

    A one-shot ``autoRange()`` + ``disableAutoRange()`` runs on the first
    ``render()`` call to fit the initial data and then lock the viewport.
    """

    def __init__(self) -> None:
        super().__init__(background=_BG)
        self.setTitle("Plot", color=_FG, size="11pt")
        self.setLabel("bottom", "x", color=_FG)
        self.setLabel("left", "y", color=_FG)
        self.setAspectLocked(True)
        self.setMouseTracking(True)

        self._plot_vb = self.getViewBox()
        self._plot_vb.setMouseEnabled(x=False, y=False)

        self._handler: ViewEventHandler | None = None
        self._drag_scene_pos: QPointF | None = None  # scene-space drag anchor

        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._curve_items: list[pg.PlotDataItem] = []
        self._polygon_items: list[pg.PlotCurveItem] = []
        self._label_items: list[pg.TextItem] = []
        self._first_render = True

    # ── Public interface ───────────────────────────────────────────────────

    def set_handler(self, handler: ViewEventHandler) -> None:
        self._handler = handler

    def display(self, scene: PlotScene) -> None:
        """Render a PlotScene snapshot; replaces all items on every call."""
        self._clear_all()
        for spec in scene.curves:
            self._add_curve(spec)
        for spec in scene.polygons:
            self._add_polygon(spec)
        for spec in scene.scatter:
            self._add_scatter(spec)
        for spec in scene.labels:
            self._add_label(spec)

        if self._first_render:
            # One-shot auto-fit so the initial data is visible; then lock viewport.
            self._plot_vb.autoRange(padding=0.15)
            self._plot_vb.disableAutoRange()
            self._first_render = False

    # ── Item constructors ─────────────────────────────────────────────────

    def _clear_all(self) -> None:
        all_items: list = (
            self._curve_items
            + self._polygon_items
            + self._scatter_items
            + self._label_items
        )
        for item in all_items:
            self.removeItem(item)
        self._curve_items.clear()
        self._polygon_items.clear()
        self._scatter_items.clear()
        self._label_items.clear()

    def _add_curve(self, spec: CurveSpec) -> None:
        if len(spec.points) == 0:
            return
        item = self.plot(
            spec.points[0, :],
            spec.points[1, :],
            pen=pg.mkPen(spec.color, width=spec.width),
            name=spec.name or None,
        )
        self._curve_items.append(item)

    def _add_polygon(self, spec: FilledPolygonSpec) -> None:
        if len(spec.vertices) == 0:
            return
        # Close the polygon by repeating the first vertex.
        x = np.append(spec.vertices[0, :], spec.vertices[0, 0])
        y = np.append(spec.vertices[1, :], spec.vertices[1, 0])
        pen = (
            pg.mkPen(spec.border_color) if spec.border_color else pg.mkPen(None)
        )
        item = pg.PlotCurveItem(x=x, y=y, pen=pen, brush=pg.mkBrush(spec.color))
        self.addItem(item)
        self._polygon_items.append(item)

    def _add_scatter(self, spec: ScatterSpec) -> None:
        if not spec.ids:
            return
        spots = [
            {
                "pos": (float(spec.positions[i, 0]), float(spec.positions[i, 1])),
                "brush": pg.mkBrush(spec.colors[i]),
                "pen": pg.mkPen(None),
                "size": spec.sizes[i],
                "data": spec.ids[i],
            }
            for i in range(len(spec.ids))
        ]
        item = pg.ScatterPlotItem()
        item.setData(spots)
        self.addItem(item)
        self._scatter_items.append(item)

    def _add_label(self, spec: LabelSpec) -> None:
        lbl = pg.TextItem(text=spec.text, color=spec.color, anchor=spec.anchor)
        lbl.setAngle(-spec.rotation)
        lbl.setPos(spec.x, spec.y)
        self.addItem(lbl)
        self._label_items.append(lbl)

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _widget_to_data(self, widget_pos: QPointF) -> QPointF:
        """Convert widget-local coordinates to ViewBox data coordinates."""
        scene_pos = self.mapToScene(widget_pos.toPoint())
        return self._plot_vb.mapSceneToView(scene_pos)

    # ── Mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_scene_pos = QPointF(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = event.position()

        if self._handler is not None:
            # Normalised mouse position in [-1, 1] for the right panel.
            w = max(self.width(), 1)
            h = max(self.height(), 1)
            nx = pos.x() / w * 2.0 - 1.0
            ny = pos.y() / h * 2.0 - 1.0
            self._handler.on_mouse_moved(nx, ny)

            # Incremental pan: compute delta in data space between consecutive moves.
            if self._drag_scene_pos is not None:
                curr = QPointF(pos)
                p_prev = self._widget_to_data(self._drag_scene_pos)
                p_curr = self._widget_to_data(curr)
                dx = float(p_curr.x() - p_prev.x())
                dy = float(p_curr.y() - p_prev.y())
                self._drag_scene_pos = curr
                self._handler.on_plot_pan(dx, dy)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_scene_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._handler is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        # 120 units per standard scroll notch; 1.15× per notch.
        factor = 1.15 ** (delta / 120.0)
        pos_data = self._widget_to_data(QPointF(event.position()))
        self._handler.on_plot_zoom(
            factor, float(pos_data.x()), float(pos_data.y())
        )
        event.accept()


# ── Main window ────────────────────────────────────────────────────────────────


class VisualizerView(QMainWindow):
    """Two-panel Qt main window.

    Left panel  — ``_DirectedGraphPanel``: space-graph with native pan/zoom.
    Right panel — ``_PlotPanel``: view-space data with custom pan/zoom.

    The caller wires the Presenter via ``set_event_handler(presenter)`` before
    starting the render loop.
    """

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        super().__init__()
        pg.setConfigOptions(antialias=True, background=_BG, foreground=_FG)
        self.setWindowTitle(f"Coordinatus Visualizer  |  socket {host}:{port}")
        self.resize(1280, 700)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self._graph_panel = _DirectedGraphPanel()
        self._plot_panel = _PlotPanel()
        splitter.addWidget(self._graph_panel)
        splitter.addWidget(self._plot_panel)
        splitter.setSizes([480, 800])

        self.statusBar().showMessage(
            f"Hover nodes to highlight  |  click to select  |  socket {host}:{port}"
        )

        # Sentinel empty scene for the topology diff on the first render call.
        self._prev_scene = Scene(graph=GraphScene(), plot=PlotScene())

    # ── Public interface ───────────────────────────────────────────────────

    def set_event_handler(self, handler: ViewEventHandler) -> None:
        """Register the Presenter as the event handler for both panels."""
        self._graph_panel.set_handler(handler)
        self._plot_panel.set_handler(handler)

    def display(self, scene: Scene) -> None:
        """Render a Scene snapshot.

        Performs a topology diff to decide between a full graph rebuild
        (node IDs or edge count changed) and a cheap style-only update
        (same topology, different colors/sizes from hover or selection).
        """
        prev_ids = frozenset(
            nid for s in self._prev_scene.graph.scatter for nid in s.ids
        )
        curr_ids = frozenset(
            nid for s in scene.graph.scatter for nid in s.ids
        )
        needs_rebuild = curr_ids != prev_ids or len(scene.graph.curves) != len(
            self._prev_scene.graph.curves
        )

        self._graph_panel.display(scene.graph, needs_rebuild)
        self._plot_panel.display(scene.plot)
        self._prev_scene = scene
