"""Presenter — orchestrator between VisualizerModel and VisualizerView.

Drives the 16 ms render loop, drains the inbox queue, and forwards user
events from the View to the Model.

PySide6 is a soft dependency: it is imported lazily inside ``start()`` so
that tests can construct and exercise a ``Presenter`` without a live Qt
application or display.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from coordinatus.viz.model import VisualizerModel
    from coordinatus.viz.view import VisualizerView

_MAX_MESSAGES_PER_TICK = 200
_TICK_INTERVAL_MS = 16


class Presenter:
    """Orchestrates Model ↔ View: render loop + event forwarding.

    Parameters
    ----------
    model:
        The domain-data / scene-generation object.
    view:
        The Qt main window renderer.
    inbox:
        Thread-safe queue that ``SocketServer`` fills with decoded dicts.
    _timer:
        Optional injectable timer (for unit tests).  When *None*, a real
        ``PySide6.QtCore.QTimer`` is created inside ``start()``.
    """

    def __init__(
        self,
        model: "VisualizerModel",
        view: "VisualizerView",
        inbox: "queue.Queue[dict]",
        *,
        _timer: Any = None,
    ) -> None:
        self.model = model
        self.view = view
        self.inbox = inbox
        self._timer = _timer  # None until start() unless injected

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Register self as the View's event handler and start the render timer."""
        if self._timer is None:
            from PySide6.QtCore import QTimer  # pragma: no cover

            self._timer = QTimer()
            self._timer.setInterval(_TICK_INTERVAL_MS)
            self._timer.timeout.connect(self._on_tick)
        self.view.set_event_handler(self)
        self._timer.start()

    # ── Render loop ───────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        """Drain inbox, apply messages to model, then render."""
        for _ in range(_MAX_MESSAGES_PER_TICK):
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                break
            self.model.apply_message(msg)

        scene = self.model.to_scene()
        self.view.render(scene)

    # ── ViewEventHandler implementation ───────────────────────────────────────

    def on_node_hovered(self, node_id: str | None) -> None:
        self.model.set_interaction(hovered_node_id=node_id)

    def on_node_clicked(self, node_id: str) -> None:
        # Toggle: clicking the already-selected node deselects it.
        new_id = None if node_id == self.model.selected_node_id else node_id
        self.model.select_space(new_id)

    def on_plot_pan(self, dx: float, dy: float) -> None:
        self.model.pan(dx, dy)

    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None:
        self.model.zoom(factor, cx, cy)

    def on_mouse_moved(self, norm_x: float, norm_y: float) -> None:
        self.model.set_interaction(mouse_x=norm_x, mouse_y=norm_y)
