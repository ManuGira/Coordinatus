"""Unit tests for viz/presenter.py — no Qt, no pyqtgraph.

Presenter accepts an injectable ``_timer`` parameter so the suite runs
headless without a QApplication or display.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, call

from coordinatus.viz.presenter import Presenter

# ── Fake timer ─────────────────────────────────────────────────────────────────


class _FakeQTimer:
    """Minimal QTimer stub that records the calls made by Presenter."""

    def __init__(self) -> None:
        self.timeout = MagicMock()
        self._started = False
        self._interval: int | None = None

    def setInterval(self, ms: int) -> None:
        self._interval = ms

    def start(self) -> None:
        self._started = True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_presenter(inbox=None):
    """Return a (Presenter, model_mock, view_mock, inbox) tuple.

    The Presenter is wired with a *_FakeQTimer* so no Qt application is needed.
    """
    model = MagicMock()
    view = MagicMock()
    if inbox is None:
        inbox = queue.Queue()
    timer = _FakeQTimer()
    p = Presenter(model=model, view=view, inbox=inbox, _timer=timer)
    return p, model, view, inbox


# ── Construction ──────────────────────────────────────────────────────────────


class TestPresenterConstruction:
    def test_stores_model_view_inbox(self):
        p, model, view, inbox = _make_presenter()
        assert p.model is model
        assert p.view is view
        assert p.inbox is inbox

    def test_timer_not_started_on_construction(self):
        p, *_ = _make_presenter()
        assert not p._timer._started


# ── start() ───────────────────────────────────────────────────────────────────


class TestPresenterStart:
    def test_registers_self_as_event_handler(self):
        p, model, view, _ = _make_presenter()
        p.start()
        view.set_event_handler.assert_called_once_with(p)

    def test_starts_timer(self):
        p, *_ = _make_presenter()
        p.start()
        assert p._timer._started

    def test_timer_interval_not_overridden_for_injected_timer(self):
        """Injected timers are owned by the caller; Presenter calls start() only."""
        p, *_ = _make_presenter()
        p.start()
        # setInterval was NOT called on an injected timer (interval stays None)
        assert p._timer._interval is None

    def test_timer_timeout_not_rewired_for_injected_timer(self):
        """Presenter wires timeout only when it creates its own QTimer."""
        p, *_ = _make_presenter()
        p.start()
        p._timer.timeout.connect.assert_not_called()


# ── _on_tick() ────────────────────────────────────────────────────────────────


class TestOnTick:
    def test_empty_inbox_still_calls_to_scene_and_display(self):
        p, model, view, _ = _make_presenter()
        p._on_tick()
        model.to_scene.assert_called_once()
        view.display.assert_called_once_with(model.to_scene.return_value)

    def test_drains_messages_from_inbox(self):
        inbox = queue.Queue()
        msgs = [{"spaces": []}, {"points": []}, {"type": "set_display_space", "space_id": "w"}]
        for m in msgs:
            inbox.put(m)
        p, model, view, _ = _make_presenter(inbox=inbox)
        p._on_tick()
        assert model.apply_message.call_count == 3
        model.apply_message.assert_has_calls([call(m) for m in msgs])

    def test_drains_at_most_200_messages_per_tick(self):
        inbox = queue.Queue()
        for i in range(300):
            inbox.put({"n": i})
        p, model, view, _ = _make_presenter(inbox=inbox)
        p._on_tick()
        assert model.apply_message.call_count == 200
        assert inbox.qsize() == 100  # 100 left unconsumed

    def test_display_called_with_scene_after_apply_messages(self):
        inbox = queue.Queue()
        inbox.put({"spaces": []})
        p, model, view, _ = _make_presenter(inbox=inbox)
        p._on_tick()
        # display was called with the scene produced *after* apply_message ran
        view.display.assert_called_once_with(model.to_scene.return_value)


# ── ViewEventHandler methods ──────────────────────────────────────────────────


class TestViewEventHandlerMethods:
    def test_on_node_hovered_calls_set_interaction(self):
        p, model, *_ = _make_presenter()
        p.on_node_hovered("sensor")
        model.set_interaction.assert_called_once_with(hovered_node_id="sensor")

    def test_on_node_hovered_with_none(self):
        p, model, *_ = _make_presenter()
        p.on_node_hovered(None)
        model.set_interaction.assert_called_once_with(hovered_node_id=None)

    def test_on_node_clicked_selects_new_node(self):
        p, model, *_ = _make_presenter()
        model.selected_node_id = "world"
        p.on_node_clicked("sensor")
        model.select_space.assert_called_once_with("sensor")

    def test_on_node_clicked_deselects_already_selected_node(self):
        """Clicking the currently-selected node deselects (toggle)."""
        p, model, *_ = _make_presenter()
        model.selected_node_id = "sensor"
        p.on_node_clicked("sensor")
        model.select_space.assert_called_once_with(None)

    def test_on_plot_pan_delegates_to_model(self):
        p, model, *_ = _make_presenter()
        p.on_plot_pan(0.5, -0.3)
        model.pan.assert_called_once_with(0.5, -0.3)

    def test_on_plot_zoom_delegates_to_model(self):
        p, model, *_ = _make_presenter()
        p.on_plot_zoom(1.15, 0.0, 0.0)
        model.zoom.assert_called_once_with(1.15, 0.0, 0.0)

    def test_on_mouse_moved_delegates_to_model(self):
        p, model, *_ = _make_presenter()
        p.on_mouse_moved(0.2, -0.7)
        model.set_interaction.assert_called_once_with(mouse_x=0.2, mouse_y=-0.7)
