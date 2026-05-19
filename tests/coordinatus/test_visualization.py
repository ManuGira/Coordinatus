"""Unit tests for visualization functions."""

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

from unittest.mock import Mock, patch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from coordinatus import Space, Space2D, Point, create_space
from coordinatus.visualization import (
    draw_space_axes,
    draw_points,
    _build_digraph,
    _tree_pos,
    _compute_layout,
    _make_color_map,
    _find_reference_space,
    _build_hierarchy_render_data,
    _compute_figure_size,
    _draw_hierarchy_subplot,
    _draw_axes_subplot,
    _HierarchyInteractor,
    draw_space_hierarchy,
)


class TestDrawSpaceAxes:
    """Tests for the draw_space_axes function."""

    def _make_ax(self):
        """Mock axes with a real transData so Affine2D + ax.transData works."""
        ax = Mock()
        ax.transData = mtransforms.IdentityTransform()
        return ax

    def test_draws_origin(self):
        """Test that origin point is drawn."""
        ax = self._make_ax()
        space = Space2D()

        draw_space_axes(ax, space, color='blue', label='Test')

        # Origin is a filled diamond drawn via ax.fill
        ax.fill.assert_called()
        first_fill = ax.fill.call_args_list[0]
        assert first_fill[1]['color'] == 'blue'

    def test_draws_arrows_for_axes(self):
        """Test that x and y axis arrows are drawn."""
        ax = self._make_ax()
        space = Space2D()

        draw_space_axes(ax, space)

        # Two arrow bodies drawn via ax.plot (one for x, one for y)
        assert ax.plot.call_count == 2

    def test_draws_axis_labels(self):
        """Test that axis labels are drawn."""
        ax = self._make_ax()
        space = Space2D()

        draw_space_axes(ax, space, label='MySpace')

        # Three PathPatch labels: space label + X axis + Y axis
        assert ax.add_patch.call_count == 3

    def test_none_space_uses_absolute(self):
        """Test that None space draws the absolute/world space."""
        ax = self._make_ax()

        # Should not raise
        draw_space_axes(ax, None)

        ax.fill.assert_called()

    def test_none_space_with_reference_space(self):
        """Test space=None, reference_space=not None (lines 86-88).

        Draws the world/identity axes as seen from an explicit reference space.
        """
        ax = self._make_ax()
        ref = create_space(parent=None, tx=1.0, ty=2.0, angle_rad=0.0, sx=1.0, sy=1.0)

        draw_space_axes(ax, None, reference_space=ref)

        ax.fill.assert_called()

    def test_respects_color_parameter(self):
        """Test that color is applied to all elements."""
        ax = self._make_ax()

        draw_space_axes(ax, Space2D(), color='red')

        # Origin fill and arrow-head fills all use the specified color
        for fill_call in ax.fill.call_args_list:
            assert fill_call[1]['color'] == 'red'
        # Arrow-body plots use the specified color
        for plot_call in ax.plot.call_args_list:
            assert plot_call[1]['color'] == 'red'

    def test_highlight_draws_grid(self):
        """Test that highlight=True causes a grid to be drawn."""
        ax = self._make_ax()
        root = Space2D()
        child = create_space(root, tx=1.0, ty=0.0, angle_rad=0.0, sx=1.0, sy=1.0)

        draw_space_axes(ax, child, reference_space=root, highlight=True)

        # highlight draws grid lines (10 calls) plus 2 arrow bodies → > 2 total
        assert ax.plot.call_count > 2

    def test_explicit_reference_space(self):
        """Test that providing an explicit reference_space works without error."""
        ax = self._make_ax()
        root = Space2D()
        child = create_space(root, tx=0.5, ty=0.5, angle_rad=0.0, sx=1.0, sy=1.0)

        draw_space_axes(ax, child, reference_space=root)

        ax.fill.assert_called()
        assert ax.plot.call_count == 2


class TestDrawPoints:
    """Tests for the draw_points function."""

    def test_draws_single_point(self):
        """Test drawing a single point."""
        ax = Mock()
        point = Point(np.array([1, 2]), space=Space2D())
        
        draw_points(ax, [point], color='red')
        
        ax.plot.assert_called()

    def test_empty_points_does_nothing(self):
        """Test that empty point list doesn't crash."""
        ax = Mock()
        
        draw_points(ax, [])
        
        ax.plot.assert_not_called()

    def test_connects_multiple_points(self):
        """Test that multiple points are connected with lines."""
        ax = Mock()
        space = Space2D()
        points = [
            Point(np.array([0, 0]), space=space),
            Point(np.array([1, 1]), space=space),
        ]
        
        draw_points(ax, points, connect=True)
        
        # Should have line plot and point plot
        assert ax.plot.call_count == 2

    def test_no_connect_option(self):
        """Test that connect=False skips line drawing."""
        ax = Mock()
        space = Space2D()
        points = [
            Point(np.array([0, 0]), space=space),
            Point(np.array([1, 1]), space=space),
        ]
        
        draw_points(ax, points, connect=False)
        
        # Should only have point plot, no line
        assert ax.plot.call_count == 1

    def test_shows_labels(self):
        """Test that point labels are shown."""
        ax = Mock()
        points = [Point(np.array([0, 0]), space=Space2D())]
        
        draw_points(ax, points, label='P', show_labels=True)
        
        ax.text.assert_called()
        assert 'P 1' in ax.text.call_args[0][2]

    def test_hides_labels(self):
        """Test that show_labels=False hides labels."""
        ax = Mock()
        points = [Point(np.array([0, 0]), space=Space2D())]
        
        draw_points(ax, points, show_labels=False)
        
        ax.text.assert_not_called()

    def test_with_explicit_reference_space(self):
        """Test draw_points when reference_space is explicitly provided (not None)."""
        ax = Mock()
        root = Space2D()
        child = create_space(root, tx=2.0, ty=3.0, angle_rad=0.0, sx=1.0, sy=1.0)
        point = Point(np.array([0.0, 0.0]), space=child)

        draw_points(ax, [point], reference_space=root, connect=False, show_labels=False)

        ax.plot.assert_called()
        plot_call = ax.plot.call_args
        xs, ys = plot_call[0][0], plot_call[0][1]
        np.testing.assert_array_almost_equal(xs, [2.0])
        np.testing.assert_array_almost_equal(ys, [3.0])

    def test_respects_reference_space(self):
        """Test that points are transformed to reference space."""
        ax = Mock()
        # Point at (0,0) in a space translated by (5, 3)
        space = create_space(parent=None, tx=5, ty=3)
        point = Point(np.array([0, 0]), space=space)
        
        # View from absolute space
        draw_points(ax, [point], reference_space=None, connect=False, show_labels=False)
        
        # Point should appear at (5, 3) in absolute coords
        plot_call = ax.plot.call_args
        xs, ys = plot_call[0][0], plot_call[0][1]
        np.testing.assert_array_almost_equal(xs, [5])
        np.testing.assert_array_almost_equal(ys, [3])


# ---------------------------------------------------------------------------
# Hierarchy helper functions
# ---------------------------------------------------------------------------

class TestBuildDigraph:
    """Tests for the _build_digraph helper."""

    def test_single_root_node(self):
        root = Space2D()
        G = _build_digraph([root])
        assert id(root) in G.nodes
        assert G.number_of_edges() == 0

    def test_parent_child_edge(self):
        root = Space2D()
        child = Space(transform=np.eye(3), parent=root)
        G = _build_digraph([root, child])
        assert G.has_edge(id(root), id(child))

    def test_two_independent_roots(self):
        r1, r2 = Space2D(), Space2D()
        G = _build_digraph([r1, r2])
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 0

    def test_deep_chain(self):
        root = Space2D()
        mid = Space(transform=np.eye(3), parent=root)
        leaf = Space(transform=np.eye(3), parent=mid)
        G = _build_digraph([root, mid, leaf])
        assert G.has_edge(id(root), id(mid))
        assert G.has_edge(id(mid), id(leaf))


class TestTreePos:
    """Tests for the _tree_pos helper."""

    def test_single_node(self):
        root = Space2D()
        G = _build_digraph([root])
        pos = _tree_pos(G, id(root))
        assert id(root) in pos

    def test_parent_above_child(self):
        root = Space2D()
        child = Space(transform=np.eye(3), parent=root)
        G = _build_digraph([root, child])
        pos = _tree_pos(G, id(root))
        # root should have a higher y-value than child
        assert pos[id(root)][1] > pos[id(child)][1]

    def test_siblings_at_same_depth(self):
        root = Space2D()
        c1 = Space(transform=np.eye(3), parent=root)
        c2 = Space(transform=np.eye(3), parent=root)
        G = _build_digraph([root, c1, c2])
        pos = _tree_pos(G, id(root))
        # Both children are at the same depth
        assert pos[id(c1)][1] == pos[id(c2)][1]


class TestComputeLayout:
    """Tests for the _compute_layout helper."""

    def test_returns_pos_for_all_nodes(self):
        root = Space2D()
        child = Space(transform=np.eye(3), parent=root)
        G = _build_digraph([root, child])
        pos = _compute_layout(G)
        assert set(pos.keys()) == {id(root), id(child)}

    def test_empty_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        pos = _compute_layout(G)
        assert pos == {}


class TestMakeColorMap:
    """Tests for the _make_color_map helper."""

    def test_length_matches_spaces(self):
        spaces = [Space2D(), Space2D(), Space2D()]
        cmap = _make_color_map(spaces)
        assert len(cmap) == 3

    def test_keys_are_ids(self):
        s1, s2 = Space2D(), Space2D()
        cmap = _make_color_map([s1, s2])
        assert id(s1) in cmap
        assert id(s2) in cmap

    def test_values_are_hex_strings(self):
        s = Space2D()
        cmap = _make_color_map([s])
        color = cmap[id(s)]
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_more_than_ten_spaces(self):
        spaces = [Space2D() for _ in range(12)]
        cmap = _make_color_map(spaces)
        assert len(cmap) == 12


class TestFindReferenceSpace:
    """Tests for the _find_reference_space helper."""

    def test_single_root_returned(self):
        root = Space2D()
        G = _build_digraph([root])
        ref = _find_reference_space([root], G)
        assert ref is root

    def test_root_with_children(self):
        root = Space2D()
        child = Space(transform=np.eye(3), parent=root)
        spaces = [root, child]
        G = _build_digraph(spaces)
        ref = _find_reference_space(spaces, G)
        assert ref is root

    def test_multiple_roots_returns_none(self):
        r1, r2 = Space2D(), Space2D()
        spaces = [r1, r2]
        G = _build_digraph(spaces)
        ref = _find_reference_space(spaces, G)
        assert ref is None


class TestComputeFigureSize:
    """Tests for the _compute_figure_size helper."""

    def test_empty_pos_returns_defaults(self):
        w, h = _compute_figure_size({})
        assert w == 4.0
        assert h == 3.0

    def test_single_node_returns_valid_size(self):
        w, h = _compute_figure_size({1: (0.0, 0.0)})
        assert w >= 4.0
        assert h >= 3.0

    def test_wide_layout_increases_width(self):
        # 10 nodes spread wide
        pos = {i: (float(i * 3), 0.0) for i in range(10)}
        w, h = _compute_figure_size(pos)
        assert w > 4.0

    def test_deep_layout_increases_height(self):
        # 5 nodes stacked deep
        pos = {i: (0.0, float(-i)) for i in range(5)}
        w, h = _compute_figure_size(pos)
        assert h > 3.0


class TestBuildHierarchyRenderData:
    """Tests for the _build_hierarchy_render_data helper."""

    def test_single_root(self):
        root = Space2D()
        labels = {id(root): "Root"}
        data = _build_hierarchy_render_data([root], labels)
        assert data.reference_space is root
        assert len(data.spaces) == 1
        assert id(root) in data.colors

    def test_parent_child(self):
        root = Space2D()
        child = Space(transform=np.eye(3), parent=root)
        labels = {id(root): "Root", id(child): "Child"}
        data = _build_hierarchy_render_data([root, child], labels)
        assert data.reference_space is root
        assert data.node_labels[id(root)] == "Root"
        assert data.node_labels[id(child)] == "Child"

    def test_multiple_roots_no_reference(self):
        r1, r2 = Space2D(), Space2D()
        labels = {id(r1): "A", id(r2): "B"}
        data = _build_hierarchy_render_data([r1, r2], labels)
        assert data.reference_space is None


# ---------------------------------------------------------------------------
# Subplot drawing functions
# ---------------------------------------------------------------------------

def _make_data(with_child: bool = True):
    """Shared fixture: build render data for a root (+ optional child) space."""
    root = Space2D()
    spaces = [root]
    if with_child:
        child = create_space(root, tx=1.0, ty=0.0, angle_rad=0.0, sx=1.0, sy=1.0)
        spaces.append(child)
    labels = {id(s): f"S{i}" for i, s in enumerate(spaces)}
    data = _build_hierarchy_render_data(spaces, labels)
    return data, spaces


class TestDrawHierarchySubplot:
    """Tests for _draw_hierarchy_subplot using real Agg axes."""

    def test_draws_without_error(self):
        data, _ = _make_data()
        fig, ax = plt.subplots()
        try:
            _draw_hierarchy_subplot(ax, data)
        finally:
            plt.close(fig)

    def test_title_set(self):
        data, _ = _make_data()
        fig, ax = plt.subplots()
        try:
            _draw_hierarchy_subplot(ax, data)
            assert "Hierarchy" in ax.get_title()
        finally:
            plt.close(fig)


class TestDrawAxesSubplot:
    """Tests for _draw_axes_subplot using real Agg axes."""

    def test_root_title_when_no_reference(self):
        r1, r2 = Space2D(), Space2D()
        labels = {id(r1): "A", id(r2): "B"}
        data = _build_hierarchy_render_data([r1, r2], labels)
        # When data has multiple roots, view_space parent is one of the actual spaces.
        # Spaces in different sub-trees are gracefully skipped during rendering.
        view_space = Space(transform=np.eye(3), parent=r1)
        fig, ax = plt.subplots()
        try:
            _draw_axes_subplot(ax, data, view_space)
            assert labels[id(r1)] in ax.get_title()
        finally:
            plt.close(fig)

    def test_named_reference_in_title(self):
        data, spaces = _make_data()
        root = spaces[0]
        view_space = Space(transform=np.eye(3), parent=data.reference_space)
        fig, ax = plt.subplots()
        try:
            _draw_axes_subplot(ax, data, view_space)
            assert data.labels[id(root)] in ax.get_title()
        finally:
            plt.close(fig)

    def test_fixed_xlim_ylim(self):
        # Limits are always fixed at [-2.5, 2.5]; pan/zoom encoded in view_space.
        data, _ = _make_data()
        view_space = Space(transform=np.eye(3), parent=data.reference_space)
        fig, ax = plt.subplots()
        try:
            _draw_axes_subplot(ax, data, view_space)
            np.testing.assert_allclose(ax.get_xlim(), (-2.5, 2.5))
            np.testing.assert_allclose(ax.get_ylim(), (-2.5, 2.5))
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# _HierarchyInteractor
# ---------------------------------------------------------------------------

class TestHierarchyInteractor:
    """Tests for the _HierarchyInteractor class using real Agg figures."""

    def _make_interactor(self, mock_renders: bool = True):
        """Create an interactor for testing.

        By default (``mock_renders=True``) the initial ``_redraw()`` is
        suppressed and ``_redraw`` is replaced with a
        :class:`~unittest.mock.Mock` object on the instance so that tests
        which only exercise state logic (pan, selection, hit-testing, …) do
        not pay for a full matplotlib render.

        Pass ``mock_renders=False`` for tests that need to invoke the real
        rendering methods.
        """
        root = Space2D()
        child = create_space(root, tx=1.0, ty=0.0, angle_rad=0.0, sx=1.0, sy=1.0)
        spaces = [root, child]
        labels = {id(s): f"S{i}" for i, s in enumerate(spaces)}
        data = _build_hierarchy_render_data(spaces, labels)
        id_to_space = {id(s): s for s in spaces}
        fig, (ax_graph, ax_axes) = plt.subplots(1, 2)
        # Patch _redraw on the class so __init__ skips the initial render.
        with patch.object(_HierarchyInteractor, "_redraw"):
            interactor = _HierarchyInteractor(
                fig, ax_graph, ax_axes, data, "Test", id_to_space
            )
        if mock_renders:
            # Suppress any further renders triggered by event-handler logic.
            interactor._redraw = Mock()  # type: ignore[method-assign]
        return interactor, fig, root, child

    # ── initial state ───────────────────────────────────────────────────────

    def test_initial_selected_node_is_root(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            assert interactor.selected_node == id(root)
        finally:
            plt.close(fig)

    # ── selection ───────────────────────────────────────────────────────────

    def test_select_node_changes_reference(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            interactor._select_node(id(child))
            assert interactor.data.reference_space is child
            assert interactor.selected_node == id(child)
            assert interactor._view_space.parent is child
            assert np.allclose(interactor._view_space.transform, np.eye(3))
        finally:
            plt.close(fig)

    def test_select_same_node_is_noop(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            original_ref = interactor.data.reference_space
            interactor._select_node(id(root))  # root is already selected
            assert interactor.data.reference_space is original_ref
        finally:
            plt.close(fig)

    # ── _node_at_graph_pos ──────────────────────────────────────────────────

    def test_node_at_graph_pos_empty_pos(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            interactor.data.pos = {}
            assert interactor._node_at_graph_pos(0.0, 0.0) is None
        finally:
            plt.close(fig)

    def test_node_at_graph_pos_finds_nearby_node(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            rx, ry = interactor.data.pos[id(root)]
            found = interactor._node_at_graph_pos(rx, ry)
            assert found == id(root)
        finally:
            plt.close(fig)

    def test_node_at_graph_pos_returns_none_when_far(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            assert interactor._node_at_graph_pos(1000.0, 1000.0) is None
        finally:
            plt.close(fig)

    # ── _node_at_axes_pos ───────────────────────────────────────────────────

    def test_node_at_axes_pos_finds_root_origin(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            # Root origin is at (0, 0) in its own (reference) coords
            found = interactor._node_at_axes_pos(0.0, 0.0)
            assert found == id(root)
        finally:
            plt.close(fig)

    def test_node_at_axes_pos_returns_none_when_far(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            assert interactor._node_at_axes_pos(999.0, 999.0) is None
        finally:
            plt.close(fig)

    # ── pan ─────────────────────────────────────────────────────────────────

    def test_start_pan_sets_state(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.x = 200.0
            event.y = 150.0
            interactor._start_pan(event)
            assert interactor._pan_start_display == (200.0, 150.0)
            assert interactor._pan_M0 is not None
            assert interactor._pan_inv_transform is not None
        finally:
            plt.close(fig)

    def test_start_pan_ignores_none_x(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.x = None
            interactor._start_pan(event)
            assert interactor._pan_start_display is None
        finally:
            plt.close(fig)

    def test_on_motion_pan_moves_limits(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            M_before = interactor._view_space.transform.copy()
            press = Mock()
            press.x = 200.0
            press.y = 150.0
            interactor._start_pan(press)
            motion = Mock()
            motion.x = 250.0  # moved 50 px right → view shifts
            motion.y = 150.0
            interactor._on_motion(motion)
            M_after = interactor._view_space.transform
            assert not np.allclose(M_before, M_after)
        finally:
            plt.close(fig)

    def test_on_release_clears_pan_state(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.x = 200.0
            event.y = 150.0
            interactor._start_pan(event)
            release = Mock()
            release.button = 1
            interactor._on_release(release)
            assert interactor._pan_start_display is None
            assert interactor._pan_M0 is None
            assert interactor._pan_inv_transform is None
        finally:
            plt.close(fig)

    def test_on_release_non_left_button_noop(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.x = 200.0
            event.y = 150.0
            interactor._start_pan(event)
            release = Mock()
            release.button = 3  # right-click
            interactor._on_release(release)
            assert interactor._pan_start_display is not None  # unchanged
        finally:
            plt.close(fig)

    # ── scroll / zoom ────────────────────────────────────────────────────────

    def test_scroll_zoom_in(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            M_before = interactor._view_space.transform.copy()
            event = Mock()
            event.inaxes = interactor.ax_axes
            event.xdata = 0.0
            event.ydata = 0.0
            event.step = 1  # scroll up = zoom in
            interactor._on_scroll(event)
            M_after = interactor._view_space.transform
            # zoom in: scale factor 0.9 applied → diagonal shrinks
            assert M_after[0, 0] < M_before[0, 0]
        finally:
            plt.close(fig)

    def test_scroll_zoom_out(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            M_before = interactor._view_space.transform.copy()
            event = Mock()
            event.inaxes = interactor.ax_axes
            event.xdata = 0.0
            event.ydata = 0.0
            event.step = -1  # scroll down = zoom out
            interactor._on_scroll(event)
            M_after = interactor._view_space.transform
            # zoom out: scale factor 1.1 applied → diagonal grows
            assert M_after[0, 0] > M_before[0, 0]
        finally:
            plt.close(fig)

    def test_scroll_ignored_on_wrong_axes(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.inaxes = interactor.ax_graph  # not the axes panel
            interactor._on_scroll(event)  # should not raise
        finally:
            plt.close(fig)

    def test_scroll_ignored_when_no_xdata(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.inaxes = interactor.ax_axes
            event.xdata = None
            interactor._on_scroll(event)  # should not raise
        finally:
            plt.close(fig)

    def test_redraw_axes_only_does_not_touch_graph(self):
        interactor, fig, root, child = self._make_interactor(mock_renders=False)
        try:
            # Capture graph node count before; _redraw_axes_only must leave the graph unchanged.
            graph_nodes_before = list(interactor.ax_graph.get_children())
            interactor._redraw_axes_only()
            graph_nodes_after = list(interactor.ax_graph.get_children())
            assert len(graph_nodes_before) == len(graph_nodes_after)
        finally:
            plt.close(fig)

    # ── on_press ─────────────────────────────────────────────────────────────

    def test_on_press_non_left_button_ignored(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.button = 3
            event.inaxes = interactor.ax_graph
            event.xdata = 0.0
            interactor._on_press(event)  # should not raise
        finally:
            plt.close(fig)

    def test_on_press_graph_selects_node(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            cx, cy = interactor.data.pos[id(child)]
            event = Mock()
            event.button = 1
            event.inaxes = interactor.ax_graph
            event.xdata = cx
            event.ydata = cy
            interactor._on_press(event)
            assert interactor.data.reference_space is child
        finally:
            plt.close(fig)

    def test_on_press_graph_miss_noop(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.button = 1
            event.inaxes = interactor.ax_graph
            event.xdata = 1000.0
            event.ydata = 1000.0
            interactor._on_press(event)  # no node found → noop
            assert interactor.data.reference_space is root
        finally:
            plt.close(fig)

    def test_on_press_axes_click_origin_selects_node(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.button = 1
            event.inaxes = interactor.ax_axes
            # Click at (1, 0) in reference coords → child's origin
            event.xdata = 1.0
            event.ydata = 0.0
            interactor._on_press(event)
            assert interactor.data.reference_space is child
        finally:
            plt.close(fig)

    def test_on_press_axes_empty_area_starts_pan(self):
        interactor, fig, root, child = self._make_interactor()
        try:
            event = Mock()
            event.button = 1
            event.inaxes = interactor.ax_axes
            event.xdata = 50.0  # far from any origin
            event.ydata = 50.0
            event.x = 100.0
            event.y = 100.0
            interactor._on_press(event)
            assert interactor._pan_start_display is not None
        finally:
            plt.close(fig)

    def test_node_at_axes_pos_no_reference_space(self):
        """Covers the else branch (reference_space is None) in _node_at_axes_pos."""
        interactor, fig, root, child = self._make_interactor()
        try:
            interactor.data.reference_space = None
            # Root's absolute origin is at (0, 0)
            found = interactor._node_at_axes_pos(0.0, 0.0)
            assert found == id(root)
        finally:
            plt.close(fig)

    def test_node_at_axes_pos_exception_swallowed(self):
        """Covers the except (ValueError, IndexError) block in _node_at_axes_pos."""
        interactor, fig, root, child = self._make_interactor()
        try:
            # Add a space with a different root so relative_to() raises ValueError
            orphan = Space2D()  # independent root — no common ancestor with root
            interactor.data.spaces = [root, child, orphan]
            # Should not raise; unrelated space is silently skipped
            result = interactor._node_at_axes_pos(0.0, 0.0)
            assert result == id(root)
        finally:
            plt.close(fig)

# ---------------------------------------------------------------------------
# draw_space_hierarchy public API
# ---------------------------------------------------------------------------

class TestDrawSpaceHierarchy:
    """Tests for the draw_space_hierarchy public function."""

    def test_basic_call_does_not_raise(self):
        root = Space2D()
        child = create_space(root, tx=1.0, ty=0.0, angle_rad=0.0, sx=1.0, sy=1.0)
        with patch("matplotlib.pyplot.show"):
            draw_space_hierarchy([root, child], title="Test hierarchy")

    def test_auto_labels_generated(self):
        root = Space2D()
        with patch("matplotlib.pyplot.show"):
            # labels=None → auto-generated "Space 0", etc.
            draw_space_hierarchy([root])

    def test_custom_labels(self):
        root = Space2D()
        child = create_space(root, tx=0.5, ty=0.5, angle_rad=0.0, sx=1.0, sy=1.0)
        labels = {id(root): "World", id(child): "Local"}
        with patch("matplotlib.pyplot.show"):
            draw_space_hierarchy([root, child], labels=labels, title="Custom")
