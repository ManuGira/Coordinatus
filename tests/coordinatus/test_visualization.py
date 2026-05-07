"""Unit tests for visualization functions."""

from unittest.mock import Mock
import numpy as np

from coordinatus import Space2D, Point, create_space
from coordinatus.visualization import draw_space_axes, draw_points


class TestDrawSpaceAxes:
    """Tests for the draw_space_axes function."""

    def test_draws_origin(self):
        """Test that origin point is drawn."""
        ax = Mock()
        space = Space2D()
        
        draw_space_axes(ax, space, color='blue', label='Test')
        
        ax.plot.assert_called()
        # First plot call is the origin
        first_call = ax.plot.call_args_list[0]
        assert first_call[1]['color'] == 'blue'
        assert 'origin' in first_call[1]['label']

    def test_draws_arrows_for_axes(self):
        """Test that x and y axis arrows are drawn."""
        ax = Mock()
        space = Space2D()
        
        draw_space_axes(ax, space)
        
        assert ax.arrow.call_count == 2  # x-axis and y-axis

    def test_draws_axis_labels(self):
        """Test that axis labels are drawn."""
        ax = Mock()
        space = Space2D()
        
        draw_space_axes(ax, space, label='MySpace')
        
        assert ax.text.call_count == 2
        text_calls = [c[0][2] for c in ax.text.call_args_list]
        assert 'MySpace X' in text_calls
        assert 'MySpace Y' in text_calls

    def test_none_space_uses_absolute(self):
        """Test that None space draws the absolute/world space."""
        ax = Mock()
        
        draw_space_axes(ax, None)
        
        # Origin should be at (0, 0)
        first_call = ax.plot.call_args_list[0]
        assert first_call[0][0] == 0  # x
        assert first_call[0][1] == 0  # y

    def test_respects_color_parameter(self):
        """Test that color is applied to all elements."""
        ax = Mock()
        
        draw_space_axes(ax, Space2D(), color='red')
        
        # Check origin color
        assert ax.plot.call_args_list[0][1]['color'] == 'red'
        # Check arrow colors
        for arrow_call in ax.arrow.call_args_list:
            assert arrow_call[1]['fc'] == 'red'
            assert arrow_call[1]['ec'] == 'red'


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
