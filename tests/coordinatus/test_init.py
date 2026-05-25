"""Unit tests for the coordinatus package __init__.py."""


class TestPackageImports:
    """Tests for package-level imports."""

    def test_import_main_classes(self):
        """Test that main classes can be imported from package."""
        from coordinatus import CoordinateKind, Space, Point, Vector, Coordinate
        from coordinatus.transforms import translate2D
        from coordinatus import create_space
        
        # Verify they are the expected types
        assert CoordinateKind is not None
        assert Space is not None
        assert Point is not None
        assert Vector is not None
        assert Coordinate is not None
        assert callable(translate2D)
        assert callable(create_space)

    def test_visualization_import_success(self):
        """Test that visualization module is available when matplotlib is installed."""
        import coordinatus
        
        # Since matplotlib is installed in dev dependencies, this should not be None
        assert coordinatus.visualization2 is not None

