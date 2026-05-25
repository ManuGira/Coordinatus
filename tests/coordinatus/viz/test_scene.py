"""Unit tests for coordinatus.viz.scene — construction only, no Qt."""

import numpy as np

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


class TestCurveSpec:
    def test_required_fields(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        s = CurveSpec(points=pts, color="#ff0000", width=1.5)
        assert np.array_equal(s.points, pts)
        assert s.color == "#ff0000"
        assert s.width == 1.5
        assert s.name == ""

    def test_name_field(self):
        pts = np.zeros((3, 2))
        s = CurveSpec(points=pts, color="#000000", width=2.0, name="rpm")
        assert s.name == "rpm"


class TestFilledPolygonSpec:
    def test_required_fields(self):
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        s = FilledPolygonSpec(vertices=verts, color="#00ff00")
        assert np.array_equal(s.vertices, verts)
        assert s.color == "#00ff00"
        assert s.border_color is None

    def test_optional_border_color(self):
        verts = np.zeros((4, 2))
        s = FilledPolygonSpec(vertices=verts, color="#ffffff", border_color="#000000")
        assert s.border_color == "#000000"


class TestScatterSpec:
    def test_construction(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = ScatterSpec(
            positions=pos,
            colors=["#ff0000", "#00ff00"],
            sizes=[5.0, 8.0],
            ids=["a", "b"],
        )
        assert np.array_equal(s.positions, pos)
        assert s.colors == ["#ff0000", "#00ff00"]
        assert s.sizes == [5.0, 8.0]
        assert s.ids == ["a", "b"]


class TestLabelSpec:
    def test_required_fields(self):
        s = LabelSpec(x=1.0, y=2.0, text="hello", color="#ffffff")
        assert s.x == 1.0
        assert s.y == 2.0
        assert s.text == "hello"
        assert s.color == "#ffffff"

    def test_defaults(self):
        s = LabelSpec(x=0.0, y=0.0, text="", color="#000000")
        assert s.rotation == 0.0
        assert s.scale == 1.0
        assert s.anchor == (0.5, 0.5)

    def test_optional_fields(self):
        s = LabelSpec(x=0.0, y=0.0, text="T", color="#aaaaaa",
                      rotation=45.0, scale=2.0, anchor=(0.0, 1.0))
        assert s.rotation == 45.0
        assert s.scale == 2.0
        assert s.anchor == (0.0, 1.0)


class TestArrowSpec:
    def test_required_fields(self):
        s = ArrowSpec(x=1.0, y=2.0, angle=90.0, size=0.1, color="#0000ff")
        assert s.x == 1.0
        assert s.y == 2.0
        assert s.angle == 90.0
        assert s.size == 0.1
        assert s.color == "#0000ff"
        assert s.border_color is None

    def test_optional_border_color(self):
        s = ArrowSpec(x=0.0, y=0.0, angle=0.0, size=1.0, color="#ff0000",
                      border_color="#ffffff")
        assert s.border_color == "#ffffff"


class TestGraphScene:
    def test_empty_construction(self):
        g = GraphScene()
        assert g.curves == []
        assert g.arrows == []
        assert g.scatter == []
        assert g.labels == []

    def test_populated_construction(self):
        c = CurveSpec(points=np.zeros((2, 2)), color="#ff0000", width=1.0)
        a = ArrowSpec(x=0.0, y=0.0, angle=0.0, size=1.0, color="#ff0000")
        sc = ScatterSpec(positions=np.zeros((1, 2)), colors=["#aaaaaa"],
                         sizes=[5.0], ids=["n1"])
        lb = LabelSpec(x=0.0, y=0.0, text="n1", color="#ffffff")
        g = GraphScene(curves=[c], arrows=[a], scatter=[sc], labels=[lb])
        assert len(g.curves) == 1
        assert len(g.arrows) == 1
        assert len(g.scatter) == 1
        assert len(g.labels) == 1

    def test_default_lists_are_independent(self):
        g1 = GraphScene()
        g2 = GraphScene()
        g1.curves.append(CurveSpec(points=np.zeros((2, 2)), color="#ff0000", width=1.0))
        assert g2.curves == []


class TestPlotScene:
    def test_empty_construction(self):
        p = PlotScene()
        assert p.curves == []
        assert p.polygons == []
        assert p.scatter == []
        assert p.labels == []

    def test_populated_construction(self):
        c = CurveSpec(points=np.zeros((2, 2)), color="#0000ff", width=1.5)
        poly = FilledPolygonSpec(vertices=np.zeros((3, 2)), color="#00ff00")
        p = PlotScene(curves=[c], polygons=[poly])
        assert len(p.curves) == 1
        assert len(p.polygons) == 1

    def test_default_lists_are_independent(self):
        p1 = PlotScene()
        p2 = PlotScene()
        p1.scatter.append(ScatterSpec(positions=np.zeros((1, 2)),
                                      colors=["#aaaaaa"], sizes=[4.0], ids=["x"]))
        assert p2.scatter == []


class TestScene:
    def test_default_construction(self):
        s = Scene()
        assert isinstance(s.graph, GraphScene)
        assert isinstance(s.plot, PlotScene)

    def test_explicit_construction(self):
        g = GraphScene()
        p = PlotScene()
        s = Scene(graph=g, plot=p)
        assert s.graph is g
        assert s.plot is p

    def test_default_sub_scenes_are_independent(self):
        s1 = Scene()
        s2 = Scene()
        s1.graph.labels.append(LabelSpec(x=0.0, y=0.0, text="X", color="#ff0000"))
        assert s2.graph.labels == []
