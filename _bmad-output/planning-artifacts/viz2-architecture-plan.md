# Architecture Plan — `visualization2` Refactor

**Date:** 2026-05-21  
**Status:** Draft — pending implementation

---

## 1. Goals

Refactor `visualization2.py` into a layered MVP architecture that:

- Receives Coordinatus `Space` / `Point` objects from a socket, stores them in a domain model, and displays them.
- Tracks UI interaction state (hovered/selected node) inside the Model rather than inside the widgets.
- Node graph panel (left) pan/zoom is handled natively by the pyqtgraph ViewBox — no Model involvement.
- Plot panel (right) pan/zoom is driven by mouse events that update `view_space` in the Model.
- Keeps PyQtGraph isolated behind an abstract View so the domain logic is testable without a running Qt application.
- Stays in `src/coordinatus/` and is runnable as a single entry point: `uv run python src/coordinatus/visualization2.py`.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Background thread                                                   │
│  SocketServer ──► queue.Queue[dict]                                  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  drained each QTimer tick (16 ms)
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  Presenter  (Qt thread, owns QTimer)                              │
│                                                                   │
│  1. Drain queue  ──►  Model.apply_message(msg)                    │
│  2. scene = Model.to_scene()                                      │
│  3. View.render(scene)                                            │
│                                                                   │
│  on_node_hovered / on_node_clicked / on_view_range_changed        │
│  on_mouse_moved  ──►  Model.update_interaction(...)               │
└──────────┬────────────────────────────────────────────────────────┘
           │  render(Scene)            fires ViewEventHandler callbacks
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  View (QMainWindow wrapping pyqtgraph)                           │
│  DirectedGraphPanel  |  LivePlotPanel                            │
│  Reads Scene → sets items.  Never calls Model directly.          │
└──────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**

| Direction | What | How |
|---|---|---|
| Socket → Model | Raw JSON messages | `queue.Queue` drained by Presenter on tick |
| Model → View | `Scene` snapshot | `Presenter.to_scene()` → `View.render(scene)` |
| View → Presenter | User events | `ViewEventHandler` callbacks |
| Presenter → Model | Interaction state | Direct method calls on Model |

---

## 3. Decisions Record

| # | Decision | Rationale |
|---|---|---|
| D1 | **Model + Presenter** (separate objects) | Keeps domain logic testable; Presenter owns the Qt timer and the translation layer |
| D2 | **`Scene` uses abstract dataclasses**, no pyqtgraph imports | View is swappable; domain tests require only numpy |
| D3 | **`queue.Queue` + drain on tick** | Thread-safe without locks; 16 ms latency is acceptable; no Qt dependency in SocketServer |
| D4 | **Graph panel uses native ViewBox pan/zoom; plot panel uses `view_space`** | The node graph panel's (left) pan/zoom is handled entirely by pyqtgraph's native ViewBox — no Model involvement. The Model holds a `view_space` whose parent is the currently selected Space; it is used to project Coordinatus objects onto the **plot** panel (right). Mouse events on the plot panel drive `model.pan()`/`model.zoom()`, which replace `view_space` with a new `Space` carrying the updated transform. |
| D5 | **Space hierarchy IS the visual graph** | Nodes = Space origins; edges = parent→child links derived from `Space.parent`. Hierarchy edges get `ArrowSpec` (directed parent→child). |

---

## 4. Proposed File Structure

```
src/coordinatus/
    visualization2.py          # thin entry point: wires all components, calls main()
    viz/
        __init__.py
        scene.py               # Scene + Spec dataclasses (no Qt, no pyqtgraph)
        model.py               # VisualizerModel (domain data + interaction state)
        presenter.py           # Presenter (QObject, owns QTimer)
        view.py                # VisualizerView (QMainWindow, pyqtgraph panels)
        server.py              # SocketServer (moved from visualization2.py)
```

`visualization2.py` becomes:

```python
from coordinatus.viz.server import SocketServer
from coordinatus.viz.model import VisualizerModel
from coordinatus.viz.presenter import Presenter
from coordinatus.viz.view import VisualizerView
# … sys.path fix, main()
```

---

## 5. Component Contracts

### 5.1 `scene.py` — display snapshot (no external deps beyond numpy)

**`GraphScene`** coordinates are in absolute (root) world units — the pyqtgraph ViewBox on the left panel handles its own viewport via native pan/zoom.
**`PlotScene`** coordinates are already projected into `view_space` units — the right panel renders them directly with no further transformation.

```python
@dataclass
class CurveSpec:
    points: np.ndarray    # (N, 2) in view-space coords
    color: str            # hex
    width: float
    name: str = ""        # shown in plot legend; empty = omit from legend

@dataclass
class FilledPolygonSpec:
    vertices: np.ndarray  # (N, 2) in view-space coords; closed automatically
    color: str            # fill hex
    border_color: str | None = None

@dataclass
class ScatterSpec:
    positions: np.ndarray  # (N, 2)
    colors: list[str]      # per-point hex
    sizes: list[float]     # per-point px
    ids: list[str]         # per-point identifier passed back in click callbacks

@dataclass
class LabelSpec:
    x: float
    y: float
    text: str
    color: str
    rotation: float = 0.0              # degrees CCW
    scale: float = 1.0                 # multiplier on default font size
    anchor: tuple[float, float] = (0.5, 0.5)

@dataclass
class ArrowSpec:
    x: float              # tip position in view-space coords
    y: float
    angle: float          # direction the arrow points, degrees CCW from +x axis
    size: float           # arrowhead length in view-space units
    color: str
    border_color: str | None = None

@dataclass
class GraphScene:
    """Flat rendering primitives for the left (spatial) panel.
    The Model bakes in all semantic decisions — color, size, label text.
    The View has zero domain knowledge; it renders whatever is here."""
    curves:   list[CurveSpec]          # space-hierarchy edges (line segments), trajectories
    arrows:   list[ArrowSpec]          # directed arrowheads on hierarchy edges (parent→child)
    scatter:  list[ScatterSpec]        # space-origin nodes, point channels
    labels:   list[LabelSpec]          # space names

@dataclass
class PlotScene:
    curves:  list[CurveSpec]           # Litst of coordinates joined by lines (right panel)
    polygons: list[FilledPolygonSpec]  # filled shapes (non-directional)
    scatter: list[ScatterSpec]         # point annotations on the right panel
    labels:  list[LabelSpec]           # text annotations on the right panel

@dataclass
class Scene:
    graph: GraphScene
    plot:  PlotScene
    # All coordinates in both panels are already in their panel's units.
```

### 5.2 `model.py` — domain data + interaction state

**Incoming domain data (populated by socket messages):**

```python
spaces:          dict[str, Space]           # id → coordinatus Space (scene graph)
point_channels:  dict[str, list[Point]]     # channel_id → list of coordinatus Points
```

> **Space reconstruction:** `apply_message` for `"space"` messages looks up
> `parent = model.spaces.get(msg["parent_id"])`. If the parent is not yet known,
> the message is held in a `_pending_spaces: list[dict]` buffer. After each
> successful insertion the buffer is re-scanned until no further spaces can be
> resolved (handles any declaration order; unresolvable cycles raise `ValueError`).
> An implicit identity root is pre-seeded so root spaces (`parent_id: null`)
> always resolve immediately.

**Interaction / camera state:**

```python
view_space:       Space        # camera Space; parent = spaces[selected_node_id]
hovered_node_id:  str | None
selected_node_id: str | None   # id of the Space whose origin is the view origin
mouse_x:          float        # normalised [-1, 1] from right-panel mouse
mouse_y:          float
```

**Key methods:**

```python
def apply_message(self, msg: dict) -> None:
    """Mutate domain data from one decoded socket message. No threading."""

def to_scene(self) -> Scene:
    """Build and return a full Scene snapshot.
    All coordinate conversion happens here via coordinatus relative_to().
    Called on the Qt thread — no locking needed."""

def select_space(self, space_id: str | None) -> None:
    """Set the selected space and reset view_space to identity with that
    Space as parent. If space_id is None, parent reverts to the implicit
    identity root. Called by Presenter on node click."""

def set_interaction(self, **kwargs) -> None:
    """Update hovered_node_id, mouse_x, mouse_y. Does NOT touch view_space."""

def pan(self, dx: float, dy: float) -> None:
    """Translate view_space by (dx, dy) in current view units (plot panel)."""

def zoom(self, factor: float, cx: float, cy: float) -> None:
    """Scale view_space by factor around (cx, cy) in current view units (plot panel)."""
```

**`view_space` mechanics:**

`view_space` is a `Space` whose **parent is the currently selected Space**
(`model.spaces[selected_node_id]`). When no node is selected the parent is
the implicit identity root. `select_space(space_id)` both sets
`selected_node_id` and resets `view_space` to a `Space(transform=identity,
parent=spaces[space_id])` — placing the view origin at that Space's origin,
aligned with its axes. If `space_id` is `None`, `view_space` is reset to
`Space(transform=identity, parent=implicit_root)`.

Pan and zoom **replace** `view_space` with a new `Space` carrying the updated
transform (immutable-value style — no in-place mutation):

```python
def pan(self, dx, dy):
    # TODO: implement pan by modifying view_space transform
    ...

def zoom(self, factor, cx, cy):
    # TODO: implement zoom around (cx, cy) by modifying view_space transform
    ...
```

Because `view_space` is only ever used as the **target** of `relative_to()`,
replacing it with a new object on each event is safe — no existing `Point`
stores it as its own space.

### 5.3 `view.py` — dumb Qt renderer

**`ViewEventHandler` protocol** (what the View calls back into; implemented by Presenter):

```python
class ViewEventHandler(Protocol):
    def on_node_hovered(self, node_id: str | None) -> None: ...
    def on_node_clicked(self, node_id: str) -> None: ...
    def on_plot_pan(self, dx: float, dy: float) -> None: ...
    # dx, dy in the plot panel's current view-space units
    def on_plot_zoom(self, factor: float, cx: float, cy: float) -> None: ...
    # factor > 1 = zoom in; cx, cy = cursor position in view-space units
    def on_mouse_moved(self, norm_x: float, norm_y: float) -> None: ...
    # norm_x, norm_y normalised [-1, 1] from the right (plot) panel
```

**`VisualizerView` public interface:**

```python
class VisualizerView(QMainWindow):
    def set_event_handler(self, handler: ViewEventHandler) -> None: ...
    def render(self, scene: Scene) -> None: ...
```

No `set_view_range()` — there is no feedback loop to break because the View never
moves the ViewBoxes programmatically. The graph panel's ViewBox uses pyqtgraph's
built-in mouse interaction; on the very first render `enableAutoRange()` is called
once so all nodes are visible initially, after which the ViewBox manages its own
state. The plot panel's ViewBox has native mouse disabled; pan/zoom is driven
exclusively by custom mouse events that fire `on_plot_pan` / `on_plot_zoom`.

`render(scene)` compares against the previous render to decide between a **full
rebuild** (topology changed: different node ids or curve count) or a **style-only
update** (same topology, different colors/sizes):

```python
# Full rebuild guard (O(n)):
prev_ids = frozenset(id for s in prev_scene.graph.scatter for id in s.ids)
curr_ids = frozenset(id for s in scene.graph.scatter  for id in s.ids)
needs_rebuild = (curr_ids != prev_ids
                 or len(scene.graph.curves) != len(prev_scene.graph.curves))
```

**Graph panel pan/zoom — native ViewBox interaction enabled:**

The graph panel's ViewBox uses pyqtgraph's built-in mouse interaction (left-drag
to pan, scroll to zoom). No custom event handling needed. Node hover and click are
still intercepted via `sigClicked` / item `hoverEvent` on each scatter spot.

**Plot panel pan/zoom — native ViewBox interaction disabled:**

```python
self._plot_vb.setMouseEnabled(x=False, y=False)
```

Mouse events captured manually on the plot panel widget:
- **Left-button drag**: compute pixel delta → convert to data-space delta via
  `vb.mapSceneToView()` → fire `handler.on_plot_pan(dx, dy)`.
- **Scroll wheel**: fire `handler.on_plot_zoom(factor, cx, cy)` where
  `factor = 1.15 ** sign(wheel_delta)` and `(cx, cy)` = cursor in view-space coords.

### 5.4 `presenter.py` — orchestrator

```python
class Presenter(QObject):
    def __init__(self, model: VisualizerModel, view: VisualizerView,
                 inbox: queue.Queue[dict]) -> None: ...

    def start(self) -> None:
        """Register self as view's event handler. Start the 16 ms render timer."""

    def _on_tick(self) -> None:
        # 1. Drain inbox (max 200 messages per tick to avoid frame drops)
        # 2. model.apply_message(msg) for each
        # 3. scene = model.to_scene()
        # 4. view.render(scene)
        # (no range push needed — view_space is already encoded in scene coords)

    # ViewEventHandler implementation
    def on_node_hovered(self, node_id):
        self.model.set_interaction(hovered_node_id=node_id)

    def on_node_clicked(self, node_id):
        # Toggle: clicking the already-selected node deselects it
        # (view_space parent reverts to implicit root)
        new_id = None if node_id == self.model.selected_node_id else node_id
        self.model.select_space(new_id)

    def on_plot_pan(self, dx, dy):
        self.model.pan(dx, dy)

    def on_plot_zoom(self, factor, cx, cy):
        self.model.zoom(factor, cx, cy)

    def on_mouse_moved(self, nx, ny):
        self.model.set_interaction(mouse_x=nx, mouse_y=ny)
```

`_on_tick()` always calls `model.to_scene()` and `view.render(scene)` unconditionally
— this keeps the render loop simple.  The cost of `to_scene()` on a tick where only
`on_mouse_moved` fired is negligible.

### 5.5 `server.py` — agnostic TCP listener

Move `SocketServer` from `visualization2.py`. Its `_DataBridge` (Qt `QObject` with signals) is replaced by a plain `queue.Queue[dict]`.

The server has **no knowledge of message content**: it reads newline-delimited bytes, decodes each line as JSON, and puts the resulting `dict` into the queue unchanged. All message interpretation lives in `Model.apply_message()`. This means the server never needs to change when the protocol evolves.

### 5.6 Scene Generation Logic — `to_scene()` in detail

All positions go through `relative_to(view_space)` before being placed in any Spec.
On `ValueError` (no common ancestor with `view_space`) the individual object is
silently skipped.

#### 5.6.1 Space origins → graph nodes (GraphScene — absolute coords)

For each `space_id, space` in `model.spaces`:

1. Convert the Space's origin to root (world) coordinates: `Point([0.0, 0.0], space=space).to_absolute()` → `(x, y)`.
2. Append one spot to a `ScatterSpec` (default grey; blue if hovered; purple if selected).
3. Extract the Space's world orientation to align its label:
   ```python
   x_axis_end = Point([1.0, 0.0], space=space).to_absolute().coords
   rotation = degrees(atan2((x_axis_end - xy)[1], (x_axis_end - xy)[0]))
   ```
4. Append `LabelSpec(x, y, text=space_id, rotation=rotation, anchor=(0.5, -0.3))`.

#### 5.6.2 Space hierarchy edges (GraphScene — absolute coords)

For each Space in `model.spaces` that has a parent also in `model.spaces`:
- Convert both origins to absolute world coords → `p0` (parent), `p1` (child).
- `CurveSpec(points=np.array([[p0x,p0y],[p1x,p1y]]), color=GREY, width=1.2)`.
- Directed arrowhead at 70% along the edge (parent → child):
  ```python
  t     = 0.70
  apex  = p0 + t * (p1 - p0)
  d     = (p1 - p0) / np.linalg.norm(p1 - p0)
  angle = degrees(atan2(d[1], d[0]))   # CCW from +x (View negates for Qt)
  size  = 0.06 * np.linalg.norm(p1 - p0)
  ArrowSpec(x=apex[0], y=apex[1], angle=angle, size=size, color=GREY)
  ```
  > Arrowhead size is proportional to edge length (scale-invariant in data space).

#### 5.6.3 Point channels → PlotScene per-channel ScatterSpec (view-space coords)

For each `channel_id` in `model.point_channels`:
- For each `Point` in the channel, try `relative_to(view_space)` → collect `(x, y)`.
- Build one `ScatterSpec` per channel with a consistent palette color.
- Skip points that raise `ValueError` (no shared ancestor with `view_space`).
- Append to `PlotScene.scatter` (right panel), **not** `GraphScene.scatter`.

---

## 6. Socket Protocol

All messages are newline-terminated JSON objects (one object per line). The server
is a pure pass-through — it never inspects message keys. All decoding logic lives
in `Model.apply_message()`.

### 6.1 Compound state-update message

The primary message type carries the **complete current state** of the scene in
one shot. Both `spaces` and `points` are optional; omitting a key leaves the
corresponding domain data unchanged.

```jsonc
{
  // Optional — replaces ALL previously stored spaces.
  // Spaces may be listed in any order; the Model resolves parent references
  // with a multi-pass pending buffer (see section 5.2).
  "spaces": [
    {"id": "world",  "parent_id": null,
     "transform": [[1,0,0],[0,1,0],[0,0,1]]},
    {"id": "sensor", "parent_id": "world",
     "transform": [[0.707,-0.707,1],[0.707,0.707,0],[0,0,1]]}
  ],

  // Optional — replaces ALL previously stored point channels.
  // Each entry is one channel: a named list of 2-D coordinates in a given space.
  "points": [
    {"channel": "lidar",   "space_id": "sensor", "coords": [[1.0,2.0],[3.0,4.0]]},
    {"channel": "targets", "space_id": "world",  "coords": [[0.5,0.5]]}
  ]
}
```

Transform matrices are **row-major** (NumPy default): `np.array(entry["transform"])`
gives the correct `(D+1) × (D+1)` homogeneous matrix directly.

### 6.2 Display-space override message

Selects which Space is used as the view origin for the plot panel.

```jsonc
{"type": "set_display_space", "space_id": "world"}
```

### 6.3 Design notes

| Property | Detail |
|---|---|
| State model | `spaces` and `points` are **replace-all** (last write wins per tick). |
| Declaration order | Spaces inside a compound message may appear in any order; the Model resolves parent links in multiple passes. |
| Missing parent | If a Space's `parent_id` cannot be resolved after all passes, a `ValueError` is raised (cycle or dangling reference). |
| Projection failure | Points whose Space shares no ancestor with `view_space` are silently dropped from the plot panel (still visible on the graph panel). |

---

## 7. Implementation Steps

### Step 1 — `viz/scene.py`: Scene dataclasses ✅ DONE
- Create `src/coordinatus/viz/__init__.py` and `scene.py`.
- Implement all Spec dataclasses exactly as in section 5.1:
  `CurveSpec`, `FilledPolygonSpec`, `ScatterSpec`, `LabelSpec`,
  `GraphScene`, `PlotScene`, `Scene`.
- Add unit tests (`tests/coordinatus/viz/test_scene.py`) — construction only, no Qt.

### Step 2 — `viz/model.py`: VisualizerModel ✅ DONE
- Implement `VisualizerModel` with all domain + interaction fields per section 5.2
  (including `view_space: Space2D`; **no** `x_range`/`y_range`/`display_space_id`).
- Implement `apply_message(msg)` for both message types: compound state update (`spaces`, `points`) and display-space override.
- Implement `pan(dx, dy)` and `zoom(factor, cx, cy)` per section 5.2 mechanics.
- Implement `to_scene()` following the generation logic in section 5.6:
  - Space nodes + hierarchy edges with arrows → `GraphScene` using `to_absolute()` coords (5.6.1–2).
  - Point channels → `PlotScene.scatter` entries via `relative_to(view_space)` (5.6.3).
  - Hover/selection colors baked in.
- Add unit tests (`tests/coordinatus/viz/test_model.py`) — no Qt, no pyqtgraph.

### Step 3 — `viz/server.py`: SocketServer refactor ✅ DONE
- Move `SocketServer` from `visualization2.py`.
- Replace `_DataBridge` (QObject with Qt signals) with a plain `queue.Queue[dict]`.
- Server now puts raw decoded `dict` into the queue; no Qt dependency.

### Step 4 — `viz/view.py`: VisualizerView ✅ DONE
- Define `ViewEventHandler` Protocol per section 5.3
  (`on_node_hovered`, `on_node_clicked`, `on_plot_pan`, `on_plot_zoom`, `on_mouse_moved`).
- Refactor current `MainWindow`, `DirectedGraphPanel`, `LivePlotPanel` into `VisualizerView`.
- Implement `set_event_handler(handler)` and `render(scene: Scene)`:
  - Graph panel: topology diff guard (section 5.3) → full rebuild vs. style-only update.
  - Plot panel: update scatter and curves from `PlotScene`.
  - **No** `set_view_range()` or `sigRangeChanged` wiring — removed entirely.
- Graph panel: leave native ViewBox mouse interaction **enabled**. On first render call
  `enableAutoRange()` once so all nodes are visible; after that the ViewBox is self-managing.
- **Disable** native ViewBox pan/zoom on the **plot** panel:
  `self._plot_vb.setMouseEnabled(x=False, y=False)`.
- Add custom mouse event handlers on the **plot** panel for pan (left-button drag → `on_plot_pan`) and zoom (scroll wheel → `on_plot_zoom`).

### Step 5 — `viz/presenter.py`: Presenter
- Implement `Presenter(QObject)` as described.
- `start()` registers self on view and starts QTimer.
- `_on_tick()` drains queue (max 100 messages per tick to avoid frame drops), calls model, calls view.
- Implement all `ViewEventHandler` methods.

### Step 6 — Wire in `visualization2.py`
- Keep sys.path fix at top.
- Import from `viz/` subpackage.
- `main()` creates `queue.Queue`, `SocketServer`, `VisualizerModel`, `VisualizerView`, `Presenter`.
- `server.start()`, `presenter.start()`, `app.exec()`.
- Example Spaces pre-loaded by Model (not by View as currently).

### Step 7 — Update `examples/pyqtgraph_socket_sender.py`
- Add examples of all new socket message types (`space`, `points`).
- Show a small Coordinatus scene: two child Spaces, a set of Points in each, projected to a shared world Space.

---

## 8. Resolved Decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Matrix layout in socket JSON | **Row-major** (NumPy default). `np.array(msg["transform"])` directly. |
| Q2 | Point projection failure (no common ancestor) | **Silently drop** — the disconnected Space is visible on the graph panel; the user will understand why points are absent. |
| Q3 | Topology diff strategy | **Full set comparison** for now (`set(edges) != set(prev_edges)`). Introduce a `topology_version` counter only if performance becomes a problem. |
| Q4 | Auto-fit when new data arrives | **No auto-fit after first render.** On the very first render the **graph panel** ViewBox does a one-shot `enableAutoRange()` to show all nodes, then its native ViewBox manages all subsequent pan/zoom. The **plot panel** ViewBox is always mouse-disabled; all pan/zoom on the right panel is driven exclusively by `model.pan()`/`model.zoom()` triggered by custom mouse events. |
| Q5 | Message granularity | **Compound messages** — one JSON object carries `"spaces": [...]` and/or `"points": [...]` as complete state snapshots (replace semantics). The server is a pure pass-through; it never inspects keys. |
