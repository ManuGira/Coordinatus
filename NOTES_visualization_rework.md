# Visualization Rework Plan

## Context

This document describes a planned rework of `src/coordinatus/visualization.py`,
specifically the `_HierarchyInteractor` class and its helpers.
It is intended to be executed step by step by an agent.

**Run any example to verify behaviour:**
```powershell
uv run examples/space_hierarchy.py
```

**Run tests after each step:**
```powershell
uv run pytest tests/coordinatus/test_visualization.py
```

---

## Current architecture (as of May 19, 2026)

- `draw_space_hierarchy(spaces, labels, title)` creates two matplotlib subplots:
  - **Left panel (`ax_graph`)**: networkx directed graph of the space hierarchy
  - **Right panel (`ax_axes`)**: coordinate frames drawn via `draw_space_axes`
- `_HierarchyInteractor` owns all interactive state and event handlers.
- `data.reference_space` is set to one of the user's `Space` objects (the selected node).
- Pan/zoom work by calling `ax_axes.set_xlim / set_ylim` — the view limits drift away from `[-2.5, 2.5]`.
- A blitted animation exists (`_start_anim`, `_anim_step`) that interpolates between
  two absolute transform matrices using `_decompose_trks2d` / `_interpolate_trks2d`.

### Key symbols to know

| Symbol | File | Role |
|---|---|---|
| `Space` | `space.py` | Wraps a `(D+1)x(D+1)` homogeneous matrix + optional parent |
| `Space2D()` | `space.py` | Identity 3x3 root space |
| `Space(transform=M, parent=S)` | `space.py` | Child space |
| `space.compute_absolute_transform()` | `space.py` | Returns world matrix (self → root) |
| `space.get_root()` | `space.py` | Walks parent chain to root |
| `translate2D(tx, ty)` | `transforms/translate.py` | 3x3 translation matrix |
| `scale2D(sx, sy)` | `transforms/scale.py` | 3x3 scale matrix |
| `trs2D(tx, ty, angle, sx, sy)` | `transforms/__init__.py` | T @ R @ S composite |
| `trks2D(tx, ty, angle, kx, ky, sx, sy)` | `transforms/__init__.py` | T @ R @ K @ S composite |
| `_decompose_trks2d(M)` | `visualization.py` | QR decompose 3x3 affine → (tx,ty,θ,kx,sx,sy) |
| `_interpolate_trks2d(M0, M1, t)` | `visualization.py` | Smoothstep interpolation between two 3x3 matrices |
| `draw_space_axes(ax, space, reference_space, ...)` | `visualization.py` | Draws one space's frame |
| `_draw_axes_subplot(ax, data, hovered_node, xlim, ylim, reference_space_override)` | `visualization.py` | Draws all space frames; returns `{id(space): [artists]}` |
| `_draw_hierarchy_subplot(ax, data, selected_node, hovered_node)` | `visualization.py` | Draws the networkx graph |
| `_HierarchyRenderData` | `visualization.py` | Dataclass: spaces, labels, graph, pos, node_labels, colors, reference_space |

---

## Target architecture

### Core idea

The **reference space used for rendering is always a temporary `Space`** — never one of
the user's spaces. Call it `_view_space`. It is defined as:

```python
_view_space = Space(transform=np.eye(3), parent=selected_space)
```

- `parent` = the currently selected space (the node with the black outline)
- `transform` = the view offset matrix (identity = "look straight at the selected space")

**Pan** mutates `_view_space.transform` by pre-composing a translation:
```python
_view_space.transform = translate2D(dx, dy) @ _view_space.transform
```

**Zoom** mutates `_view_space.transform` by composing a scale around the cursor:
```python
_view_space.transform = (
    translate2D(cx, cy) @ scale2D(f, f) @ translate2D(-cx, -cy)
    @ _view_space.transform
)
```

**Selection** creates a fresh view space:
```python
_view_space = Space(transform=np.eye(3), parent=new_selected_space)
```

**Rendering** always passes `_view_space` to `draw_space_axes` as `reference_space`,
and `ax_axes.set_xlim(-2.5, 2.5)` / `set_ylim(-2.5, 2.5)` are **always fixed** —
zoom/pan are encoded in `_view_space`, not in the axes limits.

**Highlight** (`is_reference`): a space is highlighted when `space is _view_space.parent`,
not when `space is data.reference_space`.

**Animation** (to be re-added after all steps are done): tween `_view_space.transform`
from `M_old` (the old selected space expressed in new selected space coordinates) to
`np.eye(3)` over N frames. This is simpler than the current approach because the animation
is always "approach identity" regardless of which two spaces are involved.

---

## Steps

### Step 1 — Remove animation code

Delete the following from `visualization.py`:

- Module-level constants: `_ANIM_FRAMES`, `_ANIM_INTERVAL_MS`
- Functions: `_decompose_trks2d`, `_interpolate_trks2d`
- Import: `trs2D`, `trks2D` from `.transforms` (keep `translate2D`, `scale2D`, `rotate2D`)
- In `_HierarchyInteractor.__init__`: remove animation state fields:
  `_anim_timer`, `_anim_frame`, `_anim_M_old`, `_anim_M_new`,
  `_anim_root`, `_anim_R_root_inv`, `_anim_bg`
- Methods: `_start_anim`, `_anim_step`
- In `_on_press`: remove `if self._anim_timer is not None: return` guard
- In `_on_motion`: remove `if self._anim_timer is not None: return` guard
- In `_on_axes_leave`: remove `if self._anim_timer is not None: return` guard
- Also clean up `_draw_axes_subplot`: remove `reference_space_override` parameter
  and the `effective_ref` logic that goes with it (it was only used by the animation)

After this step the interactor should still work: pan/zoom via `set_xlim/ylim`, no animation.

---

### Step 2 — Introduce `_view_space`

In `_HierarchyInteractor.__init__`:

1. Remove the `data.reference_space` assignment that pointed to the initial root.
   `data.reference_space` field on `_HierarchyRenderData` can remain as a convenience
   for the graph highlight (which node is selected), but it should no longer be used
   as the rendering reference — that role is taken by `_view_space`.

2. Add:
```python
# _view_space is the rendering reference — always a temporary Space
# whose parent is the selected user-space.
self._view_space: Space = Space(
    transform=np.eye(3),
    parent=data.reference_space,   # initial selected space (root or None)
)
```
If `data.reference_space` is `None` (multi-root case), use `Space2D()` as the parent.

3. Remove the pan state fields `_pan_xlim`, `_pan_ylim`
   (they are replaced by `_pan_M0` in Step 4).

---

### Step 3 — Update `_draw_axes_subplot` and `_draw_axes_subplot` callers

Update `_draw_axes_subplot` signature:

```python
def _draw_axes_subplot(
    ax,
    data: _HierarchyRenderData,
    view_space: Space,            # NEW — replaces reference_space_override + data.reference_space
    hovered_node: int | None = None,
) -> "dict[int, list]":
```

Internal changes:
- Replace every `data.reference_space` / `reference_space_override` reference with `view_space`
- `is_reference = space is view_space.parent` (the selected space)
- `ax.set_xlim(-2.5, 2.5)` and `ax.set_ylim(-2.5, 2.5)` — always fixed, no parameters
- Remove `xlim`, `ylim` parameters entirely

Update `_node_at_axes_pos` to use `self._view_space` instead of `self.data.reference_space`.

Update `_redraw` and `_redraw_hover` to pass `self._view_space` to `_draw_axes_subplot`.

---

### Step 4 — Rework pan into `_view_space` mutation

Replace the pan implementation:

**Old:**
```python
def _start_pan(self, event):
    self._pan_start_display = (event.x, event.y)
    self._pan_xlim = self.ax_axes.get_xlim()
    self._pan_ylim = self.ax_axes.get_ylim()
    self._pan_inv_transform = self.ax_axes.transData.inverted()

def _on_motion(self, event):
    ...
    dx = start_data[0] - curr_data[0]
    dy = start_data[1] - curr_data[1]
    self.ax_axes.set_xlim(self._pan_xlim[0] + dx, ...)
    self.ax_axes.set_ylim(self._pan_ylim[0] + dy, ...)
    self.fig.canvas.draw_idle()
```

**New:**
```python
def _start_pan(self, event):
    self._pan_start_display = (event.x, event.y)
    self._pan_M0 = self._view_space.transform.copy()
    # Capture axes transform for stable pixel→data conversion during drag.
    self._pan_inv_transform = self.ax_axes.transData.inverted()

def _on_motion(self, event):
    ...
    start_data = self._pan_inv_transform.transform(self._pan_start_display)
    curr_data  = self._pan_inv_transform.transform((event.x, event.y))
    dx = start_data[0] - curr_data[0]
    dy = start_data[1] - curr_data[1]
    self._view_space.transform = translate2D(dx, dy) @ self._pan_M0
    self._redraw()   # see Step 6 for making this fast
```

Remove `_pan_xlim` and `_pan_ylim`. Add `_pan_M0`.

Note on sign: dragging right (cursor moves +x) should reveal content to the right,
meaning the view window shifts left. The view space origin moves right, so the delta
on `_view_space.transform` translation should be `+dx, +dy` relative to the drag
direction. Test and flip sign if wrong.

---

### Step 5 — Rework zoom into `_view_space` mutation

Replace `_on_scroll`:

**Old:**
```python
def _on_scroll(self, event):
    factor = 0.9 if event.step > 0 else 1.1
    cx, cy = event.xdata, event.ydata
    xlim = self.ax_axes.get_xlim()
    ylim = self.ax_axes.get_ylim()
    self.ax_axes.set_xlim(cx + (xlim[0] - cx) * factor, ...)
    self.ax_axes.set_ylim(cy + (ylim[0] - cy) * factor, ...)
    self.fig.canvas.draw_idle()
```

**New:**
```python
def _on_scroll(self, event):
    if event.inaxes is not self.ax_axes or event.xdata is None:
        return
    factor = 0.9 if event.step > 0 else 1.1   # scroll-up = zoom in
    cx, cy = event.xdata, event.ydata           # cursor in view-space coords
    self._view_space.transform = (
        translate2D(cx, cy)
        @ scale2D(factor, factor)
        @ translate2D(-cx, -cy)
        @ self._view_space.transform
    )
    self._redraw()
```

Note: `event.xdata` / `event.ydata` are in matplotlib data coordinates,
which equal view-space coordinates because `ax_axes` is always fixed at `[-2.5, 2.5]`.

---

### Step 6 — Update selection

Replace `_select_node`:

**Old:**
```python
def _select_node(self, node_id):
    old_ref = self.data.reference_space
    new_ref = self.id_to_space[node_id]
    if new_ref is old_ref:
        return
    self.selected_node = node_id
    self.data.reference_space = new_ref
    if old_ref is not None and old_ref.get_root() is new_ref.get_root():
        self._start_anim(...)
    else:
        self._redraw()
```

**New:**
```python
def _select_node(self, node_id):
    new_ref = self.id_to_space[node_id]
    if new_ref is self._view_space.parent:
        return   # already selected
    self.selected_node = node_id
    self.data.reference_space = new_ref   # keep for graph highlight
    self._view_space = Space(transform=np.eye(3), parent=new_ref)
    self._redraw()
```

Animation will be re-added here in a later step.

---

### Step 7 — Re-add animation (future, after Steps 1–6 pass tests)

With the new architecture, animation is much simpler:

1. On `_select_node`, instead of immediately setting `_view_space.transform = eye(3)`,
   compute `M_start`: the old view expressed in the new parent's frame.

2. Tween `_view_space.transform` from `M_start` → `eye(3)` over N frames using
   `_interpolate_trks2d(M_start, np.eye(3), t)`.

3. Each frame: set `self._view_space.transform = M_t`, call `_redraw()`.

4. Use the blitting approach (capture static background once, swap only space artists)
   to keep frame rate high.

The `_decompose_trks2d` and `_interpolate_trks2d` functions can be re-added at this
point (they were correct; only the animation wiring around them was messy).

---

## Files to edit

- `src/coordinatus/visualization.py` — all changes are here

## Files NOT to edit

- `src/coordinatus/space.py`
- `src/coordinatus/coordinate.py`
- `src/coordinatus/transforms/`
- Any test files (unless a test needs updating because a public API changed)

## Tests to watch

```
tests/coordinatus/test_visualization.py
```

Run with:
```powershell
uv run pytest tests/coordinatus/test_visualization.py -v
```
