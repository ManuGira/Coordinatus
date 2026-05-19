---
type: bmad-distillate
sources:
  - "NOTES_visualization_rework.md"
downstream_consumer: "general"
created: "2026-05-19"
token_estimate: 1666
parts: 1
---

## Meta
- File under rework: `src/coordinatus/visualization.py` — only file to edit
- Do NOT edit: `space.py`, `coordinate.py`, `transforms/`, test files (unless public API changed)
- Tests: `tests/coordinatus/test_visualization.py`; run: `uv run pytest tests/coordinatus/test_visualization.py -v`
- Verify behaviour: `uv run examples/space_hierarchy.py`
- Branch: `feature/space-hierarchy-visualization`; CI: `./ci.ps1`

## Status
- Step 1 ✅ done May 19 2026; animation code removed, CI passing, 100% test coverage
- Step 2 ✅ done May 19 2026; `_view_space` added to `__init__`, `_pan_xlim`/`_pan_ylim` removed from `__init__`, 84 tests passing
- Steps 3–7 not started; Step 7 deferred until Steps 1–6 pass tests

## Current Architecture (as of May 19 2026)
- `draw_space_hierarchy(spaces, labels, title)`: two subplots — `ax_graph` (networkx directed graph left), `ax_axes` (coordinate frames via `draw_space_axes` right)
- `_HierarchyInteractor`: owns all interactive state and event handlers
- `data.reference_space`: currently set to one of the user's `Space` objects (selected node); used as rendering reference — TARGET REMOVES THIS ROLE
- Pan/zoom: `ax_axes.set_xlim/set_ylim` → view limits drift from `[-2.5, 2.5]` — TARGET FIXES THIS
- `_HierarchyRenderData` fields: spaces, labels, graph, pos, node_labels, colors, reference_space

## Target Architecture
- Rendering reference = always a temporary `Space` (`_view_space`); never one of the user's spaces
- `_view_space = Space(transform=np.eye(3), parent=selected_space)`: parent=selected node; transform=view offset (identity = "look straight at selected space")
- Pan: `_view_space.transform = translate2D(dx, dy) @ _view_space.transform`
- Zoom: `_view_space.transform = translate2D(cx,cy) @ scale2D(f,f) @ translate2D(-cx,-cy) @ _view_space.transform`
- Selection: `_view_space = Space(transform=np.eye(3), parent=new_selected_space)`
- Render: always pass `_view_space` as `reference_space` to `draw_space_axes`; `ax_axes.set_xlim/ylim(-2.5, 2.5)` always fixed — zoom/pan encoded in `_view_space` not axes limits
- Highlight: `is_reference = space is _view_space.parent` (replaces `space is data.reference_space`)
- Animation (Step 7, future): tween `_view_space.transform` from `M_old` (old selected space in new selected frame) → `np.eye(3)` over N frames; always "approach identity"

## Key Symbols
- `Space(transform=M, parent=S)`: child space; `Space2D()`: identity 3x3 root; use `Space2D()` as parent when `data.reference_space` is None
- `space.compute_absolute_transform()`: world matrix self→root; `space.get_root()`: walks parent chain
- `translate2D(tx,ty)` (`transforms/translate.py`): 3x3 translation matrix
- `scale2D(sx,sy)` (`transforms/scale.py`): 3x3 scale matrix
- `trs2D`/`trks2D` (T@R@S / T@R@K@S): removed in Step 1; re-added in Step 7
- `_decompose_trks2d(M)`: QR decompose 3x3 affine → (tx,ty,θ,kx,sx,sy); removed Step 1; re-added Step 7
- `_interpolate_trks2d(M0,M1,t)`: smoothstep interpolation; removed Step 1; re-added Step 7
- `_draw_axes_subplot`: draws all frames, returns `{id(space): [artists]}`
- `_draw_hierarchy_subplot`: draws networkx graph

## Steps

### Step 1 ✅ — Remove animation code (done)
- Removed: `_ANIM_FRAMES`, `_ANIM_INTERVAL_MS`, `_decompose_trks2d`, `_interpolate_trks2d`, imports `trs2D`/`trks2D`
- Removed `__init__` fields: `_anim_timer`, `_anim_frame`, `_anim_M_old`, `_anim_M_new`, `_anim_root`, `_anim_R_root_inv`, `_anim_bg`
- Removed methods: `_start_anim`, `_anim_step`
- Removed `if self._anim_timer is not None: return` guard from `_on_press`, `_on_motion`, `_on_axes_leave`
- Cleaned `_draw_axes_subplot`: removed `reference_space_override` param and `effective_ref` logic
- Post-step state: pan/zoom still via `set_xlim/ylim`; no animation

### Step 2 — Introduce `_view_space` (NEXT)
1. `__init__`: add `self._view_space: Space = Space(transform=np.eye(3), parent=data.reference_space or Space2D())`
2. `data.reference_space` stays for graph highlight only; no longer drives rendering
3. Remove `_pan_xlim`, `_pan_ylim` from `__init__` (replaced by `_pan_M0` in Step 4)

### Step 3 — Update `_draw_axes_subplot` and callers
- New sig: `_draw_axes_subplot(ax, data, view_space: Space, hovered_node: int|None = None) -> dict[int, list]`
- Replace `data.reference_space`/`reference_space_override` with `view_space`; `is_reference = space is view_space.parent`
- Fix limits: `ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)` — remove `xlim`/`ylim` params
- `_node_at_axes_pos`: use `self._view_space` not `self.data.reference_space`
- `_redraw`/`_redraw_hover`: pass `self._view_space` to `_draw_axes_subplot`

### Step 4 — Rework pan → `_view_space` mutation
- `_start_pan`: replace `_pan_xlim`/`_pan_ylim` with `self._pan_M0 = self._view_space.transform.copy()`; keep `_pan_start_display`, `_pan_inv_transform`
- `_on_motion` (during pan): `start_data = _pan_inv_transform.transform(_pan_start_display)`; `curr_data = _pan_inv_transform.transform((event.x, event.y))`; `dx, dy = start_data - curr_data`; `self._view_space.transform = translate2D(dx, dy) @ self._pan_M0`; call `_redraw()`
- Sign note: drag right should reveal right → delta `+dx,+dy`; test and flip sign if wrong

### Step 5 — Rework zoom → `_view_space` mutation
- `_on_scroll`: guard `if event.inaxes is not self.ax_axes or event.xdata is None: return`; `factor = 0.9 if event.step > 0 else 1.1` (scroll-up = zoom in); `cx, cy = event.xdata, event.ydata`; `self._view_space.transform = translate2D(cx,cy) @ scale2D(factor,factor) @ translate2D(-cx,-cy) @ self._view_space.transform`; call `_redraw()`
- Note: `event.xdata/ydata` = matplotlib data coords = view-space coords when axes always fixed at `[-2.5, 2.5]`

### Step 6 — Update selection
- `_select_node`: `if new_ref is self._view_space.parent: return`; `self.selected_node = node_id`; `self.data.reference_space = new_ref` (graph highlight only); `self._view_space = Space(transform=np.eye(3), parent=new_ref)`; call `_redraw()`

### Step 7 — Re-add animation (FUTURE, after Steps 1–6 pass tests)
- `_select_node`: compute `M_start` = old view expressed in new parent's frame; tween `_view_space.transform` from `M_start` → `np.eye(3)` using `_interpolate_trks2d(M_start, np.eye(3), t)` per frame; call `_redraw()` each frame; use blitting (capture static bg once, swap only space artists)
- Re-add `_decompose_trks2d`, `_interpolate_trks2d`, `trs2D`/`trks2D` imports at this point (functions were correct; only animation wiring was messy)
