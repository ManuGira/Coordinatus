# Proposal B — ProjectionSpace Implementation Plan

## Background

Proposal B adds first-class projection support to the scene graph. A `ProjectionSpace`
is a child space whose transform *reduces* dimensionality (e.g. 3D → 2D), while still
living naturally in the hierarchy. The goal is that `point.relative_to(screen)` works
transparently regardless of whether the path between the two spaces includes a
dimension-changing step.

The pre-requisite bug (`transform_coordinate` using D_in instead of D_out) has already
been fixed in commit `777501a`.

---

## What changes and where

### 1. `coordinate.py` — `transform_coordinate`

**Status: done.**  
Slices output with `transform.shape[0] - 1` (D_out) instead of D.

---

### 2. `space.py` — replace `compute_relative_transform_to` with LCA traversal

**Status: done.**  
Replaced the old `inv(absolute(target)) @ absolute(self)` body with an LCA-based path
traversal that handles non-square (projection) transforms correctly.

The new body calls `_find_lca`, builds the forward path (self → LCA) by multiplying
`node.transform` upward, then the backward path (LCA → target) by calling `_invert_step`
on each node and multiplying downward.

Helpers added (module-private, `space.py`):

- `_find_lca(a, b)` — walks ancestor sets to find the Lowest Common Ancestor; raises
  `ValueError` for unrelated trees.
- `_path_from(ancestor, descendant)` — returns `[child_of_ancestor, …, descendant]`;
  returns `[]` when `ancestor is descendant`.
- `_invert_step(node)` — returns `node.projection_matrix` for `ProjectionSpace` nodes,
  `np.linalg.inv(node.transform)` for square transforms, or raises `ValueError` for
  non-square non-`ProjectionSpace` nodes.

---

### 3. `space.py` — add `ProjectionSpace` class

**Status: done.**  
`ProjectionSpace(projection_matrix, parent)` stores `projection_matrix` (the
`(D_out+1) × (D_in+1)` matrix mapping FROM parent TO child) and saves its pseudo-inverse
as `self.transform`, so `compute_absolute_transform()` keeps working by chaining
matrices upward unchanged.

`D_in` == child dimension, `D_out` == parent dimension.

---

### 4. `space.py` — `get_root` / `compute_absolute_transform`

**Status: no changes needed.**  
These already work by chaining `node.transform` upward. `ProjectionSpace.transform`
is the pseudo-inverse; the chain still produces a matrix at each level. No changes.

---

### 5. `space.py` — back-projection helpers on `ProjectionSpace`

**Status: deferred.**  
Two optional methods for callers that need the exact inverse direction:

```python
def unproject_with_depth(self, coord: 'Coordinate', depth: float) -> 'Coordinate':
    """Reconstruct the unique parent-space point that projects to `coord` at `depth`.

    `depth` is the coordinate along the dropped axis in the parent space
    (e.g. z in camera space for a 3D → 2D projection).
    """

def ray_through(self, coord: 'Coordinate') -> 'Ray':
    """Return the parametric ray (in parent space) of all points projecting to `coord`.

    Requires the optional `Ray` data class (see section 6).
    """
```

---

### 6. New `Ray` data class

**Status: deferred.**

```python
@dataclass
class Ray:
    """A parametric ray: origin + t * direction in a given Space."""
    origin: Point
    direction: Vector
```

Used by `ProjectionSpace.ray_through()`. Can be added in a later PR.

---

### 7. `__init__.py` — public re-exports

**Status: done.**  
`ProjectionSpace` added to `src/coordinatus/__init__.py`.

If `Ray` is added later:
```python
from .coordinate import Ray
```

---

### 8. `transforms/dimension.py` — no changes needed

**Status: done (no changes required).**  
The existing `project_xyz_to_xy()` etc. already produce the correct homogeneous matrices
and are used directly as the `projection_matrix` argument to `ProjectionSpace`.

---

## Test plan

All new tests live in the existing test files following the existing conventions.

### `tests/coordinatus/test_space.py` — new classes

**`TestFindLCA`** ✓
- LCA of two siblings with a common direct parent
- LCA when one node is the ancestor of the other
- LCA of a deeper ancestor (two levels up)
- LCA of a node with itself
- LCA of two cousins (sharing a grandparent)
- Raises `ValueError` for unrelated trees

**`TestPathFrom`** ✓
- Direct child path (one step)
- Two-step path (grandparent → grandchild)
- Deep path (multiple steps, order verified)
- Path to self is empty list

**`TestInvertStep`** ✓
- Square transform → returns matrix inverse
- `ProjectionSpace` → returns `projection_matrix` (not pseudo-inverse)
- Non-square, non-`ProjectionSpace` → raises `ValueError`

**`TestProjectionSpace`** ✓
- `ProjectionSpace(proj_3x4, parent)` stores `projection_matrix` correctly
- `D_in` == 2, `D_out` == 3 for a 3D→2D projection
- `transform` is the pseudo-inverse of `projection_matrix`
- Parent is correctly stored
- Is a `Space` instance
- `compute_absolute_transform()` chains correctly

**`TestComputeRelativeTransformToWithProjection`** ✓
- Two siblings → same result as before (regression guard)
- `world.compute_relative_transform_to(screen)` → applies `projection_matrix`
- `screen.compute_relative_transform_to(world)` → applies pseudo-inverse
- Full scene-graph: 3D point projected through homogeneous transform gives correct xy
- Sibling of a `ProjectionSpace` converts correctly via the LCA path

### `tests/coordinatus/test_coordinate.py` — new class

**`TestCoordinateRelativeToProjection`** ✓
- `Point([x,y,z], space=world).relative_to(screen)` → 2D Point with correct shape
- Result values match `project_xyz_to_xy()` applied directly
- Returned coordinate's space is `screen`
- `relative_to` preserves `Point` type
- `Point` in 2D screen → `relative_to(world)` → 3D Point via pseudo-inverse
  (least-squares back-projection, not exact)
- Batch of 3D points projected to screen gives `(2, N)` result with correct values
- 3D point in a translated child of world is correctly projected to screen

### `tests/coordinatus/test_space.py` — regression guard ✓

Existing `TestComputeRelativeTransformTo` tests all pass unchanged.

---

## Implementation order

1. ✓ Add `_find_lca`, `_path_from`, `_invert_step` helpers to `space.py` — with unit tests
2. ✓ Replace `compute_relative_transform_to` body with LCA traversal — all existing tests pass
3. ✓ Add `ProjectionSpace` class to `space.py` — with unit tests
4. ✓ Add `ProjectionSpace` to `__init__.py` public API
5. ⬜ (Optional) Add `unproject_with_depth` and `Ray` / `ray_through`

---

## Backwards compatibility

- All existing `Space`, `Coordinate`, `Point`, `Vector` usage is unchanged.
- `compute_relative_transform_to` for two same-dimensional spaces produces the same
  result as before (the LCA path reduces to the old `inv(absolute(target)) @ absolute(self)`
  formula).
- The `transform_coordinate` fix (already committed) is a no-op for square matrices.
