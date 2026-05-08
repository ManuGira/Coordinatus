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

### 1. `coordinate.py` — `transform_coordinate` (already fixed)

**Status: done.**  
Slices output with `transform.shape[0] - 1` (D_out) instead of D.

---

### 2. `space.py` — replace `compute_relative_transform_to` with LCA traversal

**Current implementation** (broken for projections):
```python
inv_transform = np.linalg.inv(target_space.compute_absolute_transform())
return inv_transform @ self.compute_absolute_transform()
```
This assumes both absolute transforms are square and invertible. It fails whenever
a `ProjectionSpace` sits on the path to the target's root (non-square absolute).

**New implementation: LCA-based path multiplication**

Replace `compute_relative_transform_to` with:

```python
def compute_relative_transform_to(self, target_space: 'Space') -> np.ndarray:
    if self.get_root() is not target_space.get_root():
        raise ValueError(...)

    lca = _find_lca(self, target_space)

    # Forward path: self → LCA (multiply transforms going upward)
    T_forward = np.eye(self.D_in + 1)
    node = self
    while node is not lca:
        T_forward = node.transform @ T_forward
        node = node.parent

    # Backward path: LCA → target (invert each step going downward)
    path_down = _path_from(lca, target_space)   # ordered [child_of_lca, ..., target]
    T_backward = np.eye(lca.D_out + 1)          # identity at LCA dimension
    for node in path_down:
        T_backward = _invert_step(node) @ T_backward

    return T_backward @ T_forward
```

**Helpers to add (module-private)**:

```python
def _find_lca(a: Space, b: Space) -> Space:
    """Return the Lowest Common Ancestor of two spaces in the same hierarchy."""

def _path_from(ancestor: Space, descendant: Space) -> list[Space]:
    """Return the ordered list of spaces [child_of_ancestor, ..., descendant]."""

def _invert_step(node: Space) -> np.ndarray:
    """Return the matrix that reverses a single upward step.
    Raises ValueError for non-invertible, non-ProjectionSpace steps."""
```

**`_invert_step` logic**:
- If `node` is a `ProjectionSpace`: return `node.projection_matrix` (the exact forward
  projection; going "down" into the ProjectionSpace means applying the projection)
- Else if `node.transform` is square: return `np.linalg.inv(node.transform)`
- Else: raise `ValueError("Cannot invert non-square, non-ProjectionSpace transform")`

---

### 3. `space.py` — add `ProjectionSpace` class

```python
class ProjectionSpace(Space):
    """A Space whose transform reduces dimensionality.

    The transform stored in `Space.transform` is the pseudo-inverse (D_in x D_out+1),
    used when traversing *upward* out of this space (child → parent direction).
    The `projection_matrix` attribute (D_out+1 x D_in+1) is used when traversing
    *downward* into this space (parent → child direction).

    Args:
        projection_matrix: The (D_out+1) x (D_in+1) homogeneous matrix that maps
            FROM parent coordinates TO this space's lower-dimensional coordinates.
            For a 3D → 2D camera: a 3x4 matrix.
        parent: The higher-dimensional parent space.
    """
    def __init__(self, projection_matrix: np.ndarray, parent: Space):
        self.projection_matrix = projection_matrix
        pseudo_inv = np.linalg.pinv(projection_matrix)
        super().__init__(transform=pseudo_inv, parent=parent)
```

The pseudo-inverse stored as `self.transform` allows `compute_absolute_transform()`
to keep working by chaining matrices upward. The pseudo-inverse is the "best
least-squares inverse" of a non-square matrix.

**Note on reprojection**: `pseudo_inv @ projection_matrix ≈ I` (identity in the
higher-dimensional space), so chaining `pseudo_inv` for the backward traversal
approximates the back-projection ray's foot. For exact back-projection (ray, or depth-
parameterised) callers should use the dedicated methods on `ProjectionSpace` (see
section 5 below).

---

### 4. `space.py` — `get_root` / `compute_absolute_transform` (no changes needed)

These already work by chaining `node.transform` upward. `ProjectionSpace.transform`
is the pseudo-inverse; the chain still produces a matrix at each level. No changes.

---

### 5. `space.py` — back-projection helpers on `ProjectionSpace`

Two optional methods on `ProjectionSpace` for callers that need the inverse direction:

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

### 6. New `Ray` data class (optional, new file or in `coordinate.py`)

```python
@dataclass
class Ray:
    """A parametric ray: origin + t * direction in a given Space."""
    origin: Point
    direction: Vector
```

Used by `ProjectionSpace.ray_through()`. Can be deferred to a later PR if not needed
immediately.

---

### 7. `__init__.py` — public re-exports

Add to `src/coordinatus/__init__.py`:
```python
from .space import ProjectionSpace
```

If `Ray` is added:
```python
from .coordinate import Ray
```

---

### 8. `transforms/dimension.py` — no changes needed

The existing `project_xyz_to_xy()` etc. already produce the correct homogeneous matrices
and are used directly as the `projection_matrix` argument to `ProjectionSpace`.

---

## Test plan

All new tests go in the existing test files following the existing conventions.

### `tests/coordinatus/test_space.py`

**`TestFindLCA`**
- LCA of two siblings with a common direct parent
- LCA when one node is the ancestor of the other
- LCA of a node with itself
- Raises `ValueError` for unrelated trees

**`TestPathFrom`**
- Direct child path (one step)
- Deep path (multiple steps)
- Path to self is empty list

**`TestInvertStep`**
- Square transform → returns matrix inverse
- `ProjectionSpace` → returns `projection_matrix` (not pseudo-inverse)
- Non-square, non-`ProjectionSpace` → raises `ValueError`

**`TestComputeRelativeTransformToWithProjection`** (replaces implicit coverage)
- `space_a.compute_relative_transform_to(space_b)` for two siblings: same result as before (no regression)
- `3d_world.compute_relative_transform_to(projection_space)` → applies the projection matrix
- `projection_space.compute_relative_transform_to(3d_world)` → applies pseudo-inverse

**`TestProjectionSpace`**
- `ProjectionSpace(proj_3x4, parent=camera)` stores `projection_matrix` correctly
- `D_in` == 3, `D_out` == 2
- `compute_absolute_transform()` gives the correct chained matrix

### `tests/coordinatus/test_coordinate.py`

**`TestCoordinateRelativeToProjection`**
- `Point([x,y,z], space=world).relative_to(screen)` → 2D Point
- Result values match direct matrix multiplication
- `Point` in 2D screen → `relative_to(world)` → 3D Point via pseudo-inverse (add comment that this is least-squares, not exact)

**`TestCoordinateToAbsoluteProjection`**  
(already partially covered by the `test_projection_3d_to_2d_output_shape_via_to_absolute` test
added in the bug-fix step; extend with a full scene-graph scenario)

### `tests/coordinatus/test_space.py` — regression guard

Run existing `TestComputeRelativeTransformTo` tests unchanged; they must all still pass.

---

## Implementation order

1. Add `_find_lca`, `_path_from`, `_invert_step` helpers (private) to `space.py` — with unit tests
2. Replace `compute_relative_transform_to` body with LCA traversal — verify all existing tests still pass
3. Add `ProjectionSpace` class to `space.py` — with unit tests
4. Add `ProjectionSpace` to `__init__.py` public API
5. (Optional) Add `unproject_with_depth` and `Ray` / `ray_through`

---

## Backwards compatibility

- All existing `Space`, `Coordinate`, `Point`, `Vector` usage is unchanged.
- `compute_relative_transform_to` for two same-dimensional spaces produces the same
  result as before (the LCA path reduces to the current `inv(absolute(target)) @ absolute(self)` 
  formula).
- The `transform_coordinate` fix (already committed) is a no-op for square matrices.
