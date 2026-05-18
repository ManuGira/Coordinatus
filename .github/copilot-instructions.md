# Coordinatus — Copilot Instructions

> **Before making any architectural decision, read the [Project Architecture](#project-architecture) section below.**
> It contains the authoritative map of the project's modules, types, design rules,
> and invariants. Do not propose or implement structural changes without consulting it first.

---

## Running Python Scripts with UV

This project uses **`uv`** to manage Python scripts instead of the traditional `python` command.

To run any Python script in this project, use:

```powershell
uv run myscript.py
```

**NOT:**
```powershell
python myscript.py
```

## Project Structure

Package source code is located in the `src/coordinatus` directory and examples in the `examples/` directory.
To run a script located in the `examples/` directory, use:

```powershell
uv run examples/example_script.py
```

All tests are located in the `tests/` directory. The structure of the tests mirrors that of the `src/` directory. Test file names are prefixed with `test_`. For example, the test for `src/coordinatus/transforms/rotate.py` would be located at `tests/coordinatus/transforms/test_rotate.py`.

## Continuous Integration and Testing

Tests are run using `uv`.

Run individual test files like so:
```powershell
uv run pytest tests/coordinatus/test_example.py
```

To run all tests at once the command `uv run pytest tests` works but it is preferred to use the `ci.ps1` script:
```powershell
./ci.ps1
```
It will run all tests, generate coverage reports, and perform linting checks.

The `ci.ps1` script accepts switches to skip individual steps:
```powershell
./ci.ps1 -SkipTests        # skip pytest
./ci.ps1 -SkipStyle        # skip ruff
./ci.ps1 -SkipTypes        # skip ty
./ci.ps1 -SkipNotebooks    # skip notebook execution
./ci.ps1 -SkipTests -SkipNotebooks  # combine as needed
```

For any automation or AI agent execution, always use the `uv run` command format.

## Executing Notebooks

Never try to start or restart a Jupyter kernel interactively — it blocks. Instead, execute notebooks non-interactively via `nbconvert`:

```powershell
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 --inplace notebooks/my_notebook.ipynb
```

Always set `--ExecutePreprocessor.timeout` to a reasonable value (e.g. 60 seconds) to avoid hanging forever.

## Python Typing Style

When using type hints, **prefer the built-in collection types** (`list`, `dict`, `tuple`, etc.) over importing from `typing` (e.g., avoid `from typing import List, Dict, Tuple`).
Use `list[str]`, `dict[str, int]`, etc., for type annotations.

## Writing Style

When denoting matrix or array dimensions, prefer the ASCII `x` over the Unicode multiplication sign: write `2x2` not `2×2`, `3x3` not `3×3`, etc.

---

## Project Architecture

### Philosophy

A coordinate is meaningless without its context. Every value in Coordinatus lives
inside a `Space`, which describes *how that space relates to its parent*. Coordinates
are never bare numbers — they are always bound to the space in which they are expressed.

Spaces form a **scene-graph hierarchy**. Converting a coordinate from one space to
another requires both spaces to share a common ancestor. If they do not, the conversion
is undefined and raises a `ValueError`.

### Package layout

```
src/coordinatus/
├── __init__.py          # Public re-exports (Space*, Coordinate, Point, Vector, …)
├── types.py             # CoordinateKind enum  (POINT | VECTOR)
├── space.py             # Space hierarchy and factory helpers
├── coordinate.py        # Coordinate base class + Point, Vector subclasses
├── visualization.py     # Optional matplotlib helpers (requires extras)
└── transforms/
    ├── __init__.py      # Convenience composites: ts1D, trs2D, trks2D
    ├── translate.py     # translate, translate1D, translate2D, translate3D
    ├── rotate.py        # rotate2D, rotate3Dx, rotate3Dy, rotate3Dz, rotate3D
    ├── scale.py         # scale, scale1D, scale2D, scale3D, shear2D
    └── dimension.py     # reduce_dim, augment_dim, swap_axes, project_* helpers
```

### Key types and their responsibilities

#### `types.py` — `CoordinateKind`

A two-value enum: `POINT` or `VECTOR`.

- **POINT** (weight = 1): position in space; affected by translation, rotation, and scaling.
- **VECTOR** (weight = 0): direction/displacement; affected by rotation and scaling only.

This distinction is applied by `transform_coordinate()` when building homogeneous coordinate vectors.

#### `space.py` — `Space` and its subclasses

`Space` wraps a single `(D+1)x(D+1)` homogeneous transform matrix and an optional reference to a parent `Space`.

| Class      | Identity matrix | Typical use                  |
|------------|-----------------|------------------------------|
| `Space1D`  | `eye(2)`        | 1D number lines              |
| `Space2D`  | `eye(3)`        | 2D plane root spaces         |
| `Space3D`  | `eye(4)`        | 3D world root spaces         |
| `Space4D`  | `eye(5)`        | 4D / relativistic uses       |
| `SpaceND`  | `eye(N+1)`      | Arbitrary dimension N        |
| `Space`    | explicit matrix | Child spaces with transforms |

`Space(transform=None)` raises `ValueError` — always provide an explicit matrix or use a typed subclass.

Key methods:

- `get_root()` — walks up the parent chain; used to check for common ancestors.
- `compute_absolute_transform()` — multiplies transforms from self up to the root.
- `compute_relative_transform_to(target)` — raises `ValueError` if the two spaces have different roots; otherwise returns the combined matrix to go from self to target.

`create_space(parent, tx, ty, angle_rad, sx, sy)` is a 2D factory that builds the transform via `trs2D` and returns a `Space`.

#### `coordinate.py` — `Coordinate`, `Point`, `Vector`

`Coordinate` stores:
- `kind`: `CoordinateKind`
- `coords`: `np.ndarray` (shape `(D,)` for a single point, `(D, N)` for a batch)
- `space`: the `Space` in which `coords` are expressed

If `space=None`, an identity space of the appropriate dimension is created automatically.

`Point` and `Vector` are thin subclasses that fix `kind` and omit the `kind` argument from their constructors.

Key methods:
- `to_absolute()` — applies `compute_absolute_transform()` and returns coords in the root's identity space.
- `relative_to(target_space)` — calls `compute_relative_transform_to` on the spaces; raises `ValueError` if no common ancestor exists.

`transform_coordinate(transform, coordinates, kind)` is the low-level function; it converts to homogeneous coordinates, applies the matrix, and converts back.

Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`) operate element-wise and require both operands to be in the same space (checked via `Space.__eq__`).

#### `transforms/` — matrix factories

All functions return plain `np.ndarray` homogeneous matrices. They never create or reference a `Space`.

| Function / module    | What it produces                         |
|----------------------|------------------------------------------|
| `translate`          | Generic N-D translation                  |
| `translate1D/2D/3D`  | Shorthand for 1/2/3 dimensions           |
| `rotate2D`           | Counter-clockwise rotation in the plane  |
| `rotate3Dx/y/z`      | Rotation around a 3D axis                |
| `rotate3D`           | Rotation around an arbitrary 3D axis     |
| `scale`              | Generic N-D scaling                      |
| `scale1D/2D/3D`      | Shorthand for 1/2/3 dimensions           |
| `shear2D`            | 2D shear (kx, ky)                        |
| `swap_axes`          | Axis permutation                         |
| `reduce_dim`         | Drop a dimension from homogeneous coords |
| `augment_dim`        | Add a dimension to homogeneous coords    |
| `project_*`          | Various orthographic projections         |

Composite helpers (in `transforms/__init__.py`):

| Helper    | Composition     | Parameters                               |
|-----------|-----------------|------------------------------------------|
| `ts1D`    | `T @ S`         | `tx`, `sx`                               |
| `trs2D`   | `T @ R @ S`     | `tx`, `ty`, `angle_rad`, `sx`, `sy`      |
| `trks2D`  | `T @ R @ K @ S` | adds `kx`, `ky` shear on top of `trs2D`  |

#### `visualization.py` — optional matplotlib helpers

Only importable when `matplotlib` is installed (extras: `coordinatus[plotting]`).

- `draw_space_axes(ax, space, reference_space, …)` — draws the origin and axis arrows of `space` as seen from `reference_space`. When `reference_space=None`, coordinates are expressed in absolute space via `to_absolute()`.
- `draw_points(ax, points, reference_space, …)` — plots a list of `Point` objects with optional connecting lines and labels. Same fallback to `to_absolute()`.

### Dimensionality rules

| Matrix size   | Affine dimension | Corresponding class |
|---------------|------------------|---------------------|
| 2x2           | 1D               | `Space1D`           |
| 3x3           | 2D               | `Space2D`           |
| 4x4           | 3D               | `Space3D`           |
| 5x5           | 4D               | `Space4D`           |
| (N+1)x(N+1)   | ND               | `SpaceND(N)`        |

`Space.D_in` = number of input dimensions = `transform.shape[1] - 1`.
`Space.D_out` = number of output dimensions = `transform.shape[0] - 1`.
These differ only for dimension-changing projections.

### Common ancestor rule

`relative_to()` and `compute_relative_transform_to()` compare roots with `is` (identity
check, not equality). Two independent root spaces — even if both are `Space2D()` — are
considered unrelated and conversion raises:

```
ValueError: Cannot convert between unrelated coordinate spaces: the two spaces
do not share a common ancestor. ...
```

The fix is always to connect both hierarchies under a single shared root.

