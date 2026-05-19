---
type: bmad-distillate
sources:
  - "copilot-instructions.md"
downstream_consumer: "general"
created: "2026-05-19"
token_estimate: 870
parts: 1
---

## Tooling and Commands
- Git: run `git` in terminal; do NOT use GitKraken MCP tools
- Python runner: `uv run myscript.py` (NOT `python myscript.py`); with args: `uv run python myscript.py --help`; always use `uv run` for automation/AI agent execution
- Examples: `uv run examples/example_script.py`
- Single test: `uv run pytest tests/coordinatus/test_example.py`
- All tests: `./ci.ps1` (preferred over `uv run pytest tests`); runs pytest, coverage, ruff linting
- ci.ps1 switches: `-SkipTests`, `-SkipStyle`, `-SkipTypes`, `-SkipNotebooks` (combinable)
- Notebooks: never start/restart kernel interactively (blocks); execute via `uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 --inplace notebooks/my_notebook.ipynb`; always set `--ExecutePreprocessor.timeout`

## Commit Messages
- Format: `<type>[optional scope][optional !]: <description>` + optional blank-line-separated body + optional footers
- `fix`: bug fix → PATCH; `feat`: new feature → MINOR; BREAKING CHANGE → MAJOR
- Breaking change: `!` before `:` (e.g. `feat!: ...`) OR `BREAKING CHANGE: <description>` footer; both MAY be used together
- Other types (no implicit SemVer effect): `build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`, `revert`
- Scope: optional noun in parentheses after type, e.g. `feat(parser): add array parsing`
- Footer format: `Token: value` or `Token #value`; token uses `-` for spaces (e.g. `Reviewed-by`); exception: `BREAKING CHANGE` (uppercase, spaces allowed); `BREAKING-CHANGE` is synonymous
- If commit fits multiple types, prefer splitting into multiple commits

## Code Style
- Type hints: built-in types only (`list[str]`, `dict[str, int]`, `tuple`) — no `from typing import List, Dict, Tuple`
- Dimensions: ASCII `x` (`2x2`, `3x3`) not Unicode `×`

## Project Structure
- Source: `src/coordinatus/`; examples: `examples/`; tests: `tests/` (mirrors src/, filenames prefixed `test_`)
- `__init__.py`: public re-exports (Space*, Coordinate, Point, Vector)
- `types.py`: CoordinateKind enum (POINT | VECTOR)
- `space.py`: Space hierarchy and factory helpers
- `coordinate.py`: Coordinate base class + Point, Vector subclasses
- `visualization.py`: optional matplotlib helpers (extras: `coordinatus[plotting]`)
- `transforms/__init__.py`: composites ts1D, trs2D, trks2D
- `transforms/translate.py`: translate, translate1D/2D/3D
- `transforms/rotate.py`: rotate2D, rotate3Dx/y/z, rotate3D
- `transforms/scale.py`: scale, scale1D/2D/3D, shear2D
- `transforms/dimension.py`: reduce_dim, augment_dim, swap_axes, project_*

## Architecture — Philosophy
- Every coordinate lives inside a Space; Space describes how it relates to its parent; coordinates never bare numbers
- Spaces form scene-graph hierarchy; conversion requires common ancestor; else → ValueError
- Read Project Architecture before any architectural decision; do not propose structural changes without consulting it

## Architecture — CoordinateKind
- POINT (weight=1): position; affected by translation, rotation, scaling
- VECTOR (weight=0): direction/displacement; rotation and scaling only
- Applied by `transform_coordinate()` when building homogeneous coordinate vectors

## Architecture — Space
- Wraps single (D+1)x(D+1) homogeneous transform matrix + optional parent Space reference
- Space1D: eye(2); Space2D: eye(3); Space3D: eye(4); Space4D: eye(5); SpaceND: eye(N+1); Space: explicit matrix (child with transform)
- `Space(transform=None)` → ValueError; always provide matrix or use typed subclass
- `get_root()`: walks parent chain; used for common ancestor check
- `compute_absolute_transform()`: multiplies transforms from self to root
- `compute_relative_transform_to(target)`: ValueError if different roots; else returns combined matrix
- `create_space(parent, tx, ty, angle_rad, sx, sy)`: 2D factory via trs2D → returns Space

## Architecture — Coordinate, Point, Vector
- Coordinate stores: kind (CoordinateKind), coords (np.ndarray: (D,) single or (D,N) batch), space (Space)
- space=None → identity space auto-created for appropriate dimension
- Point/Vector: thin subclasses fixing kind; omit kind arg from constructors
- `to_absolute()`: applies compute_absolute_transform(); returns coords in root's identity space
- `relative_to(target_space)`: calls compute_relative_transform_to on spaces; ValueError if no common ancestor
- `transform_coordinate(transform, coordinates, kind)`: low-level; converts to homogeneous, applies matrix, converts back
- Arithmetic (+, -, *, /, unary -): element-wise; both operands must be in same space (Space.__eq__)

## Architecture — Transforms
- All functions return plain np.ndarray homogeneous matrices; never create/reference Space
- translate/translate1D/2D/3D; rotate2D (CCW); rotate3Dx/y/z; rotate3D (arbitrary axis); scale/scale1D/2D/3D; shear2D (kx, ky); swap_axes; reduce_dim; augment_dim; project_*
- Composites: ts1D = T@S (tx, sx); trs2D = T@R@S (tx, ty, angle_rad, sx, sy); trks2D = T@R@K@S (adds kx, ky shear)

## Architecture — Visualization
- Only importable when matplotlib installed (extras: `coordinatus[plotting]`)
- `draw_space_axes(ax, space, reference_space, …)`: draws origin and axis arrows of space as seen from reference_space; reference_space=None → to_absolute()
- `draw_points(ax, points, reference_space, …)`: plots list of Point objects; optional connecting lines and labels; reference_space=None → to_absolute()

## Architecture — Dimensionality Rules
- 2x2→1D/Space1D; 3x3→2D/Space2D; 4x4→3D/Space3D; 5x5→4D/Space4D; (N+1)x(N+1)→ND/SpaceND(N)
- Space.D_in = transform.shape[1]-1; Space.D_out = transform.shape[0]-1; differ only for dimension-changing projections

## Architecture — Common Ancestor Rule
- relative_to() and compute_relative_transform_to() compare roots with `is` (identity check, not equality)
- Two independent Space2D() instances → unrelated → ValueError even if structurally identical
- Fix: connect both hierarchies under a single shared root
