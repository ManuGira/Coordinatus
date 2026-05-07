# Running Python Scripts with UV

This project uses **`uv`** to manage Python scripts instead of the traditional `python` command.

## Usage

To run any Python script in this project, use:

```powershell
uv run myscript.py
```

**NOT:**
```powershell
python myscript.py
```

## Project Architecture

The Coordinatus package is designed to manage coordinate systems, transformations, and coordinate representations. Here's the core architecture:

### Core Concepts

1. **Coordinates**: Represented as points or vectors in a coordinate space
   - **Points**: Positions in space affected by all transformations (translation, rotation, scaling)
   - **Vectors**: Directions and magnitudes affected only by rotation and scaling (NOT translation)
   - This distinction is captured in the `CoordinateKind` enum in `types.py`

2. **Affine Transformations**: All transformations use homogeneous coordinate matrices
   - 3x3 matrices for 2D transformations
   - 4x4 matrices for 3D transformations
   - Transformations are composed via matrix multiplication

3. **Coordinate Spaces**: Hierarchical spaces that can be nested
   - Each `Space` has a transformation relative to its parent
   - Spaces form a scene-graph-like hierarchy
   - Can compute absolute transformations from any space to the root

### Main Modules

- **`types.py`**: Defines `CoordinateKind` enum (POINT vs VECTOR)
- **`coordinate.py`**: Coordinate classes and the `transform_coordinate()` function that applies transformations while respecting point/vector semantics
- **`space.py`**: `Space` class representing hierarchical coordinate spaces
- **`transforms/`**: Transformation matrix generation utilities:
  - `translate.py`: Translation matrices (2D and 3D)
  - `rotate.py`: Rotation matrices (rotate2D, rotate3Dx, rotate3Dy, rotate3Dz, rotate3D)
  - `scale.py`: Scaling and shearing matrices
  - `dimension.py`: Dimension reduction/augmentation and projection operations
  - Convenience functions like `trs2D()` and `trks2D()` combine multiple transformations
- **`visualization.py`**: Optional visualization support (only available if matplotlib is installed)

### Design Patterns

1. **Homogeneous Coordinates**: All transformations use homogeneous coordinate matrices for uniform handling of affine transformations
2. **Matrix Composition**: Transformations are composed via matrix multiplication (e.g., `T @ R @ S`)
3. **Semantic Type Distinction**: Points and vectors transform differently - this is enforced throughout the codebase
4. **Lazy Transformation**: Spaces store relative transforms; absolute transforms are computed on-demand

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


For any automation or AI agent execution, always use the `uv run` command format.

## Python Typing Style

When using type hints, **prefer the built-in collection types** (`list`, `dict`, `tuple`, etc.) over importing from `typing` (e.g., avoid `from typing import List, Dict, Tuple`).
Use `list[str]`, `dict[str, int]`, etc., for type annotations.

## Writing Style

When denoting matrix or array dimensions, prefer the ASCII `x` over the Unicode multiplication sign: write `2x2` not `2×2`, `3x3` not `3×3`, etc.

