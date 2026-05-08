- [Coordinatus](#coordinatus)
  - [Why Coordinatus?](#why-coordinatus)
  - [Installation with uv](#installation-with-uv)
    - [Installing the Development Version](#installing-the-development-version)
    - [Optional: Visualization Support](#optional-visualization-support)
  - [Quick Start](#quick-start)
  - [Core Concepts](#core-concepts)
    - [Spaces](#spaces)
    - [Points vs Vectors](#points-vs-vectors)
    - [Coordinate Conversion](#coordinate-conversion)
  - [The Relativity of Coordinates](#the-relativity-of-coordinates)
    - [View from Space 1](#view-from-space-1)
    - [View from Absolute Space](#view-from-absolute-space)
    - [View from Space 2](#view-from-space-2)
  - [API Overview](#api-overview)
    - [Creating Spaces](#creating-spaces)
    - [Transformation Utilities](#transformation-utilities)
    - [Visualization (Optional)](#visualization-optional)
  - [Examples](#examples)
  - [Testing](#testing)
  - [License](#license)



# Coordinatus

**Simple coordinate transformations with hierarchical spaces**

Ever needed to convert coordinates between different spaces?  *Coordinatus* makes it easy to work with nested coordinate systems—like transforming from a character's local space to world space, or from one object to another.

**The core philosophy:** a bare `[x, y]` array is meaningless without knowing how its origin and axes are defined. In *Coordinatus*, every coordinate carries its space with it. This means that when you pass a `Point` or `Vector` to a function, you never need to pre-convert it—as long as both the coordinate's space and the function's expected space belong to the same hierarchy, the conversion is handled automatically.

For example, converting a temperature across three unit systems is just a matter of defining the spaces:

```python
import numpy as np
from coordinatus import Space1D, Point
from coordinatus.transforms import ts1D

# Kelvin is the 1D world space (absolute)
kelvin = Space1D()

# ts1D(tx, sx) maps x → sx·x + tx
celsius    = Space(transform=ts1D(tx=273.15, sx=1),   parent=kelvin)   # C = K - 273.15
fahrenheit = Space(transform=ts1D(tx=32,     sx=9/5), parent=celsius)  # F = C * 9/5 + 32

boiling = Point([100.0], space=celsius)  # 100 °C
in_fahrenheit = boiling.relative_to(fahrenheit)   # no manual formula needed

print(f"{boiling.coords[0]:.1f} °C = {in_fahrenheit.coords[0]:.1f} °F")
# 100.0 °C = 212.0 °F
```

## Why Coordinatus?

- **Intuitive API**: Work with Points and Vectors that transform correctly (vectors ignore translation!)
- **Hierarchical Spaces**: Build parent-child relationships just like scene graphs in game engines
- **Clean transformations**: Simple functions for translation, rotation, and scaling
- **Type-safe**: Points and Vectors are distinct types with correct transformation behavior

## Installation with uv
Modern python package manager [uv](https://docs.astral.sh/uv/) is recommended for managing dependencies, but pip can also be used (simply replace `uv add` with `pip install` in example below).

```bash
uv add coordinatus
```

### Installing the Development Version

To install the latest development version from the `develop` branch:

**Using uv:**
```bash
uv add git+https://github.com/ManuGira/Coordinatus.git@develop
```

### Optional: Visualization Support

For plotting and visualization features (used in examples):

**Using uv:**
```bash
uv add coordinatus[plotting]
```

This installs matplotlib for the `coordinatus.visualization` module.

## Quick Start

```python
from coordinatus import Space2D, Point, create_space
import numpy as np

# Create a world space
world = Space2D()

# Create a car space, positioned at (100, 50) in the world
car = create_space(parent=world, tx=100, ty=50, angle_rad=np.pi/4)

# Create a wheel space, offset (10, 0) from the car
wheel = create_space(parent=car, tx=10, ty=0)

# A point at the wheel's center
point_in_wheel = Point([0, 0], space=wheel)

# Convert to world coordinates
point_in_world = point_in_wheel.to_absolute()
print(f"Wheel center in world: ({point_in_world.coords[0]}, {point_in_world.coords[1]})")

# Convert between any two spaces
point_in_car = point_in_wheel.relative_to(car)
print(f"Wheel center in car space: ({point_in_car.coords[0]}, {point_in_car.coords[1]})")
```

## Core Concepts

### Spaces
A `Space` represents a coordinate system with its own position, rotation, and scale. Spaces can be nested to create hierarchies.

### Points vs Vectors
- **Points** represent positions and are affected by translation
- **Vectors** represent directions/offsets and ignore translation

```python
from coordinatus import Point, Vector, Space, create_space

space = create_space(parent=None, tx=10, ty=5)

# Point gets translated
point = Point([0, 0], space=space)
absolute = point.to_absolute()  # coords: [10, 5]

# Vector does NOT get translated
vector = Vector([1, 0], space=space)
absolute_vec = vector.to_absolute()  # coords: [1, 0] - only rotation/scale applied
```

### Coordinate Conversion

Convert between any two spaces in your hierarchy:

```python
# Convert from space_a to space_b
point_in_a = Point(np.array([5, 3]), space=space_a)
point_in_b = point_in_a.relative_to(space_b)

# Or get absolute (world) coordinates
point_in_world = point_in_a.to_absolute()
```

## The Relativity of Coordinates

A fundamental concept in coordinate transformations is that **the same geometry looks different depending on your point of view**. The same F-shaped object can appear rotated, scaled, or sheared simply by changing which reference space you're observing from.

Consider these three views of the same scene with an F-shaped object and two coordinate spaces:

### View from Space 1
![Space1 View](https://raw.githubusercontent.com/ManuGira/Coordinatus/45d97475cd735e7b580256e23fbc62d0ae5d6862/examples/space_visualization_Space1.png)

The F shape appears undistorted in its canonical form because it was defined using Space 1 coordinates. From this perspective, Space 1's axes are the standard orthogonal x and y axes at the origin. Space 2 (green) appears in a different position and orientation relative to Space 1.

### View from Absolute Space
![Absolute View](https://raw.githubusercontent.com/ManuGira/Coordinatus/45d97475cd735e7b580256e23fbc62d0ae5d6862/examples/space_visualization_Absolute.png)

In absolute (world) space, we see how the F shape actually looks in reality. Space 1 (blue) is sheared and the F inherits this shearing. Space 2 (green) is rotated and scaled. This reveals the true geometric relationships between all elements.

### View from Space 2  
![Space2 View](https://raw.githubusercontent.com/ManuGira/Coordinatus/45d97475cd735e7b580256e23fbc62d0ae5d6862/examples/space_visualization_Space2.png)

From Space 2's perspective, Space 2 is now at the origin with standard axes. The same F shape appears with a completely different orientation and distortion, even though the geometry itself hasn't changed—only our reference space has.

**Key insight:** Coordinates are not absolute—they depend on the observer. The F shape's numerical coordinates change in each view, but the shape's position in physical space remains constant. This is the essence of relative coordinate systems.


## API Overview

### Creating Spaces

```python
from coordinatus import Space, Space2D, create_space
import numpy as np

# Manually with a transform matrix
space = Space(transform=my_matrix, parent=parent_space)

# Or start from a typed identity root
root = Space2D()

# Or use the convenient factory
space = create_space(
    parent=parent_space,
    tx=10, ty=5,           # Translation
    angle_rad=np.pi/4,     # Rotation
    sx=2, sy=2             # Scale
)
```

### Transformation Utilities

```python
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D

# Individual transformations (2D)
t = translate2D(tx=10, ty=5)
r = rotate2D(angle_rad=np.pi/2)
s = scale2D(sx=2, sy=3)

# Combined TRS (Translation-Rotation-Scale)
transform = trs2D(tx=10, ty=5, angle_rad=np.pi/4, sx=2, sy=2)
```

### Visualization (Optional)

```python
from coordinatus.visualization import draw_space_axes, draw_points
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots()

# Draw spaces and points
draw_space_axes(ax, space1, color='blue', label='Space1')
draw_space_axes(ax, space2, color='green', label='Space2')
draw_points(ax, [point1, point2], color='red')

plt.show()
```

**Note:** Requires `uv add coordinatus[plotting]`

## Notebooks

The [`notebooks/`](notebooks/) folder contains interactive Jupyter notebooks with embedded outputs — plots render directly on GitHub:

- [01_getting_started.ipynb](notebooks/01_getting_started.ipynb) — Points vs Vectors, hierarchical spaces, temperature unit conversion
- [02_nested_spaces.ipynb](notebooks/02_nested_spaces.ipynb) — Accumulating TRS transforms through nested spaces
- [03_space_visualization.ipynb](notebooks/03_space_visualization.ipynb) — The same geometry seen from different reference frames
- [04_rotating_frame.ipynb](notebooks/04_rotating_frame.ipynb) — Animated GIF: a fixed point as seen from a rotating reference frame

To run the notebooks locally, install the dev dependencies and launch Jupyter:

```bash
uv sync --group dev
uv run jupyter notebook
```

## Examples

Check out the [`examples/`](examples/) folder for complete, runnable examples:
- [nested_spaces.py](examples/nested_spaces.py)
- [space_visualization.py](examples/space_visualization.py)

## Testing

```bash
uv run pytest tests
```

## License

MIT