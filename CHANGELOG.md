# Changelog

## [0.4.0.dev1]
### Added
- `Space`: TRS (Translation-Rotation-Scale) attributes `tx`, `ty`, `angle_rad`, `sx`, `sy` as first-class mutable properties
- `Space.transform`: now a computed read-only property derived from TRS attributes on demand

### Changed
- `Space.__init__`: signature changed from `(transform, parent)` to `(parent, tx, ty, angle_rad, sx, sy)` — spaces are now constructed via TRS parameters instead of a raw matrix
- `create_space()`: simplified to delegate directly to `Space()` constructor (same API)
- Decrease numpy version >=2.4.0 to >=2.3.5 to allow compatibility with environments where numpy 2.4.0 is not yet available.

## [0.3.0] - 2026-01-02
### Added
- `Coordinate`: support for ArrayLike inputs (lists, tuples) for `coords` argument in addition to numpy arrays
- `Coordinate`: support for DxN arrays inputs for `coords` argument where D is dimension and N is number of coordinates, allowing batch operations
- `Coordinate`: properties `D` and `N` to get dimension and number of coordinates
- `Frame`: properties `D_in` and `D_out` to get input and output dimensions of the frame's coordinate frame
- `Coordinate`: Support for arithemitc operators, the coordinate acting as a numpy array, `__array__`, `__getitem__`, `__setitem__`, `__len__`, `__repr__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`, `__rmul__`, `__truediv__`, `__rtruediv__`, `__neg__`, `__abs__`, `__eq__`, `__ne__`
- `transforms`: projection transforms (dimension-changing) in addition to standard affine transforms, `reduce_dim`, `augment_dim`, `project_xy_to_x`, `project_xy_to_y`, `project_xyz_to_xy`, `project_xyz_to_xz`, `project_xyz_to_yz`, `project_xyz_to_x`, `project_xyz_to_y`, `project_xyz_to_z`,
- `transforms`: common 3D transformation matrices: `translation3d`, `scaling3d`, `rotation3Dx`, `rotation3Dy`, `rotation3Dz`, 
- `transforms`: arbitrary dimenstion translation and scaling: `translation`, `scaling`


### Changed
- Renamed `CoordinateType` to `CoordinateKind`
- Coordinate:
  - rename `coordinate_type` to `kind`, to avoid confusion with Python types and numpy dtypes
  - Renamed `local_coords` to `coords`, because `point.to_absolute().local_coords` was confusing
- Frame:
  - `compute_absolute_transform` now returns identity transform for root frames instead of raising
  - `create_frame` `parent` argument can now be `None` for root frames

## [0.2.0] - 2025-12-31

### Added
- Rename package to Coordinatus
- Increase test coverage for visualization module
- Publish test coverage reports to Github Pages

## [0.1.0] - 2025-12-31

### Added
- Initial release
- 2D coordinate frame transformations
- Hierarchical frame support
- Point and Vector classes with proper transformation semantics
- Visualization module (optional, requires matplotlib)
- Comprehensive examples and documentation