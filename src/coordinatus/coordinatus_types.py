"""Coordinate type definitions."""

from enum import Enum


class CoordinateKind(Enum):
    """
    Defines whether a coordinate represents a point or a vector.
    1. POINT: Represents a position in space. Affected by translations, rotations, and scales.
    2. VECTOR: Represents a direction and magnitude. Affected by rotations and scales, BUT NOT TRANSLATIONS.
    """
    POINT = "point"
    VECTOR = "vector"
