"""Demo: converting between spaces with no common ancestor raises an error.

Two separately defined coordinate hierarchies have no shared root, so there is
no mathematical basis for converting coordinates between them. Coordinatus
enforces this explicitly instead of silently producing meaningless results.
"""

from coordinatus import Space2D, Point
from coordinatus.transforms import translate2D

# ── Two independent 2D worlds ──────────────────────────────────────────────────
# Imagine two separate floor plans: Building A and Building B.
# They have never been georeferenced together, so we cannot meaningfully
# express a position in one in terms of the other.

building_a = Space2D()  # root of Building A's coordinate hierarchy
building_b = Space2D()  # root of Building B's coordinate hierarchy — unrelated!

# Rooms defined inside each building
room_a1 = Space2D(parent=building_a)
room_b1 = Space2D(parent=building_b)

# A point inside room A1
desk = Point([3.0, 2.0], space=room_a1)

print("=== Conversion within the same hierarchy ===")
# Converting desk from room_a1 to building_a works: they share the same root.
desk_in_building_a = desk.relative_to(building_a)
print(f"Desk in building_a coords: {desk_in_building_a.coords}")  # [3. 2.]

print()
print("=== Conversion between unrelated hierarchies ===")
# Attempting to convert the desk position into Building B's coordinate
# system must fail — the two hierarchies share no common ancestor.
try:
    desk_in_building_b = desk.relative_to(building_b)
    print("ERROR: should have raised an exception!")
except ValueError as e:
    print(f"Caught expected error:\n  {e}")

print()
print("=== Fix: establish a shared root ===")
# If you do have a common reference (e.g. a city grid), put both buildings
# under the same root space.
city_grid = Space2D()
building_c = Space2D(parent=city_grid)
building_d_transform = translate2D(100.0, 50.0)  # Building D is 100 m east, 50 m north
building_d = Space2D(parent=city_grid)

room_c1 = Space2D(parent=building_c)
front_door = Point([1.0, 0.0], space=room_c1)

# Now the conversion works because both buildings are in city_grid's hierarchy.
front_door_in_d = front_door.relative_to(building_d)
print(f"Front door expressed in building_d coords: {front_door_in_d.coords}")
