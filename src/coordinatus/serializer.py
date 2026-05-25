import numpy as np

from .space import Space
from .coordinate import Coordinate
from .coordinatus_types import CoordinateKind


def to_json(spaces: list[Space], coordinates: list[Coordinate]) -> dict:
    space_dicts = []
    for space in spaces:
        parent_uid = space.parent.uid if space.parent else ""
        space_dict = {
            "uid": space.uid,
            "parent_uid": parent_uid,
            "transform": space.transform.tolist()  # Assuming transform is a numpy array
        }
        space_dicts.append(space_dict)

    coordinate_dicts = []
    for coordinate in coordinates:
        kind: CoordinateKind = coordinate.kind
        coordinate_dict = {
            "kind": kind.value,   # Assuming kind is an enum
            "coords": coordinate.coords.tolist(),  # Assuming coords is a numpy array
            "space_uid": coordinate.space.uid,
        }
        coordinate_dicts.append(coordinate_dict)

    return {
        "spaces": space_dicts,
        "points": coordinate_dicts
    }

def from_json(data: dict) -> tuple[list[Space], list[Coordinate]]:
    uid_to_spaces = {}
    for space_dict in data["spaces"]:
        uid = space_dict["uid"]
        uid_to_spaces[uid] = Space(
            uid=uid,
            transform=np.array(space_dict["transform"])
        )

    for space_dict in data["spaces"]:
        uid = space_dict["uid"]
        parent_uid = space_dict["parent_uid"]
        if len(parent_uid) > 0:
            uid_to_spaces[uid].parent = uid_to_spaces[parent_uid]

    spaces = list(uid_to_spaces.values())

    coordinates = []
    for coordinate_dict in data["points"]:
        kind = CoordinateKind(coordinate_dict["kind"])
        coordinate = Coordinate(
            kind=kind,
            coords=np.array(coordinate_dict["coords"]),
            space=uid_to_spaces[coordinate_dict["space_uid"]]
        )
        coordinates.append(coordinate)

    return spaces, coordinates