from copy import deepcopy

import numpy as np
import shapely
from shapely import Polygon
from shapely.affinity import rotate, translate


def calculate_transformed_outline(polygon: Polygon, rotation_angle: int):
    _polygon = deepcopy(polygon)
    _polygon = rotate(_polygon, rotation_angle, origin=(0, 0))
    centroid = _polygon.centroid

    return translate(_polygon, -centroid.x, -centroid.y)


def polygon_to_array(polygon):
    return np.array(polygon.exterior.coords, dtype=np.float32)


class PanelPolygon:
    def __init__(self, coords: tuple[float, float]) -> None:
        self.coords = np.float32(coords)
        self._outline = self.make_outline(coords)

    def make_outline(self, coords: tuple[float, float]) -> None:
        outline = shapely.geometry.Polygon(coords)
        # print(f"Valid: {outline.is_valid}")
        return outline

    @property
    def centroid(self) -> tuple[float, float]:
        return self._outline.centroid.x, self._outline.centroid.y

    def get_array_observation(self):
        # return a list of coordinates of the polygon
        return list(self._outline.exterior.coords)
