import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import numpy as np
from shapely.affinity import translate

from nester.env.panel.metrics import (
    DEFAULT_OCCUPANCY_METRIC,
    min_area_rect,
    validate_occupancy_metric,
)
from nester.env.panel.panel_polygon import PanelPolygon, polygon_to_array


class Panel:
    def __init__(
        self,
        coords_list: list[tuple[float, float]],
        bounds: tuple[float, float],
        occupancy_metric: str = DEFAULT_OCCUPANCY_METRIC,
    ) -> None:
        self.coords_list = coords_list
        self.bounds = bounds
        self.occupancy_metric = validate_occupancy_metric(occupancy_metric)
        self.polygons: dict[int, PanelPolygon] = {}

        self._init_polygons()
        self.total_area = self.get_total_area()

    def _init_polygons(self):
        for index, coords in enumerate(self.coords_list):
            self.polygons[index] = PanelPolygon(coords)

    def true_aabb(self):
        """Axis-aligned box of the placed polygons, not forced to contain the origin."""
        if not self.polygons:
            z = np.zeros(2, dtype=np.float64)
            return z, z.copy()
        bounds = np.array([p._outline.bounds for p in self.polygons.values()])
        min_xy = bounds.min(axis=0)[:2]
        max_xy = bounds.max(axis=0)[2:]
        return min_xy, max_xy

    def compute_border(self):
        # min,xy, max_xy — legacy AABB occupancy forces the box to contain the origin
        min_xy, max_xy = self.true_aabb()
        min_xy = np.minimum(min_xy, np.array([0, 0]))
        max_xy = np.maximum(max_xy, np.array([0, 0]))
        return min_xy, max_xy

    def recenter_aabb(self) -> np.ndarray:
        """Translate so the true AABB centre sits at the origin. Returns the shift applied."""
        min_xy, max_xy = self.true_aabb()
        center = (min_xy + max_xy) / 2.0
        if not np.any(np.abs(center) > 1e-12):
            return np.zeros(2, dtype=np.float64)
        for panel_polygon in self.polygons.values():
            panel_polygon._outline = translate(
                panel_polygon._outline,
                xoff=-float(center[0]),
                yoff=-float(center[1]),
            )
        return np.asarray(center, dtype=np.float64)

    def get_observation(self):
        return {
            index: panel_polygon.get_array_observation()
            for index, panel_polygon in self.polygons.items()
        }

    def all_vertices(self) -> np.ndarray:
        arrays = [
            np.asarray(p._outline.exterior.coords, dtype=np.float64)[:-1]
            for p in self.polygons.values()
        ]
        if not arrays:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(arrays, axis=0)

    def compute_min_rect(self) -> dict:
        """Minimum-area enclosing rectangle over all orientations (see metrics.min_area_rect)."""
        return min_area_rect(self.all_vertices())

    def get_bounding_area(self) -> float:
        if self.occupancy_metric == "min_rect":
            return float(self.compute_min_rect()["area"])
        border = self.compute_border()
        return float(np.prod(border[1] - border[0]))

    def get_occupancy_ratio(self):
        # Overlap-agnostic occupancy for improvement setting: total polygon area over
        # the bounding-rectangle area. ``aabb`` (legacy) uses the world-axis box that
        # contains the origin; ``min_rect`` uses the oriented minimum-area rectangle,
        # which is invariant to global translation and rotation of the layout.
        border_area = self.get_bounding_area()
        if border_area <= 0:
            return 0.0
        return self.total_area / border_area

    def get_total_area(self):
        outlines = [panel_polygon._outline for panel_polygon in self.polygons.values()]
        total_area = sum([outline.area for outline in outlines])
        return total_area

    def render_centered(self):
        sns.set_theme()
        colors = sns.color_palette()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal", adjustable="box")
        exterior_lim = 1.2
        min_xy, max_xy = self.compute_border()
        center_xy = (min_xy + max_xy) / 2
        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]

        ax.set_xlim(
            center_xy[0] - width / 2 * exterior_lim,
            center_xy[0] + width / 2 * exterior_lim,
        )
        ax.set_ylim(
            center_xy[1] - height / 2 * exterior_lim,
            center_xy[1] + height / 2 * exterior_lim,
        )

        for i, panel_polygon in self.polygons.items():
            x, y = panel_polygon.centroid
            outline = panel_polygon._outline
            color_ex = colors[i]
            array = polygon_to_array(outline)
            ax.fill(*np.swapaxes(array, 0, 1), color=color_ex)
            ax.plot(*np.swapaxes(array, 0, 1), color="gray")
            ax.text(x, y, str(i), size=8)

        if self.occupancy_metric == "min_rect":
            rect = self.compute_min_rect()
            ax.add_patch(
                patches.Polygon(
                    rect["corners"],
                    closed=True,
                    linewidth=1,
                    edgecolor="firebrick",
                    facecolor="none",
                    label="border",
                ),
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    center_xy - np.array([width / 2, height / 2]),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="firebrick",
                    facecolor="none",
                    label="border",
                ),
            )

        return fig
