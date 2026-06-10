import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import numpy as np

from nester.env.panel.panel_polygon import PanelPolygon, polygon_to_array


class Panel:
    def __init__(
        self,
        coords_list: list[tuple[float, float]],
        bounds: tuple[float, float],
    ) -> None:
        self.coords_list = coords_list
        self.bounds = bounds
        self.polygons: dict[int, PanelPolygon] = {}

        self._init_polygons()
        self.total_area = self.get_total_area()

    def _init_polygons(self):
        for index, coords in enumerate(self.coords_list):
            self.polygons[index] = PanelPolygon(coords)

    def compute_border(self):
        # min,xy, max_xy
        bounds = np.array([p._outline.bounds for p in self.polygons.values()])
        min_xy = bounds.min(axis=0)[:2]
        min_xy = np.minimum(min_xy, np.array([0, 0]))
        max_xy = bounds.max(axis=0)[2:]
        max_xy = np.maximum(max_xy, np.array([0, 0]))
        return min_xy, max_xy

    def get_observation(self):
        return {
            index: panel_polygon.get_array_observation()
            for index, panel_polygon in self.polygons.items()
        }

    def get_occupancy_ratio(self):
        # Overlap-agnostic occupancy for improvement setting.
        # We use total polygon area over axis-aligned bounding rectangle area.
        border = self.compute_border()
        border_area = np.prod(border[1] - border[0])
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
