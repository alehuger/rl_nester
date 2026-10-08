"""Occupancy metrics for a set of placed polygons.

Two bounding-box conventions are supported:

``aabb``
    Legacy metric. Axis-aligned bounding box in the world frame, forced to contain
    the origin (see :meth:`Panel.compute_border`). Depends on absolute position and
    on the global orientation of the layout.

``min_rect``
    Minimum-area enclosing rectangle over all orientations (rotating calipers on the
    convex hull of every vertex). Invariant to global translation and rotation of the
    layout, so the occupancy of a nest does not depend on where or how it sits in the
    world frame.
"""

from __future__ import annotations

import numpy as np
from shapely import MultiPoint

OCCUPANCY_METRICS = ("aabb", "min_rect")
DEFAULT_OCCUPANCY_METRIC = "aabb"


def validate_occupancy_metric(metric: str | None) -> str:
    m = DEFAULT_OCCUPANCY_METRIC if metric is None else str(metric).strip().lower()
    if m not in OCCUPANCY_METRICS:
        raise ValueError(
            f"Unsupported occupancy_metric {metric!r}; expected one of {OCCUPANCY_METRICS}"
        )
    return m


def convex_hull_points(points: np.ndarray) -> np.ndarray:
    """Convex hull vertices (N, 2) of a point cloud, counter-clockwise, no repeated closing point."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] == 0:
        return pts.reshape(0, 2)
    if pts.shape[0] <= 2:
        return pts
    hull = MultiPoint(pts).convex_hull
    if hull.geom_type == "Polygon":
        coords = np.asarray(hull.exterior.coords, dtype=np.float64)[:-1]
    elif hull.geom_type == "LineString":
        coords = np.asarray(hull.coords, dtype=np.float64)
    else:  # Point
        coords = np.asarray(hull.coords, dtype=np.float64)
    return coords


def min_area_rect(points: np.ndarray) -> dict:
    """Minimum-area enclosing rectangle of a point set (rotating calipers).

    Returns a dict with ``area``, ``width``, ``height`` (width >= height), ``angle``
    (radians, orientation of the rectangle's long side in the world frame) and
    ``corners`` (4, 2) in world coordinates. Degenerate inputs return zero area.
    """
    hull = convex_hull_points(points)
    n = hull.shape[0]
    empty = {
        "area": 0.0,
        "width": 0.0,
        "height": 0.0,
        "angle": 0.0,
        "corners": np.zeros((4, 2), dtype=np.float64),
    }
    if n == 0:
        return empty
    if n == 1:
        empty["corners"] = np.repeat(hull, 4, axis=0)
        return empty

    edges = np.roll(hull, -1, axis=0) - hull  # (n, 2)
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    if not np.any(keep):
        empty["corners"] = np.repeat(hull[:1], 4, axis=0)
        return empty
    u = edges[keep] / lengths[keep, None]  # unit edge directions (m, 2)
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)  # perpendiculars (m, 2)

    proj_u = hull @ u.T  # (n, m)
    proj_v = hull @ v.T  # (n, m)
    min_u, max_u = proj_u.min(axis=0), proj_u.max(axis=0)
    min_v, max_v = proj_v.min(axis=0), proj_v.max(axis=0)
    ext_u = max_u - min_u
    ext_v = max_v - min_v
    areas = ext_u * ext_v
    k = int(np.argmin(areas))

    uu, vv = u[k], v[k]
    c00 = uu * min_u[k] + vv * min_v[k]
    c10 = uu * max_u[k] + vv * min_v[k]
    c11 = uu * max_u[k] + vv * max_v[k]
    c01 = uu * min_u[k] + vv * max_v[k]
    corners = np.stack([c00, c10, c11, c01], axis=0)

    w, h = float(ext_u[k]), float(ext_v[k])
    if w >= h:
        angle = float(np.arctan2(uu[1], uu[0]))
    else:
        angle = float(np.arctan2(vv[1], vv[0]))
        w, h = h, w
    return {
        "area": float(areas[k]),
        "width": w,
        "height": h,
        "angle": angle,
        "corners": corners,
    }


def aabb_area_with_origin(points: np.ndarray) -> float:
    """Legacy panel border: axis-aligned box forced to contain the origin."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    min_xy = np.minimum(pts.min(axis=0), 0.0)
    max_xy = np.maximum(pts.max(axis=0), 0.0)
    return float(np.prod(max_xy - min_xy))


def bounding_area(points: np.ndarray, metric: str) -> float:
    metric = validate_occupancy_metric(metric)
    if metric == "min_rect":
        return min_area_rect(points)["area"]
    return aabb_area_with_origin(points)


def occupancy_from_coords(
    polygons: list[np.ndarray], total_area: float, metric: str
) -> float:
    """Occupancy ratio of a list of placed polygon outlines (each (V, 2))."""
    arrays = [np.asarray(p, dtype=np.float64) for p in polygons if p is not None]
    arrays = [a[~np.isnan(a).any(axis=1)] for a in arrays if a.ndim == 2 and a.shape[1] == 2]
    if not arrays:
        return 0.0
    pts = np.concatenate(arrays, axis=0)
    area = bounding_area(pts, metric)
    if area <= 0:
        return 0.0
    return float(total_area) / area
