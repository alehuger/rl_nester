from time import perf_counter

import numpy as np
import CGALPY as CGAL

from nester.env.panel.panel_polygon import calculate_transformed_outline


def read_polygon(coords):
    pgn = CGAL.Pol2.Polygon_2()
    for c in coords:
        x = CGAL.Ker.FT(c[0])
        y = CGAL.Ker.FT(c[1])
        p = CGAL.Ker.Point_2(x, y)
        pgn.push_back(p)
    return pgn


def cgal_vertice_to_float(vertice):
    x = vertice.x().to_double()
    y = vertice.y().to_double()
    return [x, y]


def cgal_polygon_to_dict_outer_holes(nfp):
    outer_vertices = nfp.outer_boundary().vertices()
    outer_vertices = [cgal_vertice_to_float(vertice) for vertice in outer_vertices]
    outer_vertices = np.float32(outer_vertices)

    list_holes = []

    if nfp.has_holes():
        holes_generator = nfp.holes()

        for i in range(nfp.number_of_holes()):
            hole = holes_generator.__next__()
            hole_vertices = hole.vertices()
            hole_vertices = [
                cgal_vertice_to_float(vertice) for vertice in hole_vertices
            ]
            hole_vertices = np.float32(hole_vertices)
            list_holes.append(hole_vertices)

    dict_for_shapely = {"shell": outer_vertices, "holes": list_holes}

    return dict_for_shapely


def get_cgal_poly(polygon0, polygon1, rotation_i, rotation_j):
    stationary_polygon = calculate_transformed_outline(polygon0, rotation_i).simplify(0)
    moving_polygon = calculate_transformed_outline(polygon1, rotation_j).simplify(0)

    stationary_coords = np.array(stationary_polygon.exterior.coords, dtype=np.float32)
    moving_coords = np.array(moving_polygon.exterior.coords, dtype=np.float32)

    p = read_polygon(stationary_coords[:-1])
    q = read_polygon(-moving_coords[:-1])
    return p, q


def _run_c_binding(polygon0, polygon1, rotation_i, rotation_j):
    t0 = perf_counter()
    p, q = get_cgal_poly(polygon0, polygon1, rotation_i, rotation_j)

    if q.is_clockwise_oriented():
        q.reverse_orientation()
    if p.is_clockwise_oriented():
        p.reverse_orientation()
    nfp = c__fonc(p, q)

    dict_for_shapely = cgal_polygon_to_dict_outer_holes(nfp)

    duration = perf_counter() - t0
    return dict_for_shapely, duration


def c__fonc(p, q):
    nfp = CGAL.Ms2.minkowski_sum_2(p, q)
    return nfp


if __name__ == "__main__":
    from shapely import Polygon

    polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.3, 0.3]])
    nfp = _run_c_binding(polygon, polygon, 0, 0)
    print(nfp)
