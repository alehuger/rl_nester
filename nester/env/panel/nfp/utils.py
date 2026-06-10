from shapely.geometry import MultiPolygon, Polygon

# Stage-2 NFP component semantics (1-based ids; 0 is padding in the agent):
#   1 — exterior of the primary forbidden polygon
#   2 — largest hole (by enclosed area)
#   3 — second-largest hole
MAX_NFP_HOLES = 2
MAX_NFP_COMPONENTS = 1 + MAX_NFP_HOLES


def _primary_forbidden_polygon(forbidden_geom):
    if forbidden_geom is None or forbidden_geom.is_empty:
        return None
    if forbidden_geom.geom_type == "Polygon":
        return forbidden_geom
    if forbidden_geom.geom_type == "MultiPolygon":
        return max(forbidden_geom.geoms, key=lambda g: g.area)
    return None


def select_nfp_component_rings(forbidden_geom, max_holes: int = MAX_NFP_HOLES):
    """
    Pick up to three NFP boundary rings from the forbidden union.

    The union may contain multiple disconnected polygons; only the largest-area
    polygon is kept. Its exterior is component 1. Interior rings (holes) are ranked
    by enclosed area; the two largest become components 2 and 3. All other geometry
    is discarded.
    """
    poly = _primary_forbidden_polygon(forbidden_geom)
    if poly is None or poly.is_empty:
        return []

    rings = []
    exterior = poly.exterior
    if exterior is None or exterior.is_empty or len(exterior.coords) < 2:
        return []
    rings.append((1, exterior))

    ranked_holes = []
    for interior in poly.interiors:
        hole_poly = Polygon(interior)
        if hole_poly.is_empty or hole_poly.area <= 0:
            continue
        ranked_holes.append((float(hole_poly.area), interior))
    ranked_holes.sort(key=lambda item: item[0], reverse=True)

    for comp_id, (_, interior) in enumerate(ranked_holes[:max_holes], start=2):
        rings.append((comp_id, interior))

    return rings
