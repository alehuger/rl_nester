from shapely.geometry import MultiPolygon, Polygon

# Stage-2 NFP component semantics (1-based ids; 0 is padding in the agent):
#   1 — exterior of the primary (largest-area) forbidden polygon
#   2+ — remaining rings: holes of the primary polygon first (largest area),
#        then exteriors and holes of additional connected components.
# Paper default is 1 exterior + 2 holes.
MAX_NFP_HOLES = 2
MAX_NFP_COMPONENTS = 1 + MAX_NFP_HOLES


def _forbidden_polygons(forbidden_geom):
    if forbidden_geom is None or forbidden_geom.is_empty:
        return []
    if forbidden_geom.geom_type == "Polygon":
        geoms = [forbidden_geom]
    elif forbidden_geom.geom_type == "MultiPolygon":
        geoms = list(forbidden_geom.geoms)
    else:
        return []
    geoms = [g for g in geoms if g is not None and (not g.is_empty) and float(g.area) > 0.0]
    geoms.sort(key=lambda g: float(g.area), reverse=True)
    return geoms


def _primary_forbidden_polygon(forbidden_geom):
    polys = _forbidden_polygons(forbidden_geom)
    return polys[0] if polys else None


def _ranked_holes(poly):
    ranked = []
    for interior in poly.interiors:
        hole_poly = Polygon(interior)
        if hole_poly.is_empty or hole_poly.area <= 0:
            continue
        ranked.append((float(hole_poly.area), interior))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [interior for _, interior in ranked]


def _valid_exterior(poly):
    exterior = poly.exterior
    if exterior is None or exterior.is_empty or len(exterior.coords) < 2:
        return None
    return exterior


def select_nfp_component_rings(
    forbidden_geom,
    max_holes: int = MAX_NFP_HOLES,
    max_components: int | None = None,
):
    """
    Pick NFP boundary rings from the forbidden union.

    The largest-area polygon is always first: its exterior is component 1, then
    its holes ranked by enclosed area. If ``max_components`` is larger than
    ``1 + max_holes``, additional connected components are appended (exterior
    then holes) until the budget is filled. Paper default ``max_components=3``
    matches the historical “largest polygon + two holes” cap.
    """
    budget = int(max_components) if max_components is not None else 1 + int(max_holes)
    if budget <= 0:
        return []

    polys = _forbidden_polygons(forbidden_geom)
    if not polys:
        return []

    rings = []
    primary_ext = _valid_exterior(polys[0])
    if primary_ext is None:
        return []
    rings.append((1, primary_ext))

    def _append_holes(poly):
        for interior in _ranked_holes(poly):
            if len(rings) >= budget:
                return
            rings.append((len(rings) + 1, interior))

    _append_holes(polys[0])
    for poly in polys[1:]:
        if len(rings) >= budget:
            break
        ext = _valid_exterior(poly)
        if ext is None:
            continue
        rings.append((len(rings) + 1, ext))
        _append_holes(poly)

    return rings
