from abc import abstractmethod

import numpy as np
from shapely import Polygon, unary_union

from nester.env.utils.sharing_memory import load_polygons_from_shared_memory
from nester.env.utils.utils import (
    load_list_from_dict,
    npz_load_polygons,
    rotate_np_array,
)
from nester.env.panel.nfp.cgal import _run_c_binding
from nester.env.panel.nfp.utils import select_nfp_component_rings


class Scenario:
    def __init__(self, placed_polygons: list[list[tuple[float, float]]]):
        self.placed_polygons = placed_polygons


class Generator:
    def __init__(
        self,
        db_path: str,
        family_size_range: tuple[int, int],
        shared_memory_dict: dict[str, str] | None,
        limit_samples: int,
        translation_range: float,
        **kwargs,
    ):
        self.db_path = db_path
        self.family_size_range = family_size_range
        self.shared_memory_dict = shared_memory_dict
        self.limit_samples = limit_samples
        self.translation_range = translation_range

        if shared_memory_dict is not None:
            self._memory_db_dict = self.load_db_from_shared_memory(shared_memory_dict)
        else:
            self._memory_db_dict = self.load_db_from_file(self.db_path)

        self.nb_polygons = len(self._memory_db_dict)

    @abstractmethod
    def load_db_from_file(self, db_path: str) -> list[np.ndarray]:
        pass

    @abstractmethod
    def load_db_from_shared_memory(self, shared_memory_dict: dict[str, str]):
        pass

    @abstractmethod
    def generate(self, seed_polygons: int, seed_initial_placement: int) -> Scenario:
        pass

    def _place_polygon_randomly(self, polygon: np.ndarray):
        polygon = polygon[~np.isnan(polygon).any(axis=1)]
        random_translation = np.random.uniform(-self.translation_range, self.translation_range, 2)
        random_rotation = np.random.uniform(-np.pi, np.pi)
        polygon = rotate_np_array(polygon, random_rotation)
        polygon = polygon + random_translation
        return polygon

    def _place_randomly(self, family: list[np.ndarray]) -> Scenario:
        placed_polygons = []
        for polygon in family:
            placed_polygons.append(self._place_polygon_randomly(polygon))
        return Scenario(placed_polygons)


class GeneratorPolygons(Generator):
    def __init__(
        self,
        polygons_db_path: str,
        shared_memory_dict: dict[str, str],
        family_size_range: tuple[int, int],
        limit_samples: int,
        translation_range: float,
        **kwargs,
    ):
        super().__init__(
            polygons_db_path,
            family_size_range,
            shared_memory_dict,
            limit_samples,
            translation_range,
        )

    def load_db_from_shared_memory(self, shared_memory_dict: dict[str, str]):
        return load_polygons_from_shared_memory(shared_memory_dict)

    def load_db_from_file(self, db_path: str):
        data_dict = npz_load_polygons(db_path, self.limit_samples)
        return load_list_from_dict(data_dict)

    def generate(self, seed_polygons: int, seed_initial_placement: int):
        family = self.sample_family(seed_polygons)
        np.random.seed(seed_initial_placement)
        return self._place_randomly(family)

    def sample_family(self, seed: int) -> list[np.ndarray]:
        np.random.seed(seed)
        family_size = np.random.randint(self.family_size_range[0], self.family_size_range[1] + 1)
        polygon_indices = np.random.choice(self.nb_polygons, family_size, replace=False)
        return [self._memory_db_dict[i] for i in polygon_indices]


class GeneratorNFPRandom(GeneratorPolygons):
    def __init__(
        self,
        polygons_db_path: str,
        shared_memory_dict: dict[str, str],
        family_size_range: tuple[int, int],
        limit_samples: int,
        translation_range: float,
        resample_length: float = 0.1,
        max_vertices: int = 1000,
    ):
        super().__init__(
            polygons_db_path,
            shared_memory_dict,
            family_size_range,
            limit_samples,
            translation_range,
        )
        self.resample_length = resample_length
        self.max_vertices = max_vertices

    def _compute_nfp_candidates(self, moving_poly: Polygon, placed_polys):
        moving_centroid = np.array(moving_poly.centroid.coords[0], dtype=np.float32)
        forbidden_regions = []
        for placed in placed_polys:
            other_poly = Polygon(placed)
            nfp_dict, _ = _run_c_binding(other_poly, moving_poly, 0, 0)
            shell = np.array(nfp_dict.get("shell", []), dtype=np.float32)
            holes = [np.array(h, dtype=np.float32) for h in nfp_dict.get("holes", [])]
            if shell.shape[0] < 3:
                continue
            ocx, ocy = other_poly.centroid.x, other_poly.centroid.y
            shift = np.array([ocx, ocy], dtype=np.float32)
            shell_world = shell + shift
            holes_world = [h + shift for h in holes]
            try:
                forbidden_regions.append(Polygon(shell_world, holes_world))
            except Exception as e:
                raise RuntimeError(
                    "NFP forbidden-region construction failed during random initialization"
                ) from e
        if not forbidden_regions:
            raise ValueError(
                "No forbidden NFP regions found while building random NFP initialization"
            )

        forbidden_union = unary_union(forbidden_regions)
        component_rings = select_nfp_component_rings(forbidden_union)
        if not component_rings:
            raise ValueError("Empty NFP boundary while building random NFP initialization")

        pts_list = []
        for _, line in component_rings:
            coords = np.array(line.coords, dtype=np.float32)
            if coords.shape[0] < 2:
                continue
            pts_list.append(coords)
        if not pts_list:
            raise ValueError(
                "NFP contour sampling produced zero points during random NFP initialization"
            )

        pts = np.concatenate(pts_list, axis=0).astype(np.float32)
        if pts.shape[0] > self.max_vertices:
            idx = np.random.choice(pts.shape[0], self.max_vertices, replace=False)
            pts = pts[idx]

        return (pts - moving_centroid).astype(np.float32)

    def generate(self, seed_polygons: int, seed_initial_placement: int) -> Scenario:
        np.random.seed(seed_polygons)
        family_size = np.random.randint(self.family_size_range[0], self.family_size_range[1] + 1)
        polygon_indices = np.random.choice(self.nb_polygons, family_size, replace=False)
        family = []
        for idx in polygon_indices:
            polygon = self._memory_db_dict[idx].copy()
            polygon = polygon[~np.isnan(polygon).any(axis=1)]
            family.append(polygon.astype(np.float32))

        rng = np.random.default_rng(seed_initial_placement)
        order = rng.permutation(family_size)
        placed_by_id = {}
        placed_sequence = []

        for step_idx, poly_id in enumerate(order.tolist()):
            polygon = family[poly_id]
            theta = float(rng.uniform(-np.pi, np.pi))
            rotated = rotate_np_array(polygon, theta).astype(np.float32)
            if step_idx == 0:
                translation = np.zeros((2,), dtype=np.float32)
            else:
                candidates = self._compute_nfp_candidates(Polygon(rotated), placed_sequence)
                translation = candidates[int(rng.integers(0, candidates.shape[0]))]
                if translation.shape[0] == 0:
                    raise ValueError(
                        f"No feasible NFP location during random initialization for polygon id={poly_id} "
                        f"(seed_polygons={seed_polygons}, seed_initial_placement={seed_initial_placement})"
                    )

            placed = rotated + translation
            placed_by_id[poly_id] = placed.astype(np.float32)
            placed_sequence.append(placed.astype(np.float32))

        placed_polygons = [placed_by_id[i] for i in range(family_size)]
        return Scenario(placed_polygons)


class GeneratorBench(GeneratorNFPRandom):
    """Fixed validation family from SHM (bench); NFP-feasible sequential init like ``nfp_random``."""

    def load_db_from_file(self, db_path: str):
        raise RuntimeError("bench generator requires shared_memory_dict")

    def load_db_from_shared_memory(self, shared_memory_dict: dict[str, str]):
        view = load_polygons_from_shared_memory(shared_memory_dict)
        return [view[i] for i in range(len(view))]

    def generate(self, seed_polygons: int, seed_initial_placement: int) -> Scenario:
        K = len(self._memory_db_dict)
        if K == 0:
            raise ValueError("bench: empty polygon list in shared memory")
        lo, hi = self.family_size_range
        if lo != hi or hi != K:
            raise ValueError(
                f"bench: family_size_range {(lo, hi)} must equal fixed family size K={K}"
            )

        family: list[np.ndarray] = []
        for i in range(K):
            polygon = self._memory_db_dict[i].copy()
            polygon = polygon[~np.isnan(polygon).any(axis=1)]
            family.append(polygon.astype(np.float32))

        rng = np.random.default_rng(seed_initial_placement)
        order = rng.permutation(K)
        placed_by_id: dict[int, np.ndarray] = {}
        placed_sequence: list[np.ndarray] = []

        for step_idx, poly_id in enumerate(order.tolist()):
            polygon = family[poly_id]
            theta = float(rng.uniform(-np.pi, np.pi))
            rotated = rotate_np_array(polygon, theta).astype(np.float32)
            if step_idx == 0:
                translation = np.zeros((2,), dtype=np.float32)
            else:
                try:
                    candidates = self._compute_nfp_candidates(Polygon(rotated), placed_sequence)
                    translation = candidates[int(rng.integers(0, candidates.shape[0]))]
                    if translation.shape[0] == 0:
                        raise ValueError(
                            f"No feasible NFP location during bench init for polygon id={poly_id}"
                        )
                except Exception as e:
                    print(f"Error computing NFP candidates for bench polygon id={poly_id}: {e}")
                    return self.generate(seed_polygons + 1, seed_initial_placement + 1)

            placed = rotated + translation
            placed_by_id[poly_id] = placed.astype(np.float32)
            placed_sequence.append(placed.astype(np.float32))

        placed_polygons = [placed_by_id[i] for i in range(K)]
        return Scenario(placed_polygons)

