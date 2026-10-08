# Methods: init, reset, step, render, get_reward
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon, unary_union
from shapely.affinity import rotate, translate

from nester.env.constants import (
    MAX_LENGTH,
    MAX_WIDTH,
    TRANSLATION_INIT_RANGE,
    TRANSLATION_SCALE,
)
from nester.env.env.generator import (
    GeneratorBench,
    GeneratorNFPRandom,
    GeneratorPolygons,
    Scenario,
)
from nester.env.panel.metrics import DEFAULT_OCCUPANCY_METRIC, validate_occupancy_metric
from nester.env.panel.nfp.cgal import _run_c_binding
from nester.env.panel.nfp.utils import MAX_NFP_COMPONENTS, select_nfp_component_rings
from nester.env.panel.panel import Panel


class BaseEnv(gym.Env):
    def __init__(
        self,
        db_path: str,
        family_size_range: tuple[int, int],
        episode_length: int,
        generator_type: str,
        shared_memory_dict: dict[str, str],
        limit_samples: int,
        resample_length: float,
        resample_length_nfp: float | None = None,
        seed: int | None = None,
        env_index: int = 0,
        max_vertices: int = 1000,
        max_nfp_vertices: int | None = None,
        nfp_patch_size: int = 10,
        max_reward: bool = False,
        track_best_panel: bool = False,
        occupancy_metric: str = DEFAULT_OCCUPANCY_METRIC,
        max_nfp_components: int = MAX_NFP_COMPONENTS,
        step_warmup: int = 0,
    ):
        self.config_db_path = db_path
        self.occupancy_metric = validate_occupancy_metric(occupancy_metric)
        self.episode_length = episode_length
        self.family_size_range = family_size_range
        self.generator_type = generator_type
        self.resample_length = resample_length
        self.resample_length_nfp = resample_length_nfp
        self.seed = seed
        self.env_index = env_index

        self.max_vertices = max_vertices
        self.max_nfp_vertices = max_nfp_vertices
        self.nfp_patch_size = int(nfp_patch_size)
        self.max_reward = bool(max_reward)
        self.track_best_panel = bool(track_best_panel)
        self.max_nfp_components = int(max_nfp_components)
        self.step_warmup = max(int(step_warmup), 0)

        self.translation_range = TRANSLATION_INIT_RANGE
        self.action_bounds = TRANSLATION_INIT_RANGE * TRANSLATION_SCALE

        if generator_type == "polygons":
            self.generator_class = GeneratorPolygons
        elif generator_type == "nfp_random":
            self.generator_class = GeneratorNFPRandom
        elif generator_type in ("bench", "bench_polygons"):
            self.generator_class = GeneratorBench
        else:
            raise ValueError(
                f"Invalid generator type: {generator_type}. "
                "Expected one of: polygons, nfp_random, bench, bench_polygons"
            )

        self.generator = self.generator_class(
            db_path,
            shared_memory_dict,
            family_size_range,
            limit_samples,
            self.translation_range,
            resample_length=self.resample_length,
            max_vertices=self.max_vertices,
            max_nfp_components=self.max_nfp_components,
        )

        # SPACES
        self.panel_bounds = np.array([MAX_LENGTH, MAX_WIDTH]).astype(np.float32)

        self.action_space = gym.spaces.Dict(
            {
                "polygon_id": gym.spaces.Discrete(self.family_size_range[1], seed=self.seed),
                "theta": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32, seed=self.seed
                ),
                "translation": gym.spaces.Box(
                    low=-self.action_bounds,
                    high=self.action_bounds,
                    shape=(2,),
                    dtype=np.float32,
                    seed=self.seed,
                ),
            }
        )
        # Nominal upper bounds; runtime obs uses only placed polygons (see _get_flattened_observation).
        max_polys = int(self.family_size_range[1])
        max_flat_pts = int(self.max_vertices) * max_polys
        coord_hi = float(max(self.panel_bounds) / 2)
        self.observation_space = gym.spaces.Dict(
            {
                "flat_vertices": gym.spaces.Box(
                    low=-coord_hi,
                    high=coord_hi,
                    shape=(max_flat_pts, 2),
                    dtype=np.float32,
                ),
                "poly_lengths": gym.spaces.Box(
                    low=0,
                    high=self.max_vertices,
                    shape=(max_polys,),
                    dtype=np.int64,
                ),
                "poly_ids": gym.spaces.Box(
                    low=0,
                    high=max(max_polys - 1, 0),
                    shape=(max_polys,),
                    dtype=np.int64,
                ),
                "poly_theta": gym.spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(max_polys,),
                    dtype=np.float32,
                ),
                "normalized_step": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
                "prev_best_occupancy": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
            }
        )

        self.panel: Panel | None = None
        self.previous_occupancy_ratio = None
        self.current_occupancy_ratio = None
        self.base_polygons = {}
        self.current_theta: dict[int, float] = {}
        self.placed_ids = set()

        self.reset(seed=self.seed)

    def _resample_polyline_numpy(self, coords: np.ndarray, distance: float) -> np.ndarray:
        if coords.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.array(coords, dtype=np.float32)
        vectors = pts[1:] - pts[:-1]
        lengths = np.linalg.norm(vectors, axis=1)
        total_length = np.sum(lengths)
        if total_length <= 0.0:
            return np.zeros((0, 2), dtype=np.float32)
        target_dists = np.arange(0, total_length, distance, dtype=np.float32)
        if target_dists.shape[0] == 0:
            return pts[:1].astype(np.float32)
        cum_lengths = np.cumsum(lengths)
        seg_idx = np.searchsorted(cum_lengths, target_dists, side="right")
        seg_idx = np.clip(seg_idx, 0, len(vectors) - 1)
        seg_starts = np.zeros_like(lengths)
        seg_starts[1:] = cum_lengths[:-1]
        offsets = target_dists - seg_starts[seg_idx]
        safe = lengths[seg_idx].copy()
        safe[safe == 0] = 1.0
        ratios = offsets / safe
        out = pts[seg_idx] + ratios[:, None] * vectors[seg_idx]
        return out.astype(np.float32)

    def compute_nfp_for_action(self, polygon_id_batch, theta_batch):
        polygon_id_arr = np.asarray(polygon_id_batch).reshape(-1)
        theta_arr = np.asarray(theta_batch).reshape(-1)

        if self.env_index < 0 or self.env_index >= polygon_id_arr.shape[0]:
            raise ValueError(
                f"env_index {self.env_index} out of range for polygon_id_batch of shape {polygon_id_arr.shape}"
            )

        polygon_id = int(polygon_id_arr[self.env_index])
        theta = float(theta_arr[self.env_index])

        base_coords = self.base_polygons[polygon_id]
        moving_poly = Polygon(base_coords)
        moving_centroid = np.array(moving_poly.centroid.coords[0], dtype=np.float32)
        rotated_moving = rotate(moving_poly, theta, origin=moving_poly.centroid, use_radians=True)

        P = self.nfp_patch_size
        if len(self.placed_ids) == 0:
            return {
                "nfp_patches": np.zeros((1, P, 2), dtype=np.float32),
                "nfp_patch_mask": np.array([True], dtype=np.bool_),
                "nfp_vertex_mask": np.ones((1, P), dtype=np.bool_),
                "nfp_poly_ids": np.array([1], dtype=np.int64),
            }
        if polygon_id not in self.placed_ids:
            raise ValueError(
                f"compute_nfp_for_action requires selected polygon to be placed, got polygon_id={polygon_id}"
            )

        forbidden_regions = []
        # Keep CGAL input consistent with the selected orientation:
        # moving polygon is already rotated, so binding rotations stay zero.
        moving_base = rotated_moving

        for other_id in self.placed_ids:
            if other_id == polygon_id:
                continue
            other = self.panel.polygons[other_id]
            nfp_dict, _ = _run_c_binding(other._outline, moving_base, 0, 0)
            shell = np.array(nfp_dict.get("shell", []), dtype=np.float32)
            holes = [np.array(h, dtype=np.float32) for h in nfp_dict.get("holes", [])]
            if shell.shape[0] < 3:
                continue
            # Shift from origin reference to world centroid reference of the stationary polygon.
            ocx, ocy = other.centroid
            # Convert first-vertex-reference (CGAL output) to centroid-reference for moving polygon.
            shift = np.array([ocx, ocy], dtype=np.float32)  # - moving_first_vertex_offset
            shell_world = shell + shift
            holes_world = [h + shift for h in holes]
            try:
                poly = Polygon(shell_world, holes_world)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                forbidden_regions.append(poly)
            except Exception as e:
                raise RuntimeError(
                    f"NFP forbidden-region construction failed for polygon_id={polygon_id} "
                    f"with other_id={other_id}"
                ) from e

        if not forbidden_regions:
            raise ValueError(
                f"No forbidden NFP regions found for polygon_id={polygon_id} with placed polygons"
            )

        forbidden_union = unary_union(forbidden_regions)
        component_rings = select_nfp_component_rings(
            forbidden_union, max_components=self.max_nfp_components
        )
        if not component_rings:
            raise ValueError(f"Empty NFP boundary after geometry processing for polygon_id={polygon_id}")

        patch_list = []
        patch_mask_list = []
        patch_id_list = []
        total_points = 0
        for comp_idx, line in component_rings:
            coords = np.array(line.coords, dtype=np.float32)
            if coords.shape[0] < 2:
                continue
            pts = self._resample_polyline_numpy(coords, self.resample_length_nfp)
            if pts.shape[0] == 0:
                pts = coords
            if self.max_nfp_vertices is not None and total_points + pts.shape[0] > self.max_nfp_vertices:
                raise ValueError(
                    f"NFP contour has {total_points + pts.shape[0]} points, exceeds "
                    f"max_nfp_vertices={self.max_nfp_vertices}."
                )
            chunk = (pts - moving_centroid).astype(np.float32)
            total_points += pts.shape[0]
            k = chunk.shape[0]
            start = 0
            while start < k:
                end = min(start + P, k)
                piece = chunk[start:end]
                piece_mask = np.zeros((P,), dtype=np.bool_)
                piece_mask[: piece.shape[0]] = True
                if piece.shape[0] < P:
                    pad = np.repeat(piece[-1:], P - piece.shape[0], axis=0)
                    piece = np.concatenate([piece, pad], axis=0)
                patch_list.append(piece)
                patch_mask_list.append(piece_mask)
                patch_id_list.append(comp_idx)
                start += P

        if len(patch_list) == 0:
            raise ValueError(f"NFP contour sampling produced zero points for polygon_id={polygon_id}")

        nfp_patches = np.stack(patch_list, axis=0).astype(np.float32)
        nfp_vertex_mask = np.stack(patch_mask_list, axis=0).astype(np.bool_)
        nfp_poly_ids = np.asarray(patch_id_list, dtype=np.int64)

        return {
            "nfp_patches": nfp_patches,
            "nfp_patch_mask": np.ones((nfp_patches.shape[0],), dtype=np.bool_),
            "nfp_vertex_mask": nfp_vertex_mask,
            "nfp_poly_ids": nfp_poly_ids,
        }

    def _resample_polygon_numpy(self, coords: list, distance: float) -> np.ndarray:
        """
        Resamples a polygon's exterior using vectorized NumPy operations.
        Approximately 50-100x faster than Shapely iteration.
        """
        # 1. Basic validation (Shapely needs 3, but math works with 2. Keeping 3 to match original logic)
        if len(coords) < 3:
            return np.array([], dtype=np.float32).reshape(0, 2)

        pts = np.array(coords, dtype=np.float32)

        # 2. Ensure the polygon is closed (last point == first point)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        # 3. Calculate segment lengths
        # vectors of each segment (dx, dy)
        vectors = pts[1:] - pts[:-1]
        # Length of each segment
        lengths = np.linalg.norm(vectors, axis=1)

        # 4. Handle empty/degenerate polygons
        total_length = np.sum(lengths)
        if total_length == 0.0:
            return np.array([], dtype=np.float32).reshape(0, 2)

        # 5. Cumulative distance at each vertex (starts at 0)
        # cum_lengths[i] is the distance from start to the END of segment i
        cum_lengths = np.cumsum(lengths)

        # 6. Generate target distances
        # We stop before total_length to match standard interpolate behavior
        # (or use <= if you want the endpoint included)
        target_dists = np.arange(0, total_length, distance)

        if len(target_dists) == 0:
            return np.array([], dtype=np.float32).reshape(0, 2)

        # 7. Find which segment each target distance falls into
        # searchsorted finds indices where elements should be inserted to maintain order.
        # We search in cum_lengths to find the segment index.
        segment_indices = np.searchsorted(cum_lengths, target_dists, side="right")

        # 8. Calculate the exact coordinates
        # For points exactly on vertices, searchsorted might push to the next segment,
        # so we clip to valid range.
        segment_indices = np.clip(segment_indices, 0, len(vectors) - 1)

        # The distance 'local' to the specific segment
        # Note: We need the start distance of the segment.
        # If index is 0, start is 0. If index > 0, start is cum_lengths[index-1]
        segment_starts = np.empty_like(lengths)
        segment_starts[0] = 0
        segment_starts[1:] = cum_lengths[:-1]

        start_dists_for_points = segment_starts[segment_indices]

        # How far along the segment are we?
        offsets = target_dists - start_dists_for_points

        # Avoid division by zero for zero-length segments
        safe_lengths = lengths[segment_indices]
        # Replace 0 with 1 to avoid NaN (the offset will be 0 anyway)
        safe_lengths[safe_lengths == 0] = 1.0

        ratios = offsets / safe_lengths

        # Linear interpolation: Start + Ratio * Vector
        # reshape ratios to (N, 1) to broadcast against vectors (N, 2)
        new_points = pts[segment_indices] + (ratios[:, np.newaxis] * vectors[segment_indices])

        return new_points.astype(np.float32)

    def _get_flattened_observation(self) -> dict:
        frame_center = np.array([0.0, 0.0], dtype=np.float32)

        flat_verts_list = []
        lengths_list = []
        ids_list = []
        theta_list = []

        denom = float(self.episode_length + self.step_warmup)
        normalized_step = (
            np.float32(np.clip(self.step_count / denom, 0.0, 1.0))
            if denom > 0.0
            else np.float32(0.0)
        )

        for i in range(self.family_size_range[1]):
            if i in self.panel.polygons:
                verts = self._resample_polygon_numpy(
                    self.panel.polygons[i].get_array_observation(), self.resample_length
                )
                if verts.shape[0] > self.max_vertices:
                    raise ValueError(
                        f"Polygon slot {i} resampled to {verts.shape[0]} vertices, "
                        f"exceeds max_vertices={self.max_vertices}. "
                        f"Increase environment.max_vertices or use a larger resample_length "
                        f"(currently {self.resample_length})."
                    )

                if verts.shape[0] > 0:
                    verts = verts - frame_center
                    flat_verts_list.append(verts)
                    lengths_list.append(verts.shape[0])
                    ids_list.append(i)
                    theta_list.append(np.float32(self.current_theta.get(i, 0.0)))

        # Handle the edge case: Environment is completely empty
        prev_best = np.float32(self.prev_best_occupancy_for_obs)
        if not flat_verts_list:
            return {
                "flat_vertices": np.zeros((0, 2), dtype=np.float32),
                "poly_lengths": np.zeros((0,), dtype=np.int64),
                "poly_ids": np.zeros((0,), dtype=np.int64),
                "poly_theta": np.zeros((0,), dtype=np.float32),
                "normalized_step": normalized_step,
                "prev_best_occupancy": prev_best,
            }

        # World-frame vertices. For ``aabb`` the origin-clamped metric keeps the nest
        # near (0, 0). For ``min_rect`` the panel is recentered on its AABB after
        # each transition (see ``_maybe_recenter_panel``), so this stays in-range.
        return {
            "flat_vertices": np.concatenate(flat_verts_list, axis=0).astype(np.float32),
            "poly_lengths": np.array(lengths_list, dtype=np.int64),
            "poly_ids": np.array(ids_list, dtype=np.int64),
            "poly_theta": np.array(theta_list, dtype=np.float32),
            "normalized_step": normalized_step,
            "prev_best_occupancy": prev_best,
        }

    def get_panel_observation(self):
        return self.panel.get_observation()

    def _panel_snapshot(self) -> dict[int, np.ndarray]:
        return {
            int(k): np.asarray(v, dtype=np.float32).copy()
            for k, v in self.get_panel_observation().items()
        }

    def _load_panel_from_scenario(self, scenario: Scenario) -> None:
        self.panel = Panel(
            scenario.placed_polygons, self.panel_bounds, occupancy_metric=self.occupancy_metric
        )
        self.base_polygons = {}
        self.current_theta = {}
        self.placed_ids = set(self.panel.polygons.keys())
        for i, panel_polygon in self.panel.polygons.items():
            coords = np.array(panel_polygon._outline.exterior.coords, dtype=np.float32)
            centered = coords - np.array(panel_polygon.centroid, dtype=np.float32)
            self.base_polygons[i] = centered
            self.current_theta[i] = 0.0

    def _maybe_recenter_panel(self) -> None:
        """Keep the nest near the origin when occupancy is translation-invariant.

        ``min_rect`` does not penalize a rigid shift, so without this the cluster can
        walk outside the declared observation box (±MAX_LENGTH/2). AABB occupancy
        already pins the layout to the origin, so we leave that metric unchanged.
        """
        if self.occupancy_metric != "min_rect":
            return
        self.panel.recenter_aabb()

    def _init_episode_metrics(self) -> None:
        self.initial_occupancy_ratio = self.panel.get_occupancy_ratio()
        self.previous_occupancy_ratio = self.initial_occupancy_ratio
        self.best_occupancy_rollout = float(self.initial_occupancy_ratio)
        self.prev_best_occupancy_for_obs = float(self.initial_occupancy_ratio)
        self.reward_cumulative = 0
        self.nonzero_reward_count = 0
        self.best_validation_score = float(self.initial_occupancy_ratio)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.step_count = 0
        if options is not None:
            seed_polygons = options.get("seed_polygons", seed)
            seed_initial_placement = options.get("seed_initial_placement", seed)
        else:
            seed_polygons = seed
            seed_initial_placement = seed

        if seed_polygons is None:
            seed_polygons = self.seed
        if seed_initial_placement is None:
            seed_initial_placement = self.seed

        pending_seed = getattr(self, "_pending_seed_initial_placement", None)
        if pending_seed is not None:
            self._pending_seed_initial_placement = None
            seed_initial_placement = int(pending_seed)

        pending_polygons = getattr(self, "_pending_initial_placed_polygons", None)
        if pending_polygons is not None:
            self._pending_initial_placed_polygons = None
            scenario = Scenario(pending_polygons)
        else:
            scenario = self.generator.generate(
                seed_polygons=seed_polygons, seed_initial_placement=seed_initial_placement
            )
        self._load_panel_from_scenario(scenario)
        self._init_episode_metrics()
        self._maybe_recenter_panel()
        if self.track_best_panel:
            self.best_occupancy_ratio = self.initial_occupancy_ratio
            self.best_panel = self.get_panel_observation()
        return self._get_flattened_observation(), {}

    def curriculum_warmup_snapshot(self) -> dict:
        return {
            "current_occupancy": float(self.panel.get_occupancy_ratio()),
            "best_occupancy_rollout": float(self.best_occupancy_rollout),
        }

    def curriculum_start_training_phase(self, active_mask: np.ndarray | None = None):
        """Start the on-policy segment after warmup.

        Panel geometry stays at the **last warmup step** (not reverted to peak layout).
        ``best_occupancy_rollout`` and ``prev_best_occupancy_for_obs`` are set to the
        **peak occupancy** observed during warmup (reward bar for ``max_reward``).
        """
        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
            if not bool(mask[int(self.env_index)]):
                return self._get_flattened_observation()

        warmup_peak = float(self.best_occupancy_rollout)
        current_occupancy = float(self.panel.get_occupancy_ratio())
        self.step_count = 0
        self.initial_occupancy_ratio = current_occupancy
        self.previous_occupancy_ratio = current_occupancy
        self.best_occupancy_rollout = warmup_peak
        self.prev_best_occupancy_for_obs = warmup_peak
        self.reward_cumulative = 0.0
        self.nonzero_reward_count = 0
        return self._get_flattened_observation()

    def curriculum_masked_step(self, env_action: dict, active_mask: np.ndarray):
        """Step only if active_mask[env_index]; otherwise return current obs without changing state."""
        i = int(self.env_index)
        mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        if not bool(mask[i]):
            cur = float(self.panel.get_occupancy_ratio())
            return (
                self._get_flattened_observation(),
                0.0,
                False,
                False,
                {"current_occupancy": cur},
            )

        polygon_ids = np.asarray(env_action["polygon_id"], dtype=np.int64).reshape(-1)
        translations = np.asarray(env_action["translation"], dtype=np.float32)
        thetas = np.asarray(env_action["theta"], dtype=np.float32)

        single_action = {
            "polygon_id": int(np.asarray(polygon_ids[i]).reshape(-1)[0]),
            "translation": translations[i].reshape(2).astype(np.float32),
            "theta": np.atleast_1d(thetas[i]).astype(np.float32),
        }
        return self.step(single_action)

    def step(self, action: dict):
        self.step_count += 1
        prev_best = self.best_occupancy_rollout

        polygon_id = int(np.asarray(action["polygon_id"]).reshape(-1)[0])
        if polygon_id not in self.placed_ids:
            raise ValueError(f"Improvement mode requires placed polygon, got polygon_id={polygon_id}")

        delta_theta = float(np.asarray(action["theta"]).reshape(-1)[0])
        translation = np.array(action["translation"], dtype=np.float32).reshape(2)
        theta_abs = float(
            np.arctan2(
                np.sin(self.current_theta.get(polygon_id, 0.0) + delta_theta),
                np.cos(self.current_theta.get(polygon_id, 0.0) + delta_theta),
            )
        )
        base_poly = Polygon(self.base_polygons[polygon_id])
        placed_poly = rotate(base_poly, theta_abs, origin=base_poly.centroid, use_radians=True)
        placed_poly = translate(placed_poly, xoff=float(translation[0]), yoff=float(translation[1]))
        self.panel.polygons[polygon_id]._outline = placed_poly
        self.current_theta[polygon_id] = theta_abs
        self.placed_ids.add(polygon_id)

        current_occupancy_ratio = self.panel.get_occupancy_ratio()
        reward = self.get_reward(current_occupancy_ratio, prev_best)
        self.reward_cumulative += reward
        if reward > 0.0:
            self.nonzero_reward_count += 1

        self.best_occupancy_rollout = max(prev_best, float(current_occupancy_ratio))
        # Recenter after scoring: ``min_rect`` occupancy is translation-invariant, so
        # this does not change the reward, but it keeps ``flat_vertices`` / NFP patches
        # inside the declared ±panel_bounds/2 observation box.
        self._maybe_recenter_panel()
        if self.track_best_panel and current_occupancy_ratio > getattr(
            self, "best_occupancy_ratio", -float("inf")
        ):
            self.best_occupancy_ratio = current_occupancy_ratio
            self.best_panel = self.get_panel_observation()
        self.prev_best_occupancy_for_obs = self.best_occupancy_rollout
        self.previous_occupancy_ratio = current_occupancy_ratio

        info = {
            "current_occupancy": current_occupancy_ratio,
        }
        done = self.compute_done()
        if done:
            info["increase_occupancy_ratio"] = (
                current_occupancy_ratio - self.initial_occupancy_ratio
            )
            info["final_occupancy_ratio"] = current_occupancy_ratio
            self.best_validation_score = float(self.best_occupancy_rollout)
            info["best_validation_score"] = self.best_validation_score
            info["best_occupancy_ratio"] = float(self.best_occupancy_rollout)
            info["initial_occupancy_ratio"] = float(self.initial_occupancy_ratio)
            info["final_reward_cumulative"] = self.reward_cumulative
            info["nonzero_reward_ratio"] = self.nonzero_reward_count / max(self.step_count, 1)
            if self.track_best_panel:
                info["best_occupancy_ratio"] = self.best_occupancy_ratio
                info["best_panel"] = self.best_panel

        return self._get_flattened_observation(), reward, done, False, info

    def render(self):
        fig = self.panel.render_centered()
        occupancy_ratio = self.panel.get_occupancy_ratio()
        reward = self.get_reward(occupancy_ratio, self.best_occupancy_rollout)
        plt.title(f"Occupancy Ratio: {occupancy_ratio:.4f}, Reward: {reward:.4f}")
        return fig

    def get_reward(self, current_occupancy_ratio: float, prev_best_rollout: float):
        if self.max_reward:
            d = float(current_occupancy_ratio) - float(prev_best_rollout)
            return d if d > 0.0 else 0.0
        return float(current_occupancy_ratio) - float(self.previous_occupancy_ratio)

    def compute_done(self):
        return self.step_count >= self.episode_length


if __name__ == "__main__":
    env = BaseEnv(
        db_path="data/db/processed/train_db.npz",
        family_size_range=(4, 4),
        episode_length=5,
        generator_type="nfp_random",
        shared_memory_dict=None,
        limit_samples=100,
        resample_length=0.1,
        seed=None,
    )
