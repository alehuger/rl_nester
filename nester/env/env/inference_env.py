"""Dedicated env for bench vector inference (fixed family in shared memory)."""

from __future__ import annotations

import numpy as np

from nester.env.env.base_env import BaseEnv
from nester.env.panel.nfp.utils import MAX_NFP_COMPONENTS
from nester.env.utils.sharing_memory import PolygonsShmView


_INFERENCE_GENERATOR = "bench"


class InferenceEnv(BaseEnv):
    """Bench inference env; always uses the ``bench`` generator."""

    def __init__(
        self,
        episode_length: int,
        shared_memory_dict: dict[str, str],
        resample_length: float,
        seed: int,
        env_index: int,
        max_vertices: int,
        nfp_patch_size: int,
        max_reward: bool,
        use_nfp_cache: bool = False,
        nfp_cache_bins: int = 720,
        nfp_cache_shm_meta: dict | None = None,
        placement_seed: int | None = None,
        resample_length_nfp: float | None = None,
        occupancy_metric: str = "aabb",
        max_nfp_components: int = MAX_NFP_COMPONENTS,
        step_warmup: int = 0,
    ):
        _ = (use_nfp_cache, nfp_cache_bins, nfp_cache_shm_meta, placement_seed)

        n_poly = len(PolygonsShmView(shared_memory_dict))
        super().__init__(
            db_path="",
            family_size_range=(n_poly, n_poly),
            episode_length=episode_length,
            generator_type=_INFERENCE_GENERATOR,
            shared_memory_dict=shared_memory_dict,
            limit_samples=n_poly,
            resample_length=resample_length,
            resample_length_nfp=resample_length_nfp,
            seed=seed,
            env_index=env_index,
            max_vertices=max_vertices,
            nfp_patch_size=nfp_patch_size,
            max_reward=max_reward,
            track_best_panel=True,
            occupancy_metric=occupancy_metric,
            max_nfp_components=max_nfp_components,
            step_warmup=step_warmup,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._sync_bench_snapshot()
        self._refresh_bench_export()
        return obs, info

    def step(self, action: dict):
        prev_best = self.best_occupancy_rollout
        obs, reward, done, truncated, info = super().step(action)
        if self.best_occupancy_rollout > prev_best + 1e-12:
            self._sync_bench_snapshot()
        self._refresh_bench_export()
        return obs, reward, done, truncated, info

    def _refresh_bench_export(self) -> None:
        self._last_bench_export = {
            "best_occupancy_ratio": float(self.best_occupancy_rollout),
            "best_panel": {
                int(k): v.copy() for k, v in self._bench_best_panel_snapshot.items()
            },
            "max_vertices": int(self.max_vertices),
            "best_occ_step": int(self._bench_best_step),
        }

    def _sync_bench_snapshot(self) -> None:
        self._bench_best_panel_snapshot = {
            int(k): np.asarray(v, dtype=np.float32).copy()
            for k, v in self.get_panel_observation().items()
        }
        self._bench_best_step = int(self.step_count)

    def get_bench_rollout_exports(self):
        ex = getattr(self, "_last_bench_export", None)
        if ex is None:
            return {
                "best_occupancy_ratio": float(self.best_occupancy_rollout),
                "best_panel": {},
                "max_vertices": int(self.max_vertices),
                "best_occ_step": int(getattr(self, "_bench_best_step", 0)),
            }
        return {
            "best_occupancy_ratio": float(ex["best_occupancy_ratio"]),
            "best_panel": {
                int(k): np.asarray(v, dtype=np.float32).copy() for k, v in ex["best_panel"].items()
            },
            "max_vertices": int(ex["max_vertices"]),
            "best_occ_step": int(ex.get("best_occ_step", 0)),
        }
