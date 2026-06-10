"""Gym environment for RL training (variable families, train generators)."""

from __future__ import annotations

from nester.env.env.base_env import BaseEnv

_TRAINING_GENERATOR = "nfp_random"


class TrainingEnv(BaseEnv):
    """Training-facing env; always uses the ``nfp_random`` generator."""

    def __init__(
        self,
        db_path: str,
        family_size_range: tuple[int, int],
        episode_length: int,
        shared_memory_dict: dict[str, str],
        limit_samples: int,
        resample_length: float,
        resample_length_nfp: float | None = None,
        seed: int | None = None,
        env_index: int = 0,
        nfp_patch_size: int = 10,
        max_reward: bool = False,
        track_best_panel: bool = False,
    ):
        super().__init__(
            db_path=db_path,
            family_size_range=family_size_range,
            episode_length=episode_length,
            generator_type=_TRAINING_GENERATOR,
            shared_memory_dict=shared_memory_dict,
            limit_samples=limit_samples,
            resample_length=resample_length,
            resample_length_nfp=resample_length_nfp,
            seed=seed,
            env_index=env_index,
            nfp_patch_size=nfp_patch_size,
            max_reward=max_reward,
            track_best_panel=track_best_panel,
        )
