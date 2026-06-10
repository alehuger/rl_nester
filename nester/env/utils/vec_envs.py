import gymnasium as gym
from gymnasium.vector.async_vector_env import AsyncState, NoAsyncCallError
from gymnasium.core import ObsType
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Union
import multiprocessing


class DynamicAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """
    An asynchronous vectorized environment that handles dynamic observation spaces.
    It uses the standard Gymnasium worker for stability but overrides step/reset
    to handle dictionary-based observation concatenation.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
        autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    ):
        # We generally do NOT pass a custom worker here to avoid version conflicts.
        # Letting it default to None uses Gymnasium's internal compatible worker.
        # Default SAME_STEP matches RL training; use DISABLED for bench export after the
        # last step (otherwise the worker's autoreset reset() overwrites rollout-best state).
        super().__init__(
            env_fns,
            shared_memory=False,  # Essential for dynamic/dict observations
            copy=copy,
            context=context,
            daemon=daemon,
            worker=None,
            autoreset_mode=autoreset_mode,
        )

    def step_wait(
        self, timeout: Optional[Union[int, float]] = None
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict]:
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `step_wait` has timed out after {timeout} second(s).")

        observations, rewards, terminations, truncations, infos = [], [], [], [], {}
        successes = []
        for env_idx, pipe in enumerate(self.parent_pipes):
            env_step_return, success = pipe.recv()

            successes.append(success)
            if success:
                observations.append(env_step_return[0])
                rewards.append(env_step_return[1])
                terminations.append(env_step_return[2])
                truncations.append(env_step_return[3])
                infos = self._add_info(infos, env_step_return[4], env_idx)

        self._raise_if_errors(successes)

        # Custom dynamic concatenation
        self.observations = self.concatenate_dynamic(observations)

        self._state = AsyncState.DEFAULT
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards, dtype=np.float64),
            np.array(terminations, dtype=np.bool_),
            np.array(truncations, dtype=np.bool_),
            infos,
        )

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `reset_wait` has timed out after {timeout} second(s).")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        # Unpack results: results is a tuple of ((obs, info), (obs, info), ...)
        obs_list, info_list = zip(*results)

        infos = {}
        for i, info in enumerate(info_list):
            infos = self._add_info(infos, info, i)

        self.observations = self.concatenate_dynamic(list(obs_list))

        self._state = AsyncState.DEFAULT
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def concatenate_dynamic(self, observations: list[Any]) -> dict:
        """Concatenate dynamic observations as a dictionary mapping env_idx -> obs."""
        if not observations:
            return {}
        return {index: obs for index, obs in enumerate(observations)}

    def compute_nfp(self, polygon_id_batch, theta_batch):
        return self.call("compute_nfp_for_action", polygon_id_batch, theta_batch)

    def curriculum_warmup_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        results = self.call("curriculum_warmup_snapshot")
        current = np.array([r["current_occupancy"] for r in results], dtype=np.float64)
        best = np.array([r["best_occupancy_rollout"] for r in results], dtype=np.float64)
        return current, best

    def curriculum_start_training_phase(self, active_mask: np.ndarray | None = None):
        results = self.call("curriculum_start_training_phase", active_mask)
        self.observations = self.concatenate_dynamic(list(results))
        return deepcopy(self.observations) if self.copy else self.observations

    def curriculum_masked_step(
        self, env_action: dict, active_mask: np.ndarray
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict]:
        results = self.call("curriculum_masked_step", env_action, active_mask)
        observations, rewards, terminations, truncations = [], [], [], []
        infos: dict = {}
        for env_idx, r in enumerate(results):
            observations.append(r[0])
            rewards.append(r[1])
            terminations.append(r[2])
            truncations.append(r[3])
            infos = self._add_info(infos, r[4], env_idx)

        self.observations = self.concatenate_dynamic(observations)
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards, dtype=np.float64),
            np.array(terminations, dtype=np.bool_),
            np.array(truncations, dtype=np.bool_),
            infos,
        )
