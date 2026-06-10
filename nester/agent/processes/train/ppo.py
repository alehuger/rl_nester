import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from nester.agent.agent.utils import CPUProfiler, GPUProfiler, TimingProfiler
from nester.agent.processes.train.base import BaseTrainer


class PPOTrainer(BaseTrainer):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.ppo_cfg = self.global_cfg["training"]["ppo"]
        self.num_envs = self.scale_cfg["num_envs"]
        cl = self.ppo_cfg.get("curriculum") or {}
        self._curriculum_enabled = bool(cl.get("enabled", False))
        self._curriculum_xi = int(cl.get("xi", 10))
        self._curriculum_max_warmup = int(cl.get("max_warmup", 10**9))
        self._episode_length = int(self.env_cfg.get("episode_length", 20))
        self.num_episodes = self._resolve_num_episodes()
        self.num_steps = self.num_episodes * self._episode_length

    def _resolve_num_episodes(self) -> int:
        if self._episode_length <= 0:
            raise ValueError("environment.episode_length must be positive")

        num_episodes = self.scale_cfg.get("num_episodes")
        if num_episodes is None:
            raise ValueError("training.scale.num_episodes is required")
        num_episodes = int(num_episodes)
        if num_episodes <= 0:
            raise ValueError("training.scale.num_episodes must be positive")
        return num_episodes

    def setup(self):
        super().setup()
        if self.is_main:
            print(f"Number of environments (per rank): {self.num_envs}")
            print(
                f"Number of episodes per iteration: {self.num_episodes} "
                f"({self.num_steps} rollout steps)"
            )
            print(f"World size: {self.world_size}")
            if self._curriculum_enabled:
                print(
                    f"Curriculum learning: enabled (xi={self._curriculum_xi}, "
                    f"max_warmup={self._curriculum_max_warmup}, "
                    f"episode_length={self._episode_length})"
                )
        self.gpu_profiler = GPUProfiler(device_index=self.local_rank)
        self.cpu_profiler = CPUProfiler()
        self.timing_profiler = TimingProfiler(sync_cuda=False)
        self.use_bf16 = (
            bool(self.settings_cfg.get("use_bf16", False)) and self.device.type == "cuda"
        )
        if bool(self.settings_cfg.get("use_bf16", False)) and self.device.type != "cuda":
            print("use_bf16=True requested, but CUDA is unavailable. Falling back to FP32.")
    def _autocast_ctx(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _clipped_pg_loss(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """PPO clipped surrogate for one action factor (item / rotation / placement)."""
        clip = self.ppo_cfg["clip_coef"]
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
        return torch.max(pg_loss1, pg_loss2).mean()

    def create_storage_tensor(self):
        return torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def create_storage_tensor_long(self):
        return torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(self.device)

    def _pad_nfp_batch(self, nfp_batch: list[dict]):
        if len(nfp_batch) == 0:
            raise ValueError("Cannot pad empty nfp_batch")
        max_t = max(int(np.asarray(entry["nfp_patches"]).shape[0]) for entry in nfp_batch)
        p = int(np.asarray(nfp_batch[0]["nfp_patches"]).shape[1])
        batch_size = len(nfp_batch)
        padded = np.zeros((batch_size, max_t, p, 2), dtype=np.float32)
        padded_mask = np.zeros((batch_size, max_t), dtype=np.bool_)
        padded_vertex_mask = np.zeros((batch_size, max_t, p), dtype=np.bool_)
        padded_ids = np.zeros((batch_size, max_t), dtype=np.int64)
        for idx, entry in enumerate(nfp_batch):
            seq = np.asarray(entry["nfp_patches"], dtype=np.float32)
            ids = np.asarray(entry["nfp_poly_ids"], dtype=np.int64)
            vertex_mask = np.asarray(
                entry.get("nfp_vertex_mask", np.ones(seq.shape[:2], dtype=np.bool_)),
                dtype=np.bool_,
            )
            length = int(seq.shape[0])
            if length <= 0:
                raise ValueError("NFP patches must be non-empty")
            if seq.shape[1] != p or seq.shape[2] != 2:
                raise ValueError("nfp_patches must be (T, P, 2) with fixed P across batch")
            if ids.shape[0] != length:
                raise ValueError("nfp_patches and nfp_poly_ids length mismatch")
            if vertex_mask.shape != seq.shape[:2]:
                raise ValueError("nfp_vertex_mask must match nfp_patches leading dims")
            padded[idx, :length] = seq
            padded_mask[idx, :length] = True
            padded_vertex_mask[idx, :length] = vertex_mask
            padded_ids[idx, :length] = ids
        return (
            torch.from_numpy(padded).to(self.device),
            torch.from_numpy(padded_mask).to(self.device).bool(),
            torch.from_numpy(padded_vertex_mask).to(self.device).bool(),
            torch.from_numpy(padded_ids).to(self.device).long(),
        )

    def _curriculum_schedule(self, iteration: int) -> tuple[int, int]:
        raw_schedule_K = iteration // self._curriculum_xi
        K_max_schedule = min(raw_schedule_K, self._curriculum_max_warmup)
        return raw_schedule_K, K_max_schedule

    def _curriculum_warmup_lengths(
        self,
        K_max_schedule: int,
        *,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        if K_max_schedule <= 0:
            return np.zeros(self.num_envs, dtype=np.int64)
        k_per_env = np.full(self.num_envs, K_max_schedule, dtype=np.int64)
        return np.where(active_mask, k_per_env, 0).astype(np.int64)

    def _run_curriculum_warmup(
        self,
        next_obs,
        *,
        iteration: int,
        active_mask: np.ndarray | None = None,
    ):
        _raw_schedule_K, K_max_schedule = self._curriculum_schedule(iteration)
        if active_mask is None:
            active_mask = np.ones(self.num_envs, dtype=np.bool_)
        else:
            active_mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
            if active_mask.shape[0] != self.num_envs:
                raise ValueError("active_mask must have one entry per environment")

        if not np.any(active_mask):
            return next_obs, {
                "warmup_k": [],
                "init_occupancy": [],
                "best_warmup_occupancy": [],
            }

        k_per_env = self._curriculum_warmup_lengths(K_max_schedule, active_mask=active_mask)
        k_actual_max = int(k_per_env.max()) if k_per_env.size else 0

        if k_actual_max > 0:
            for t in range(k_actual_max):
                step_active_mask = np.logical_and(active_mask, t < k_per_env)
                with torch.no_grad():
                    with self._autocast_ctx():
                        stage1 = self.agent.get_action_and_value(
                            next_obs, nfp_patches=None, nfp_mask=None
                        )
                        encoded = stage1["context"]
                        stage1_action = stage1["stage1"]["action"]
                        nfp_list = self.envs.compute_nfp(
                            stage1_action["polygon_id"].cpu().numpy(),
                            stage1_action["theta_abs"].cpu().numpy(),
                        )
                        nfp_pad, nfp_mask, nfp_vertex_mask, nfp_ids = self._pad_nfp_batch(
                            nfp_list
                        )
                        rollout_out = self.agent.get_action_and_value_from_encoded(
                            encoded,
                            action={"stage1": stage1_action},
                            nfp_patches=nfp_pad,
                            nfp_mask=nfp_mask,
                            nfp_vertex_mask=nfp_vertex_mask,
                            nfp_poly_ids=nfp_ids,
                        )
                        action_dict = rollout_out["action"]
                        env_action = {
                            "polygon_id": action_dict["polygon_id"].cpu().numpy(),
                            "translation": action_dict["translation"].cpu().numpy(),
                            "theta": action_dict["theta"].cpu().numpy(),
                        }
                        next_obs, _r, _term, _trunc, _info = self.envs.curriculum_masked_step(
                            env_action, step_active_mask
                        )

        init_occ_all, best_warmup_all = self.envs.curriculum_warmup_snapshot()
        next_obs = self.envs.curriculum_start_training_phase(active_mask)
        active_idxs = np.flatnonzero(active_mask)
        warmup_k = k_per_env[active_idxs].astype(np.float64)
        init_occupancy = init_occ_all[active_idxs]
        best_warmup_occupancy = best_warmup_all[active_idxs]

        return next_obs, {
            "warmup_k": warmup_k.tolist(),
            "init_occupancy": init_occupancy.tolist(),
            "best_warmup_occupancy": best_warmup_occupancy.tolist(),
        }

    def _accumulate_curriculum_warmup_stats(
        self,
        warmup_k: list[float],
        init_occupancy: list[float],
        best_warmup_occupancy: list[float],
        stats: dict,
    ) -> None:
        warmup_k.extend(stats.get("warmup_k", []))
        init_occupancy.extend(stats.get("init_occupancy", []))
        best_warmup_occupancy.extend(stats.get("best_warmup_occupancy", []))

    def train(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = self.ppo_cfg["minibatch_size"]
        global_batch_size = self.batch_size * self.world_size
        num_iterations = self.scale_cfg["total_timesteps"] // global_batch_size
        num_iterations = num_iterations // 100 * 100
        start_iteration = int(getattr(self, "resume_iteration", 0)) + 1
        if start_iteration > num_iterations and self.is_main:
            print(
                f"Warning: resume starts past num_iterations ({start_iteration} > {num_iterations}); "
                "no PPO updates will run (raise total_timesteps or use a newer ckpt)."
            )
        if self.is_main:
            print(f"Number of iterations: {num_iterations} (starting at {start_iteration})")
            print(
                f"Global batch size: {global_batch_size} ({self.batch_size} per rank x {self.world_size} ranks)"
            )

        best_episodic_return = -np.inf

        actions_polygon_id = self.create_storage_tensor_long()
        actions_theta_idx = self.create_storage_tensor_long()
        actions_nfp_patch_idx = self.create_storage_tensor_long()
        actions_nfp_subregion_idx = self.create_storage_tensor_long()

        logprobs = self.create_storage_tensor()
        logprobs_stage1_item = self.create_storage_tensor()
        logprobs_stage1_rotation = self.create_storage_tensor()
        logprobs_stage2_place = self.create_storage_tensor()
        rewards = self.create_storage_tensor()
        dones = self.create_storage_tensor()
        values = self.create_storage_tensor()

        global_step = self.global_step_init
        start_time = time.time()
        seed = self.settings_cfg.get("seed", 1)
        if not self._curriculum_enabled:
            next_obs, _ = self.envs.reset(seed=seed)
        else:
            next_obs = None

        next_done = torch.zeros(self.num_envs).to(self.device)
        running_episodic_return = torch.zeros(self.num_envs).to(self.device)

        moving_max_adv = None

        for iteration in range(start_iteration, num_iterations + 1):
            obs_list = []
            nfp_data_list = []
            curriculum_warmup_k: list[float] = []
            curriculum_init_occ: list[float] = []
            curriculum_best_warmup_occ: list[float] = []
            self.timing_profiler.reset()

            if self.is_main:
                print(f"\n #### Iteration {iteration} ####")

            if self._curriculum_enabled:
                reset_seed = seed + self.rank * 1_000_003 + iteration * 17_917
                next_obs, _ = self.envs.reset(seed=reset_seed)
                next_done = torch.zeros(self.num_envs).to(self.device)
                running_episodic_return = torch.zeros(self.num_envs).to(self.device)
                next_obs, curriculum_stats = self._run_curriculum_warmup(
                    next_obs,
                    iteration=iteration,
                )
                self._accumulate_curriculum_warmup_stats(
                    curriculum_warmup_k,
                    curriculum_init_occ,
                    curriculum_best_warmup_occ,
                    curriculum_stats,
                )

            if self.optimizer_cfg.get("anneal_lr", False):
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * float(self.optimizer_cfg["learning_rate"])
                self.optimizer.param_groups[0]["lr"] = lrnow

            # ROLLOUT STEP
            for step in range(0, self.num_steps):
                global_step += self.num_envs * self.world_size
                # next_obs is a dict where each key is a list of N elements
                # as many key as max polygons, N is the number of environments
                obs_list.append(next_obs)
                dones[step] = next_done

                with self.timing_profiler.timer("rollout_inference"):
                    with torch.no_grad():
                        with self._autocast_ctx():
                            stage1 = self.agent.get_action_and_value(
                                next_obs, nfp_patches=None, nfp_mask=None
                            )
                            encoded = stage1["context"]
                            stage1_action = stage1["stage1"]["action"]
                            values[step] = stage1["value"].flatten()
                            nfp_list = self.envs.compute_nfp(
                                stage1_action["polygon_id"].cpu().numpy(),
                                stage1_action["theta_abs"].cpu().numpy(),
                            )
                            nfp_pad, nfp_mask, nfp_vertex_mask, nfp_ids = self._pad_nfp_batch(
                                nfp_list
                            )
                            rollout_out = self.agent.get_action_and_value_from_encoded(
                                encoded,
                                action={"stage1": stage1_action},
                                nfp_patches=nfp_pad,
                                nfp_mask=nfp_mask,
                                nfp_vertex_mask=nfp_vertex_mask,
                                nfp_poly_ids=nfp_ids,
                            )

                action_dict = rollout_out["action"]
                actions_polygon_id[step] = action_dict["polygon_id"]
                actions_theta_idx[step] = action_dict["theta_idx"]
                actions_nfp_patch_idx[step] = action_dict["nfp_patch_idx"]
                actions_nfp_subregion_idx[step] = action_dict["nfp_subregion_idx"]
                nfp_data_list.extend(nfp_list)
                logprobs[step] = rollout_out["joint_logprob"]
                logprobs_stage1_item[step] = rollout_out["logprob_components"]["stage1_item"]
                logprobs_stage1_rotation[step] = rollout_out["logprob_components"][
                    "stage1_rotation"
                ]
                logprobs_stage2_place[step] = rollout_out["logprob_components"]["stage2_place"]

                env_action = {
                    "polygon_id": action_dict["polygon_id"].cpu().numpy(),
                    "translation": action_dict["translation"].cpu().numpy(),
                    "theta": action_dict["theta"].cpu().numpy(),
                }

                with self.timing_profiler.timer("env_step"):
                    self.cpu_profiler.start()

                    next_obs, reward, terminated, truncated, info = self.envs.step(env_action)

                # Gymnasium semantics:
                # - `terminated`: task completion / failure (we want to use only this for GAE bootstrapping)
                # - `truncated`: time-limit cutoff (we still treat it as an episode boundary for logging/returns)
                episode_done = np.logical_or(terminated, truncated)
                episode_done_mask = torch.as_tensor(
                    episode_done, dtype=torch.bool, device=self.device
                )
                terminal_mask = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
                reward_tensor = torch.tensor(reward).to(self.device).view(-1)
                rewards[step] = reward_tensor

                # Store only `terminated` for GAE bootstrapping masks.
                next_done = terminal_mask

                running_episodic_return += reward_tensor

                if torch.any(episode_done_mask):
                    # Extract final reward
                    if "_final_info" in info and info["_final_info"].any():
                        final_info_dict = info["final_info"]

                        final_returns = running_episodic_return[episode_done_mask]
                        episodic_return = final_returns.cpu().numpy()
                        episodic_return_mean = np.mean(episodic_return)
                        running_episodic_return[episode_done_mask] = 0.0  # Reset for finished envs

                        increase_occupancy_ratio = final_info_dict["increase_occupancy_ratio"]
                        final_occupancy_ratio = final_info_dict["final_occupancy_ratio"]
                        best_occupancy_ratio = final_info_dict["best_occupancy_ratio"]
                        validation_score = final_info_dict["best_validation_score"]
                        final_reward_cumulative = final_info_dict["final_reward_cumulative"]
                        nonzero_reward_ratio = final_info_dict.get("nonzero_reward_ratio")

                        self.logger.log_rollout_metrics(
                            increase_occupancy_ratio,
                            final_occupancy_ratio,
                            global_step,
                            episodic_return=episodic_return,
                            validation_score=validation_score,
                            best_occupancy_ratio=best_occupancy_ratio,
                            final_reward_cumulative=final_reward_cumulative,
                            nonzero_reward_ratio=nonzero_reward_ratio,
                        )

                    # Save
                    if episodic_return_mean > best_episodic_return and iteration > 10:
                        best_episodic_return = episodic_return_mean
                        self.save_model(iteration, global_step, best=True)

                    if self._curriculum_enabled and step < self.num_steps - 1:
                        done_mask = np.asarray(episode_done, dtype=np.bool_).reshape(-1)
                        next_obs, warmup_stats = self._run_curriculum_warmup(
                            next_obs,
                            iteration=iteration,
                            active_mask=done_mask,
                        )
                        self._accumulate_curriculum_warmup_stats(
                            curriculum_warmup_k,
                            curriculum_init_occ,
                            curriculum_best_warmup_occ,
                            warmup_stats,
                        )

                if (
                    self.settings_cfg.get("save_freq", 0) > 0
                    and iteration > 0
                    and iteration % self.settings_cfg["save_freq"] == 0
                    and step == self.num_steps - 1
                ):
                    self.save_model(iteration, global_step, best=False)

            if self._curriculum_enabled and self.is_main:
                self.logger.log_curriculum_metrics(
                    global_step,
                    warmup_steps=(
                        float(np.mean(curriculum_warmup_k)) if curriculum_warmup_k else 0.0
                    ),
                    init_occupancy=(
                        float(np.mean(curriculum_init_occ)) if curriculum_init_occ else float("nan")
                    ),
                    best_warmup_occupancy=(
                        float(np.mean(curriculum_best_warmup_occ))
                        if curriculum_best_warmup_occ
                        else float("nan")
                    ),
                )

            cpu_stats = self.cpu_profiler.stop()

            # ADVANTAGES AND RETURNS STEP
            with torch.no_grad():
                with self._autocast_ctx():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.ppo_cfg["gamma"] * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.ppo_cfg["gamma"]
                        * self.ppo_cfg["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values

            # organize batch data
            # obs_list: num_steps * num_envs * num_polygons, num_points
            # b_obs: (num_steps * num_envs) * num_polygons, num_points
            b_obs = [obs_list[i][j] for i in range(self.num_steps) for j in range(self.num_envs)]

            b_logprobs = logprobs.reshape(-1)
            b_logprobs_stage1_item = logprobs_stage1_item.reshape(-1)
            b_logprobs_stage1_rotation = logprobs_stage1_rotation.reshape(-1)
            b_logprobs_stage2_place = logprobs_stage2_place.reshape(-1)
            b_actions_polygon_id = actions_polygon_id.reshape(-1)
            b_actions_theta_idx = actions_theta_idx.reshape(-1)
            b_actions_nfp_patch_idx = actions_nfp_patch_idx.reshape(-1)
            b_actions_nfp_subregion_idx = actions_nfp_subregion_idx.reshape(-1)
            b_nfp_data = nfp_data_list
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            ### Advantage Filtering ###
            current_max_adv = b_advantages.abs().max().item()

            if moving_max_adv is None:
                moving_max_adv = current_max_adv
            else:
                beta = self.ppo_cfg.get("adv_filter_beta", 0.25)
                moving_max_adv = beta * current_max_adv + (1 - beta) * moving_max_adv

            adv_after = int(self.ppo_cfg.get("adv_filter_start_after_iteration", 50))
            if iteration > adv_after:
                eta = self.ppo_cfg.get("eta", 0.01)
                filter_threshold = eta * moving_max_adv
                valid_mask = b_advantages.abs() >= filter_threshold
            else:
                valid_mask = torch.ones_like(b_advantages, dtype=torch.bool)
            b_inds = torch.where(valid_mask)[0].cpu().numpy()
            filtered_batch_size = len(b_inds)

            # Ensure every rank executes the same number of full minibatches.
            # This prevents distributed collective call-count mismatches (Gloo abort)
            # and keeps batch sizes stable across ranks.
            num_full_minibatches_local = filtered_batch_size // self.minibatch_size
            num_full_minibatches_tensor = torch.tensor(
                [num_full_minibatches_local], device=self.device, dtype=torch.long
            )
            if self.world_size > 1 and dist.is_initialized():
                dist.all_reduce(num_full_minibatches_tensor, op=dist.ReduceOp.MIN)
            num_full_minibatches_global = int(num_full_minibatches_tensor.item())
            usable_filtered_batch_size = num_full_minibatches_global * self.minibatch_size

            if self.is_main and iteration > adv_after:
                filtered_pct = 100 * (1 - filtered_batch_size / self.batch_size)
                print(
                    f"Advantage Filtering: Filtered out {filtered_pct:.1f}% of samples. Retained {filtered_batch_size}/{self.batch_size}."
                )

            # Batch usage accounting (per-rank), logged on rank 0 as means over ranks.
            local_usable_before_sync = num_full_minibatches_local * self.minibatch_size
            local_drop_adv = self.batch_size - filtered_batch_size
            local_drop_incomplete = filtered_batch_size - local_usable_before_sync
            local_drop_global_sync = local_usable_before_sync - usable_filtered_batch_size
            local_total_unused = self.batch_size - usable_filtered_batch_size

            batch_drop_stats = torch.tensor(
                [
                    float(local_drop_adv),
                    float(local_drop_incomplete),
                    float(local_drop_global_sync),
                    float(local_total_unused),
                ],
                device=self.device,
                dtype=torch.float32,
            )
            if self.world_size > 1 and dist.is_initialized():
                dist.all_reduce(batch_drop_stats, op=dist.ReduceOp.SUM)
                batch_drop_stats /= self.world_size

            if self.is_main:
                mean_drop_adv = float(batch_drop_stats[0].item())
                mean_drop_incomplete = float(batch_drop_stats[1].item())
                mean_drop_global_sync = float(batch_drop_stats[2].item())
                mean_total_unused = float(batch_drop_stats[3].item())
                trained_per_gpu = self.batch_size - mean_total_unused
                denom = float(self.batch_size) if self.batch_size > 0 else 1.0
                print(
                    f"[batch_drop] iter={iteration} world_size={self.world_size} "
                    f"(mean over ranks, vs local batch_size={self.batch_size} per GPU)"
                )
                print(
                    f"advantage_filter: {100.0 * mean_drop_adv / denom:.2f}% "
                    f"(~{int(round(mean_drop_adv))} samples/GPU)"
                )
                print(
                    f"incomplete_minibatch: {100.0 * mean_drop_incomplete / denom:.2f}% "
                    f"(~{int(round(mean_drop_incomplete))} samples/GPU)"
                )
                print(
                    f"global_min_sync: {100.0 * mean_drop_global_sync / denom:.2f}% "
                    f"(~{int(round(mean_drop_global_sync))} samples/GPU)"
                )
                print(
                    f"total_unused: {100.0 * mean_total_unused / denom:.2f}% "
                    f"(~{int(round(mean_total_unused))} samples/GPU; "
                    f"trained {int(round(trained_per_gpu))}/GPU)"
                )

            clipfracs = []
            mb_metric_sums: dict[str, float] = {}
            mb_metric_count = 0
            self.gpu_profiler.start()

            for epoch in range(self.ppo_cfg["update_epochs"]):
                if self.is_main:
                    print(f"Epoch {epoch}")
                np.random.shuffle(b_inds)
                for start in range(0, usable_filtered_batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Convert the list of single-env observations to the dict format the agent expects
                    mb_obs_list = [b_obs[i] for i in mb_inds]
                    mb_obs_dict = {i: obs for i, obs in enumerate(mb_obs_list)}

                    mb_action_stage1 = {
                        "polygon_id": b_actions_polygon_id[mb_inds],
                        "theta_idx": b_actions_theta_idx[mb_inds],
                    }
                    mb_action_stage2 = {
                        "nfp_patch_idx": b_actions_nfp_patch_idx[mb_inds],
                        "nfp_subregion_idx": b_actions_nfp_subregion_idx[mb_inds],
                    }
                    mb_nfp_batch = [b_nfp_data[i] for i in mb_inds]
                    mb_nfp_pad, mb_nfp_mask, mb_nfp_vertex_mask, mb_nfp_poly_ids = (
                        self._pad_nfp_batch(mb_nfp_batch)
                    )

                    with self.timing_profiler.timer("forward_pass"):
                        with self._autocast_ctx():
                            eval_out = self.agent.evaluate_actions(
                                mb_obs_dict,
                                action_1=mb_action_stage1,
                                action_2=mb_action_stage2,
                                nfp_patches=mb_nfp_pad,
                                nfp_mask=mb_nfp_mask,
                                nfp_vertex_mask=mb_nfp_vertex_mask,
                                nfp_poly_ids=mb_nfp_poly_ids,
                            )
                            newlogprob = eval_out["joint_logprob"]
                            newvalue = eval_out["value"]
                    newlogprob = newlogprob.float()
                    newvalue = newvalue.float()

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    logratio_stage1_item = (
                        eval_out["stage1"]["logprob_components"]["item"]
                        - b_logprobs_stage1_item[mb_inds]
                    ).float()
                    ratio_stage1_item = logratio_stage1_item.exp()
                    logratio_stage1_rotation = (
                        eval_out["stage1"]["logprob_components"]["rotation"]
                        - b_logprobs_stage1_rotation[mb_inds]
                    ).float()
                    ratio_stage1_rotation = logratio_stage1_rotation.exp()
                    logratio_stage2_place = (
                        eval_out["stage2"]["logprob_components"]["place"]
                        - b_logprobs_stage2_place[mb_inds]
                    ).float()
                    ratio_stage2_place = logratio_stage2_place.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        approx_kl_stage1_item = (
                            (ratio_stage1_item - 1) - logratio_stage1_item
                        ).mean()
                        approx_kl_stage1_rotation = (
                            (ratio_stage1_rotation - 1) - logratio_stage1_rotation
                        ).mean()
                        approx_kl_stage2_place = (
                            (ratio_stage2_place - 1) - logratio_stage2_place
                        ).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.ppo_cfg["clip_coef"]).float().mean().item()
                        ]
                        clipfrac_stage1_item = (
                            ((ratio_stage1_item - 1.0).abs() > self.ppo_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        )
                        clipfrac_stage1_rotation = (
                            ((ratio_stage1_rotation - 1.0).abs() > self.ppo_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        )
                        clipfrac_stage2_place = (
                            ((ratio_stage2_place - 1.0).abs() > self.ppo_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if self.ppo_cfg["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Factorized policy: clip item, rotation, and placement separately.
                    # A single joint ratio clips whenever item or placement moves, which
                    # zeroes rotation gradients even when rotation log-probs change.
                    pg_loss = (
                        self._clipped_pg_loss(ratio_stage1_item, mb_advantages)
                        + self._clipped_pg_loss(ratio_stage1_rotation, mb_advantages)
                        + self._clipped_pg_loss(ratio_stage2_place, mb_advantages)
                    )

                    newvalue = newvalue.view(-1)
                    if self.ppo_cfg["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.ppo_cfg["clip_coef"],
                            self.ppo_cfg["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    ent_coef_item = self.ppo_cfg.get("ent_coef_item", 0.05)
                    ent_coef_rotation = self.ppo_cfg.get("ent_coef_rotation", 0.01)
                    ent_coef_place = self.ppo_cfg.get("ent_coef_place", 0.01)
                    ent_item = eval_out["stage1"]["entropy_components"]["item"].float().mean()
                    ent_rotation = (
                        eval_out["stage1"]["entropy_components"]["rotation"].float().mean()
                    )
                    ent_place = eval_out["stage2"]["entropy_components"]["place"].float().mean()
                    entropy_loss = ent_item + ent_rotation + ent_place
                    loss = (
                        pg_loss
                        - ent_coef_item * ent_item
                        - ent_coef_rotation * ent_rotation
                        - ent_coef_place * ent_place
                        + v_loss * self.ppo_cfg["vf_coef"]
                    )

                    with self.timing_profiler.timer("backward_pass"):
                        self.optimizer.zero_grad()
                        loss.backward()
                        if self.world_size > 1:
                            grads = [p.grad for p in self.agent.parameters() if p.grad is not None]
                            flat = torch.cat([g.reshape(-1) for g in grads]).cpu()
                            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                            flat /= self.world_size
                            offset = 0
                            for g in grads:
                                n = g.numel()
                                g.copy_(flat[offset : offset + n].reshape(g.shape).to(g.device))
                                offset += n
                        torch.nn.utils.clip_grad_norm_(
                            self.agent.parameters(), self.ppo_cfg["max_grad_norm"]
                        )
                        self.optimizer.step()

                    mb_metric_count += 1
                    mb_metric_sums["v_loss"] = mb_metric_sums.get("v_loss", 0.0) + float(
                        v_loss.item()
                    )
                    mb_metric_sums["pg_loss"] = mb_metric_sums.get("pg_loss", 0.0) + float(
                        pg_loss.item()
                    )
                    mb_metric_sums["entropy_loss"] = mb_metric_sums.get(
                        "entropy_loss", 0.0
                    ) + float(entropy_loss.item())
                    mb_metric_sums["entropy_stage1_item"] = mb_metric_sums.get(
                        "entropy_stage1_item", 0.0
                    ) + float(
                        eval_out["stage1"]["entropy_components"]["item"].mean().item()
                    )
                    mb_metric_sums["entropy_stage1_rotation"] = mb_metric_sums.get(
                        "entropy_stage1_rotation", 0.0
                    ) + float(
                        eval_out["stage1"]["entropy_components"]["rotation"].mean().item()
                    )
                    mb_metric_sums["entropy_stage2_place"] = mb_metric_sums.get(
                        "entropy_stage2_place", 0.0
                    ) + float(
                        eval_out["stage2"]["entropy_components"]["place"].mean().item()
                    )
                    mb_metric_sums["old_approx_kl"] = mb_metric_sums.get(
                        "old_approx_kl", 0.0
                    ) + float(old_approx_kl.item())
                    mb_metric_sums["approx_kl"] = mb_metric_sums.get("approx_kl", 0.0) + float(
                        approx_kl.item()
                    )
                    mb_metric_sums["approx_kl_stage1_item"] = mb_metric_sums.get(
                        "approx_kl_stage1_item", 0.0
                    ) + float(approx_kl_stage1_item.item())
                    mb_metric_sums["approx_kl_stage1_rotation"] = mb_metric_sums.get(
                        "approx_kl_stage1_rotation", 0.0
                    ) + float(approx_kl_stage1_rotation.item())
                    mb_metric_sums["approx_kl_stage2_place"] = mb_metric_sums.get(
                        "approx_kl_stage2_place", 0.0
                    ) + float(approx_kl_stage2_place.item())
                    mb_metric_sums["clipfrac_stage1_item"] = mb_metric_sums.get(
                        "clipfrac_stage1_item", 0.0
                    ) + float(clipfrac_stage1_item)
                    mb_metric_sums["clipfrac_stage1_rotation"] = mb_metric_sums.get(
                        "clipfrac_stage1_rotation", 0.0
                    ) + float(clipfrac_stage1_rotation)
                    mb_metric_sums["clipfrac_stage2_place"] = mb_metric_sums.get(
                        "clipfrac_stage2_place", 0.0
                    ) + float(clipfrac_stage2_place)
                    mb_metric_sums["logit_abs_max_item"] = max(
                        mb_metric_sums.get("logit_abs_max_item", 0.0),
                        float(eval_out["stage1"]["logit_abs_max"]["item"]),
                    )
                    mb_metric_sums["logit_abs_max_rotation"] = max(
                        mb_metric_sums.get("logit_abs_max_rotation", 0.0),
                        float(eval_out["stage1"]["logit_abs_max"]["rotation"]),
                    )
                    mb_metric_sums["logit_abs_max_place"] = max(
                        mb_metric_sums.get("logit_abs_max_place", 0.0),
                        float(eval_out["stage2"]["logit_abs_max"]["place"]),
                    )

                if self.ppo_cfg["target_kl"] is not None:
                    if self.world_size > 1:
                        kl_tensor = torch.tensor([approx_kl.item()])
                        dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
                        kl_tensor /= self.world_size
                        approx_kl_global = kl_tensor.item()
                    else:
                        approx_kl_global = approx_kl
                    if approx_kl_global > self.ppo_cfg["target_kl"]:
                        break

            adv_filter_retained_ratio = filtered_batch_size / self.batch_size
            adv_filtered_pct = 100.0 * (1.0 - adv_filter_retained_ratio)
            if mb_metric_count > 0:
                avg = {k: v / mb_metric_count for k, v in mb_metric_sums.items()}
            else:
                avg = {k: 0.0 for k in mb_metric_sums}
            update_metrics = {
                "v_loss": avg.get("v_loss", 0.0),
                "pg_loss": avg.get("pg_loss", 0.0),
                "entropy_loss": avg.get("entropy_loss", 0.0),
                "entropy_stage1_item": avg.get("entropy_stage1_item", 0.0),
                "entropy_stage1_rotation": avg.get("entropy_stage1_rotation", 0.0),
                "entropy_stage2_place": avg.get("entropy_stage2_place", 0.0),
                "old_approx_kl": avg.get("old_approx_kl", 0.0),
                "approx_kl": avg.get("approx_kl", 0.0),
                "approx_kl_stage1_item": avg.get("approx_kl_stage1_item", 0.0),
                "approx_kl_stage1_rotation": avg.get("approx_kl_stage1_rotation", 0.0),
                "approx_kl_stage2_place": avg.get("approx_kl_stage2_place", 0.0),
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "clipfrac_stage1_item": avg.get("clipfrac_stage1_item", 0.0),
                "clipfrac_stage1_rotation": avg.get("clipfrac_stage1_rotation", 0.0),
                "clipfrac_stage2_place": avg.get("clipfrac_stage2_place", 0.0),
                "global_step": global_step,
                "adv_filtered_pct": adv_filtered_pct,
                "adv_filter_retained_ratio": adv_filter_retained_ratio,
                "logit_abs_max_item": avg.get("logit_abs_max_item", 0.0),
                "logit_abs_max_rotation": avg.get("logit_abs_max_rotation", 0.0),
                "logit_abs_max_place": avg.get("logit_abs_max_place", 0.0),
            }

            self.logger.log_end_update(**update_metrics)
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            lrnow = self.optimizer.param_groups[0]["lr"]
            sps = int(global_step / (time.time() - start_time))
            gpu_stats = self.gpu_profiler.stop()
            timing_stats = self.timing_profiler.get_stats(self.num_steps, self.num_envs)

            end_update_metrics = {
                "sps": sps,
                "explained_var": explained_var,
                "lrnow": lrnow,
                "global_step": global_step,
                "gpu_stats": gpu_stats,
                "cpu_stats": cpu_stats,
                "timing_stats": timing_stats,
            }

            self.logger.log_training_metrics(**end_update_metrics)
            if iteration % 20 == 0:
                cloud_cfg = self.global_cfg.get("cloud", {})
                bucket_name = cloud_cfg.get("bucket")
                if bucket_name:
                    self.upload_to_bucket(bucket_name, iteration)

        self._close_shared_data()
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
