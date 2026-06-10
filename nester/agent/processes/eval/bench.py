"""Bench evaluation pipeline (`EvalPipeline`)."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gymnasium.vector import AutoresetMode
from tqdm.auto import tqdm

from nester import PATH_TO_PROJECT
from nester.agent.agent.agent import PotRLAgent
from nester.agent.agent.utils import (
    apply_env_model_overrides,
    load_agent_state_dict_merged,
    make_inference_env,
)
from nester.agent.processes.eval.layout import (
    bench_eval_budget_strat_tag,
    bench_eval_ckpt_subdir_name,
    bench_eval_name_segment,
    bench_eval_shard_triplet,
    elite_seeds_from_completed,
    exploit_p_normalized,
    instance_indices_for_shard,
    sampling_flags,
)
from nester.agent.processes.eval.plots import (
    save_best_occ_rollout_init_trace_png,
    save_best_occ_vs_log_likelihood_png,
)
from nester.env.utils.sharing_memory import SharedBenchData
from nester.env.utils.vec_envs import DynamicAsyncVectorEnv

# Re-export layout helpers for run/cloud imports that use `bench` as entry module.
from nester.agent.processes.eval.layout import (  # noqa: F401
    bench_checkpoint_step_count,
    bench_checkpoint_steps_slug,
    bench_eval_budget_strat_tag_from_config,
    bench_eval_ckpt_job_token,
    bench_eval_run_path_segments,
)

_BEST_OCC_LOGPROB_GAMMA = 0.8
_BEST_OCC_LOGPROB_LOOKBACK = 6
_JSON_PRETTY_MAX_EPISODES = 2000


def gamma_discounted_logprob_before_peak(
    logprobs_by_step: list[float],
    peak_step: int,
    *,
    gamma: float = _BEST_OCC_LOGPROB_GAMMA,
    lookback: int = _BEST_OCC_LOGPROB_LOOKBACK,
) -> float | None:
    """γ-weighted mean of joint log-probs for up to ``lookback`` actions before ``peak_step``.

    ``peak_step`` is the 1-based env step index where best occupancy was achieved; the
    action at ``peak_step`` is excluded. The most recent pre-peak action uses γ⁰.
    """
    if peak_step <= 1 or not logprobs_by_step:
        return None
    indices: list[int] = []
    for k in range(1, lookback + 1):
        env_step = peak_step - k
        if env_step < 1:
            break
        idx = env_step - 1
        if idx < len(logprobs_by_step):
            indices.append(idx)
    if not indices:
        return None
    weights = np.array([gamma**i for i in range(len(indices))], dtype=np.float64)
    vals = np.array([logprobs_by_step[i] for i in indices], dtype=np.float64)
    return float(np.dot(weights, vals) / weights.sum())


def _bench_eval_resolve_torch_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    lr_raw = os.environ.get("LOCAL_RANK")
    if lr_raw is not None and str(lr_raw).strip() != "":
        lr = int(lr_raw)
        torch.cuda.set_device(lr)
        return torch.device(f"cuda:{lr}")
    return torch.device("cuda")


def _use_tqdm_progress_bar():
    if sys.stderr.isatty():
        return True
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        return get_ipython() is not None
    except ImportError:
        return False


def _panel_dict_to_placed_list(panel_dict: dict) -> list:
    placed = []
    for k in sorted(panel_dict.keys()):
        if isinstance(k, str) and "_" in str(k):
            continue
        arr = np.asarray(panel_dict[k], dtype=np.float32)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            placed.append(arr)
    return placed



class EvalPipeline:
    def __init__(self, config):
        self.config = config
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.settings_config = config["settings"]
        self._shard_rank, self._shard_world_size, self._local_rank = bench_eval_shard_triplet()
        self.instance_size = 1
        self.global_num_instances = 1
        self.instance_indices: list[int] = []
        self.batch_size = self.settings_config.get("batch_size")
        self.device = _bench_eval_resolve_torch_device()
        self.use_bf16 = bool(
            self.settings_config.get("use_bf16", self.model_config.get("use_bf16", False))
        )
        self.num_episodes = self.settings_config.get("num_episodes")
        self.episode_length = int(self.settings_config.get("episode_length", 1))

        self._nfp_pad_buf_max_t: int = 0
        self._nfp_pad_buf_patch_size: int = 0
        self._nfp_pad_buf_batch_size: int = 0
        self._nfp_padded_np: np.ndarray | None = None
        self._nfp_padded_mask_np: np.ndarray | None = None
        self._nfp_vertex_mask_np: np.ndarray | None = None
        self._nfp_ids_np: np.ndarray | None = None
        self._nfp_padded_cpu_t: torch.Tensor | None = None
        self._nfp_padded_mask_cpu_t: torch.Tensor | None = None
        self._nfp_vertex_mask_cpu_t: torch.Tensor | None = None
        self._nfp_ids_cpu_t: torch.Tensor | None = None
        self._nfp_padded_gpu_t: torch.Tensor | None = None
        self._nfp_padded_mask_gpu_t: torch.Tensor | None = None
        self._nfp_vertex_mask_gpu_t: torch.Tensor | None = None
        self._nfp_ids_gpu_t: torch.Tensor | None = None

        self.init_strategy = str(self.settings_config.get("type", "random")).lower()
        self.exploit_p = float(self.settings_config.get("exploit_p", 30.0))
        self.elite_frac = float(self.settings_config.get("elite_frac", 0.01))
        _plot_top = self.settings_config.get("plot_rollout_top_frac")
        self.plot_rollout_top_frac = (
            float(_plot_top) if _plot_top is not None else float(self.elite_frac)
        )
        _plot_occ_ll = self.settings_config.get("plot_occ_likelihood_top_frac")
        self.plot_occ_likelihood_top_frac = float(_plot_occ_ll) if _plot_occ_ll is not None else 1.0
        self.max_occ_likelihood_plot_points = int(
            self.settings_config.get("max_occ_likelihood_plot_points", 5000)
        )
        self.warm_start_p = float(self.settings_config.get("warm_start_p", 0.0))
        self.critic_elite_coef = float(self.settings_config.get("critic_elite_coef", 0.0))
        self.use_critic_elite_ranking = bool(
            self.settings_config.get("use_critic_elite_ranking", False)
        )
        self.sampling_mode = str(self.settings_config.get("sampling_mode", "both")).lower()
        _policy_stoch, self.placement_varied = sampling_flags(self.sampling_mode)
        del _policy_stoch
        if not self.placement_varied:
            self.init_strategy = "random"
            self.warm_start_p = 0.0

        self._bench_budget_strat_tag = self._compute_bench_budget_strat_tag()

        if not self.use_bf16:
            torch.set_float32_matmul_precision("high")

        shard_note = (
            f" [shard rank {self._shard_rank + 1}/{self._shard_world_size}, "
            f"CUDA local_rank={self._local_rank}]"
            if self._shard_world_size > 1
            else ""
        )
        print(
            f"Evaluating model {self.model_config.get('model_folder')} on "
            f"{self.data_config.get('num_instances', '?')} instances (global) "
            f"with {self.num_episodes} episodes per instance{shard_note}",
            flush=True,
        )
        strat_eff = self.init_strategy if self.placement_varied else "random"
        print(
            f"Inference init: init_strategy={strat_eff}, sampling_mode={self.sampling_mode}, "
            f"exploit_p={self.exploit_p}, elite_frac={self.elite_frac}, warm_start_p={self.warm_start_p}, "
            f"use_critic_elite_ranking={self.use_critic_elite_ranking}, critic_elite_coef={self.critic_elite_coef}",
            flush=True,
        )

    @staticmethod
    def _is_compatible_expand_shape(param_name: str, model_shape: tuple, ckpt_shape: tuple) -> bool:
        if (
            param_name == "encoder.mlp_heads.0.weight"
            and len(model_shape) == 2
            and len(ckpt_shape) == 2
            and ckpt_shape[0] == model_shape[0]
            and ckpt_shape[1] + 1 == model_shape[1]
        ):
            return True
        if (
            param_name == "value_network.value_network.0.weight"
            and len(model_shape) == 2
            and len(ckpt_shape) == 2
            and ckpt_shape[0] == model_shape[0]
            and ckpt_shape[1] + 1 == model_shape[1]
        ):
            return True
        return False

    def _report_checkpoint_coverage(self, agent: PotRLAgent, state_dict: dict) -> None:
        model_sd = agent.state_dict()
        model_keys = set(model_sd.keys())
        ckpt_keys = set(state_dict.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        mismatched = []
        compatible_expand = []
        for k in sorted(model_keys & ckpt_keys):
            ms = tuple(model_sd[k].shape)
            cs = tuple(state_dict[k].shape)
            if ms != cs:
                if self._is_compatible_expand_shape(k, ms, cs):
                    compatible_expand.append((k, cs, ms))
                else:
                    mismatched.append((k, cs, ms))

        if not missing and not unexpected and not mismatched:
            print("[ckpt] full parameter match: all model weights present and shape-compatible.")
            if compatible_expand:
                print(
                    f"[ckpt] note: {len(compatible_expand)} legacy tensors use expansion-compatible load."
                )
            return

        print(
            "[ckpt] parameter coverage check:"
            f" missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}"
        )
        if compatible_expand:
            print(f"[ckpt] expansion-compatible tensors: {len(compatible_expand)}")
        if missing:
            print("[ckpt] missing model params not found in checkpoint:")
            for k in missing[:30]:
                print(f"  - {k}")
        if unexpected:
            print("[ckpt] unexpected checkpoint params not used by model:")
            for k in unexpected[:30]:
                print(f"  - {k}")
        if mismatched:
            print("[ckpt] shape mismatches (checkpoint -> model):")
            for k, cs, ms in mismatched[:20]:
                print(f"  - {k}: {cs} -> {ms}")

        if bool(self.settings_config.get("require_full_checkpoint_match", False)):
            raise ValueError("[ckpt] strict mode failed: checkpoint does not fully match model.")

    def load_agent(self):
        mounted = str(self.model_config.get("mounted_folder", "data/runs/"))
        mf = str(self.model_config.get("model_folder", "")).strip("/")
        cf = str(self.model_config.get("checkpoint_file", ""))
        if mounted.startswith("/"):
            model_path = Path(mounted) / mf / "models" / cf
        else:
            model_path = Path(PATH_TO_PROJECT) / mounted / mf / "models" / cf
        try:
            model = torch.load(str(model_path), map_location=self.device, weights_only=False)
        except RuntimeError as exc:
            if "failed finding central directory" in str(exc) or "PytorchStreamReader" in str(exc):
                raise RuntimeError(
                    f"Checkpoint at {model_path} is corrupt or incomplete (often from parallel "
                    f"GCS download on shared storage). Delete it and re-run eval; rank 0 should "
                    f"download alone."
                ) from exc
            raise
        ckpt_cfg = model.get("config") or {}
        env_cfg = ckpt_cfg.get("environment") or {}
        model_kwargs = (ckpt_cfg.get("model") or {}).get("model_kwargs")
        if not model_kwargs:
            raise ValueError("Checkpoint missing config.model.model_kwargs")
        patch = self.data_config.get("nfp_patch_size")
        if patch is None:
            patch = model_kwargs.get("data_patch_size")
        if patch is None:
            patch = env_cfg.get("nfp_patch_size", 10)
        model_kwargs["data_patch_size"] = int(patch)
        enc_cfg = model_kwargs.setdefault("encoder_config", {})
        enc_cfg["data_patch_size"] = int(patch)
        env_for_overrides = dict(env_cfg)
        eval_fsr = self.data_config.get("family_size_range")
        if eval_fsr is not None:
            env_for_overrides["family_size_range"] = eval_fsr
        model_kwargs = apply_env_model_overrides(env_for_overrides, model_kwargs)
        self.agent = PotRLAgent(**model_kwargs, device=self.device)
        agent_state_dict = model.get("agent_state_dict")
        self._report_checkpoint_coverage(self.agent, agent_state_dict)
        load_agent_state_dict_merged(self.agent, agent_state_dict)

    def load_data(self):
        n_all = int(self.data_config.get("num_instances", 0))
        self.global_num_instances = n_all
        self.instance_indices = instance_indices_for_shard(
            n_all, self._shard_rank, self._shard_world_size
        )
        self.instance_size = len(self.instance_indices)
        self._bench_budget_strat_tag = self._compute_bench_budget_strat_tag()
        val_path = self.data_config.get("val_db_local")
        if val_path and not str(val_path).startswith("/"):
            val_path = str(Path(PATH_TO_PROJECT) / val_path)
        self.families_data = np.load(val_path)
        self.families_polygons = self.families_data["polygons"][:n_all]
        self.families_lengths = self.families_data["lengths"][:n_all]
        if self._shard_world_size > 1:
            print(
                f"[shard] rank {self._shard_rank}/{self._shard_world_size}: "
                f"{len(self.instance_indices)} instance(s) "
                f"{self.instance_indices if len(self.instance_indices) <= 20 else self.instance_indices[:20] + ['…']}",
                flush=True,
            )

    def _compute_bench_budget_strat_tag(self) -> str:
        n_inst = int(self.data_config.get("num_instances", self.instance_size))
        return bench_eval_budget_strat_tag(
            num_instances=n_inst,
            num_episodes=int(self.num_episodes or 0),
            episode_length=int(self.episode_length),
            init_strategy=str(self.init_strategy),
            exploit_p=float(self.exploit_p),
            warm_start_p=float(self.warm_start_p),
            placement_varied=bool(self.placement_varied),
        )

    def run_eval_instance(self, instance_index, num_episodes) -> float:
        polygons_data = self.families_polygons[instance_index]
        lengths_data = self.families_lengths[instance_index]
        shared_bench_data = SharedBenchData(polygons_data, lengths_data)
        shared_memory_dict = shared_bench_data.create_shared_memories()

        placement_base = int(self.data_config.get("seed_initial_placement_start", 0))
        resample_length = self.data_config.get("resample_length")
        resample_length_nfp = self.data_config.get("resample_length_nfp", resample_length)
        family_size = int(len(lengths_data))

        envs = None
        try:
            envs = DynamicAsyncVectorEnv(
                [
                    make_inference_env(
                        shared_memory_dict=shared_memory_dict,
                        episode_length=self.episode_length,
                        resample_length=resample_length,
                        seed=instance_index,
                        env_index=env_i,
                        family_size=family_size,
                        max_vertices=self.data_config.get("max_vertices"),
                        nfp_patch_size=self.data_config.get("nfp_patch_size"),
                        max_reward=self.data_config.get("max_reward"),
                        placement_seed=placement_base + instance_index * 1_000_003 + env_i,
                        resample_length_nfp=resample_length_nfp,
                    )
                    for env_i in range(self.batch_size)
                ],
                copy=False,
                autoreset_mode=AutoresetMode.DISABLED,
            )

            best = self.inference(
                envs,
                num_episodes,
                instance_index=instance_index,
                placement_base=placement_base,
                family_size=family_size,
            )

            self._dump_best_panel_json(instance_index, best)
            self._plot_panels(best, instance_index)
            self._plot_best_occ_progression(best.get("best_occ_values") or [], instance_index)
            self._plot_best_occ_rollout_init_markers(best, instance_index)
            self._plot_best_occ_vs_step(
                best.get("best_occ_values") or [],
                best.get("best_occ_steps") or [],
                instance_index,
            )
            self._plot_best_occ_vs_log_likelihood(best, instance_index)
            return float(best.get("best_occ", float("nan")))
        finally:
            if envs is not None:
                envs.close()
            shared_bench_data.close()

    def inference(
        self,
        envs,
        num_episodes,
        deterministic=False,
        instance_index=None,
        placement_base=0,
        family_size: int | None = None,
    ):
        agent = self.agent.to(self.device)

        if self.init_strategy not in ("random", "exploit"):
            raise ValueError(
                f"settings.type must be 'random' or 'exploit', got {self.init_strategy!r}"
            )

        total_final_occ: list[float] = []
        total_final_occ_step: list[int] = []
        total_peak_log_likelihood: list[float | None] = []
        started_from_best_per_episode: list[bool] = []
        exploit_elite_seed_per_episode: list[bool] = []
        best_occ = -float("inf")
        best_panel_coords = None
        max_stage2_seq_observed = 0
        max_vertices_cfg = int(self.data_config.get("max_vertices", 0))
        target_total_episodes = int(num_episodes)
        num_envs = int(self.batch_size)
        processed_episodes = 0
        inst_idx = 0 if instance_index is None else int(instance_index)
        root_inst = int(placement_base) + inst_idx * 1_000_003
        completed: list[tuple[int, float]] = []
        next_seed = int(root_inst)
        rng = np.random.default_rng(inst_idx + 7919)
        self._eval_family_size = int(family_size) if family_size is not None else 0
        start_time = time.perf_counter()
        progress_desc = (
            f"Instance {instance_index + 1}/{self.instance_size}"
            if instance_index is not None
            else "Inference"
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bf16
            else nullcontext()
        )

        use_tqdm_bar = _use_tqdm_progress_bar()
        heartbeat_s = (
            0.0 if use_tqdm_bar else float(os.environ.get("NESTER_PROGRESS_HEARTBEAT_S", "60"))
        )
        last_heartbeat = time.monotonic()

        with torch.no_grad():
            with autocast_context:
                with tqdm(
                    total=target_total_episodes,
                    desc=progress_desc,
                    unit="episode",
                    disable=not use_tqdm_bar,
                    file=sys.stderr,
                ) as pbar:
                    if not use_tqdm_bar and heartbeat_s > 0:
                        print(
                            f"[progress] {progress_desc}: starting {target_total_episodes} episodes "
                            f"(heartbeat every {heartbeat_s:g}s)",
                            flush=True,
                        )
                    while processed_episodes < target_total_episodes:
                        remaining = target_total_episodes - processed_episodes
                        active_envs = min(num_envs, remaining)
                        if active_envs <= 0:
                            break
                        batch_start = processed_episodes
                        placement_seeds, next_seed, elite_seed_flags = self._placement_seeds_vector(
                            batch_start=batch_start,
                            batch_size=num_envs,
                            completed=completed,
                            next_seed=next_seed,
                            rng=rng,
                            root_inst=root_inst,
                        )
                        pending_placed_polygons, warm_start_flags = (
                            self._pending_placed_polygons_for_rollout(
                                batch_size=num_envs,
                                active_envs=active_envs,
                                best_panel_coords=best_panel_coords,
                                rng=rng,
                            )
                        )

                        info, rollout_max_stage2_seq, v_start_batch = self._run_rollout(
                            envs,
                            agent,
                            self.episode_length,
                            deterministic=deterministic,
                            placement_seeds=placement_seeds,
                            pending_placed_polygons=pending_placed_polygons,
                        )
                        max_stage2_seq_observed = max(
                            max_stage2_seq_observed, int(rollout_max_stage2_seq)
                        )

                        info["final_info"]["best_occupancy_ratio"] = info["final_info"][
                            "best_occupancy_ratio"
                        ][:active_envs]
                        info["final_info"]["best_panel"] = {
                            key: value[:active_envs]
                            for key, value in info["final_info"]["best_panel"].items()
                        }
                        info["final_info"]["best_occ_step"] = info["final_info"]["best_occ_step"][
                            :active_envs
                        ]
                        peak_ll_batch = info["final_info"].get("peak_log_likelihood")
                        if peak_ll_batch is not None:
                            peak_ll_batch = peak_ll_batch[:active_envs]
                        best_occ, best_panel_coords = self._structure_info(
                            info, best_occ, best_panel_coords
                        )
                        occ_batch = np.asarray(
                            info["final_info"]["best_occupancy_ratio"], dtype=np.float64
                        )
                        occ_step_batch = np.asarray(
                            info["final_info"]["best_occ_step"], dtype=np.int64
                        )
                        total_final_occ.extend(occ_batch.tolist())
                        total_final_occ_step.extend(int(s) for s in occ_step_batch.tolist())
                        if peak_ll_batch is not None:
                            total_peak_log_likelihood.extend(peak_ll_batch)
                        started_from_best_per_episode.extend(warm_start_flags[:active_envs])
                        exploit_elite_seed_per_episode.extend(
                            [bool(elite_seed_flags[i]) for i in range(active_envs)]
                        )
                        if self.init_strategy == "exploit":
                            for i in range(active_envs):
                                if self.use_critic_elite_ranking:
                                    vs = (
                                        float(v_start_batch[i])
                                        if v_start_batch is not None
                                        else 0.0
                                    )
                                    completed.append(
                                        (int(placement_seeds[i]), float(occ_batch[i]), vs)
                                    )
                                else:
                                    completed.append(
                                        (int(placement_seeds[i]), float(occ_batch[i]))
                                    )
                        processed_episodes += active_envs
                        pbar.update(active_envs)
                        if not use_tqdm_bar and heartbeat_s > 0:
                            now = time.monotonic()
                            if now - last_heartbeat >= heartbeat_s:
                                print(
                                    f"[progress] {progress_desc}: episodes "
                                    f"{processed_episodes}/{target_total_episodes}",
                                    flush=True,
                                )
                                last_heartbeat = now

        elapsed_s = time.perf_counter() - start_time
        total_steps = processed_episodes * self.episode_length
        sps = total_steps / max(elapsed_s, 1e-12)
        print(
            f"Inference done: episodes={processed_episodes}, "
            f"mean_best_occupancy={np.mean(total_final_occ):.4f}, "
            f"max_best_occupancy={np.max(total_final_occ):.4f}",
            flush=True,
        )
        print(
            f"NFP summary: max_stage2_seq={max_stage2_seq_observed}, max_vertices={max_vertices_cfg}",
            flush=True,
        )
        print(f"Timing: elapsed={elapsed_s:.2f}s, steps={total_steps}, sps={sps:.2f}", flush=True)

        if instance_index is not None:
            self._dump_best_occ_values(
                instance_index,
                total_final_occ,
                best_occ,
                started_from_best_per_episode,
                exploit_elite_seed_per_episode,
                total_final_occ_step,
                total_peak_log_likelihood,
            )
        return {
            "best_occ": best_occ,
            "best_occ_values": total_final_occ,
            "best_occ_steps": total_final_occ_step,
            "peak_log_likelihood": list(total_peak_log_likelihood),
            "best_panel_coords": best_panel_coords,
            "max_stage2_seq_observed": max_stage2_seq_observed,
            "max_vertices_cfg": max_vertices_cfg,
            "started_from_best_per_episode": list(started_from_best_per_episode),
            "exploit_elite_seed_per_episode": list(exploit_elite_seed_per_episode),
        }

    def run_eval(self):
        self.load_data()
        recap_rows: list[dict[str, float | int]] = []
        if not self.instance_indices:
            print(
                f"[shard] rank {self._shard_rank}: no instances assigned; skipping eval and model load.",
                flush=True,
            )
            os.makedirs(self._bench_log_dir(), exist_ok=True)
            self._dump_eval_best_occ_recap(recap_rows)
            return self._bench_log_dir()
        self.load_agent()
        for k, global_i in enumerate(self.instance_indices):
            print(
                f"\n=== Instance {global_i + 1}/{self.global_num_instances} (global) — "
                f"shard {self._shard_rank + 1}/{self._shard_world_size}, "
                f"local {k + 1}/{len(self.instance_indices)} ===",
                flush=True,
            )
            best_occ = self.run_eval_instance(global_i, self.num_episodes)
            recap_rows.append({"instance_index": int(global_i), "best_occ": float(best_occ)})
        self._dump_eval_best_occ_recap(recap_rows)
        return self._bench_log_dir()

    def _run_rollout(
        self,
        envs,
        agent,
        episode_length,
        deterministic,
        placement_seeds,
        pending_placed_polygons: list | None = None,
    ):
        n = int(len(placement_seeds))
        if pending_placed_polygons is None:
            pending_placed_polygons = [None] * n
        if len(pending_placed_polygons) != n:
            raise ValueError(
                f"pending_placed_polygons length {len(pending_placed_polygons)} != placement_seeds {n}"
            )
        envs.set_attr("_pending_seed_initial_placement", placement_seeds.tolist())
        envs.set_attr("_pending_initial_placed_polygons", pending_placed_polygons)
        obs, _ = envs.reset(seed=None)
        rollout_max_stage2_seq = 0
        v_start_batch: np.ndarray | None = None
        logprobs_per_env: list[list[float]] = [[] for _ in range(n)]

        for step_i in range(episode_length):
            stage1 = agent.get_action_and_value(obs, deterministic=deterministic)
            if step_i == 0 and self.use_critic_elite_ranking:
                v_start_batch = stage1["value"].detach().float().cpu().numpy().reshape(-1)
            stage1_action = stage1["stage1"]["action"]

            nfp_list = envs.call(
                "compute_nfp_for_action",
                stage1_action["polygon_id"].cpu().numpy(),
                stage1_action["theta_abs"].cpu().numpy(),
            )
            rollout_max_stage2_seq = max(
                rollout_max_stage2_seq, max(len(entry["nfp_patches"]) for entry in nfp_list)
            )
            nfp_pad, nfp_mask, nfp_vertex_mask, nfp_poly_ids = self._pad_nfp_batch_torch(nfp_list)

            rollout_out = agent.get_action_and_value_from_encoded(
                stage1["context"],
                action={"stage1": stage1_action},
                deterministic=deterministic,
                nfp_patches=nfp_pad,
                nfp_mask=nfp_mask,
                nfp_vertex_mask=nfp_vertex_mask,
                nfp_poly_ids=nfp_poly_ids,
            )
            actions = rollout_out["action"]
            # Match AsyncVectorEnv batched Dict spaces: polygon_id (B,), theta Box (B,1), translation (B,2)
            polygon_np = (
                actions["polygon_id"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            )
            theta_np = (
                actions["theta"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1, 1)
            )
            trans_np = (
                actions["translation"]
                .float()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1, 2)
            )
            if not (polygon_np.shape[0] == theta_np.shape[0] == trans_np.shape[0]):
                raise ValueError(
                    f"action batch mismatch: polygon_id {polygon_np.shape}, theta {theta_np.shape}, translation {trans_np.shape}"
                )
            env_action = {
                "polygon_id": polygon_np,
                "translation": trans_np,
                "theta": theta_np,
            }
            joint_logprob = (
                rollout_out["joint_logprob"].detach().float().cpu().numpy().reshape(-1)
            )
            obs, _, _, _, _info = envs.step(env_action)
            for i in range(n):
                logprobs_per_env[i].append(float(joint_logprob[i]))

        exports_list = envs.call("get_bench_rollout_exports")
        active = int(len(placement_seeds))
        max_vertices = int(exports_list[0]["max_vertices"]) if exports_list else 0
        occ_batch = np.array(
            [float(e["best_occupancy_ratio"]) for e in exports_list[:active]],
            dtype=np.float64,
        )
        occ_step_batch = np.array(
            [int(e.get("best_occ_step", 0)) for e in exports_list[:active]],
            dtype=np.int64,
        )
        poly_keys = sorted({int(k) for e in exports_list[:active] for k in e["best_panel"]})
        best_panel: dict = {}
        for pk in poly_keys:
            rows = []
            for e in exports_list[:active]:
                arr = e["best_panel"].get(pk)
                if arr is None:
                    pad = np.full((max_vertices, 2), np.nan, dtype=np.float32)
                else:
                    a = np.asarray(arr, dtype=np.float32)
                    pad = np.full((max_vertices, 2), np.nan, dtype=np.float32)
                    n = min(a.shape[0], max_vertices)
                    pad[:n] = a[:n]
                rows.append(pad)
            best_panel[pk] = np.stack(rows, axis=0)

        peak_log_likelihood: list[float | None] = []
        for i in range(active):
            peak_step = int(occ_step_batch[i])
            ll = gamma_discounted_logprob_before_peak(logprobs_per_env[i], peak_step)
            peak_log_likelihood.append(None if ll is None else float(ll))

        info = {
            "final_info": {
                "best_occupancy_ratio": occ_batch,
                "best_panel": best_panel,
                "best_occ_step": occ_step_batch,
                "peak_log_likelihood": peak_log_likelihood,
            }
        }
        return info, rollout_max_stage2_seq, v_start_batch

    def _pad_nfp_batch_torch(self, nfp_list):
        batch_size = len(nfp_list)
        max_t_needed = max(int(np.asarray(entry["nfp_patches"]).shape[0]) for entry in nfp_list)
        patch_size = int(np.asarray(nfp_list[0]["nfp_patches"]).shape[1])

        if (
            self._nfp_padded_np is None
            or self._nfp_pad_buf_max_t < max_t_needed
            or self._nfp_pad_buf_patch_size != patch_size
            or self._nfp_pad_buf_batch_size != batch_size
        ):
            self._nfp_pad_buf_max_t = int(max_t_needed)
            self._nfp_pad_buf_patch_size = int(patch_size)
            self._nfp_pad_buf_batch_size = int(batch_size)

            self._nfp_padded_np = np.zeros(
                (batch_size, self._nfp_pad_buf_max_t, patch_size, 2), dtype=np.float32
            )
            self._nfp_padded_mask_np = np.zeros(
                (batch_size, self._nfp_pad_buf_max_t), dtype=np.bool_
            )
            self._nfp_vertex_mask_np = np.zeros(
                (batch_size, self._nfp_pad_buf_max_t, patch_size), dtype=np.bool_
            )
            self._nfp_ids_np = np.zeros((batch_size, self._nfp_pad_buf_max_t), dtype=np.int64)

            self._nfp_padded_cpu_t = torch.from_numpy(self._nfp_padded_np)
            self._nfp_padded_mask_cpu_t = torch.from_numpy(self._nfp_padded_mask_np)
            self._nfp_vertex_mask_cpu_t = torch.from_numpy(self._nfp_vertex_mask_np)
            self._nfp_ids_cpu_t = torch.from_numpy(self._nfp_ids_np)

            self._nfp_padded_gpu_t = torch.empty(
                (batch_size, self._nfp_pad_buf_max_t, patch_size, 2),
                dtype=torch.float32,
                device=self.device,
            )
            self._nfp_padded_mask_gpu_t = torch.empty(
                (batch_size, self._nfp_pad_buf_max_t), dtype=torch.bool, device=self.device
            )
            self._nfp_vertex_mask_gpu_t = torch.empty(
                (batch_size, self._nfp_pad_buf_max_t, patch_size),
                dtype=torch.bool,
                device=self.device,
            )
            self._nfp_ids_gpu_t = torch.empty(
                (batch_size, self._nfp_pad_buf_max_t), dtype=torch.long, device=self.device
            )

        self._nfp_padded_np.fill(0)
        self._nfp_padded_mask_np.fill(False)
        self._nfp_vertex_mask_np.fill(False)
        self._nfp_ids_np.fill(0)

        for i, entry in enumerate(nfp_list):
            s = np.asarray(entry["nfp_patches"], dtype=np.float32)
            L = int(s.shape[0])
            self._nfp_padded_np[i, :L] = s
            self._nfp_padded_mask_np[i, :L] = True
            self._nfp_vertex_mask_np[i, :L] = np.asarray(
                entry.get("nfp_vertex_mask", np.ones(s.shape[:2], dtype=np.bool_)),
                dtype=np.bool_,
            )
            self._nfp_ids_np[i, :L] = np.asarray(entry["nfp_poly_ids"], dtype=np.int64)

        self._nfp_padded_gpu_t.copy_(self._nfp_padded_cpu_t, non_blocking=True)
        self._nfp_padded_mask_gpu_t.copy_(self._nfp_padded_mask_cpu_t, non_blocking=True)
        self._nfp_vertex_mask_gpu_t.copy_(self._nfp_vertex_mask_cpu_t, non_blocking=True)
        self._nfp_ids_gpu_t.copy_(self._nfp_ids_cpu_t, non_blocking=True)

        return (
            self._nfp_padded_gpu_t,
            self._nfp_padded_mask_gpu_t,
            self._nfp_vertex_mask_gpu_t,
            self._nfp_ids_gpu_t,
        )

    def _bench_log_dir(self) -> str:
        """``data/bench/{model_folder}/eval/ckpt-{stepslug}/{n-i-t-strat}/[{eval_name}/]``.

        With multi-GPU ``torchrun`` (``WORLD_SIZE`` > 1), appends
        ``rank_{RANK}_of_{WORLD_SIZE}/`` so each rank writes disjoint artifacts.
        """
        model_folder = str(self.model_config.get("model_folder", "default_model")).strip("/")
        run_tag = self._bench_budget_strat_tag
        ckpt_dir = bench_eval_ckpt_subdir_name(self.model_config.get("checkpoint_file"))
        parts = [model_folder, "eval", ckpt_dir, run_tag]
        name_seg = bench_eval_name_segment(self.settings_config.get("eval_name"))
        if name_seg:
            parts.append(name_seg)
        if self._shard_world_size > 1:
            parts.append(f"rank_{self._shard_rank:02d}_of_{self._shard_world_size:02d}")
        rel = "/".join(p for p in parts if p)
        return os.path.join(PATH_TO_PROJECT, "data/bench", rel)

    def _structure_info(self, info, previous_occupancy, previous_panel_coords):
        best_occupancy = info["final_info"]["best_occupancy_ratio"]
        index_max = int(np.argmax(best_occupancy))
        max_best_occupancy = float(best_occupancy[index_max])
        if max_best_occupancy > previous_occupancy:
            best_panel_coords = {
                key: info["final_info"]["best_panel"][key][index_max]
                for key in info["final_info"]["best_panel"]
            }
            return max_best_occupancy, best_panel_coords
        return previous_occupancy, previous_panel_coords

    @staticmethod
    def _axis_aligned_bbox_from_panel_coords(best_panel_coords: dict | None) -> dict | None:
        if not best_panel_coords:
            return None
        panel_coords = {
            key: value
            for key, value in best_panel_coords.items()
            if not (isinstance(key, str) and "_" in str(key))
        }
        all_pts: list[np.ndarray] = []
        for _, coords in panel_coords.items():
            arr = np.asarray(coords, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.shape[0] < 3:
                continue
            all_pts.append(arr)
        if not all_pts:
            return None
        stacked = np.concatenate(all_pts, axis=0)
        min_xy = np.nanmin(stacked, axis=0)
        max_xy = np.nanmax(stacked, axis=0)
        min_xy = np.minimum(min_xy, np.array([0.0, 0.0]))
        max_xy = np.maximum(max_xy, np.array([0.0, 0.0]))
        width = float(max_xy[0] - min_xy[0])
        height = float(max_xy[1] - min_xy[1])
        return {
            "min_xy": [float(min_xy[0]), float(min_xy[1])],
            "max_xy": [float(max_xy[0]), float(max_xy[1])],
            "width": width,
            "height": height,
            "bounding_area": float(width * height),
        }

    def _dump_best_panel_json(self, instance_index: int, best: dict) -> None:
        best_occ = best.get("best_occ")
        best_panel_coords = best.get("best_panel_coords")
        bbox = self._axis_aligned_bbox_from_panel_coords(best_panel_coords)
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_panel.json")
        payload = {
            "instance_index": int(instance_index),
            "best_occ": None if best_occ is None else float(best_occ),
            "axis_aligned_bounding_box": bbox,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved best panel meta: {out_path}", flush=True)

    def _plot_panels(self, best, instance_index):
        best_occ = best.get("best_occ", None)
        best_panel_coords = best.get("best_panel_coords", None)
        if not best_panel_coords:
            return
        log_dir = self._bench_log_dir()
        print(f"Saving best panel plot to: {log_dir}", flush=True)
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_panel.png")

        sns.set_theme(style="ticks", context="notebook", font_scale=0.9)
        fig, ax = plt.subplots(figsize=(10, 8))
        panel_coords = {
            key: value
            for key, value in best_panel_coords.items()
            if not (isinstance(key, str) and "_" in str(key))
        }
        npoly = max(len(panel_coords), 1)
        fill = sns.color_palette("pastel", npoly)
        line = sns.color_palette("dark", npoly)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f8f9fa")
        all_pts = []
        for poly_idx, coords in panel_coords.items():
            arr = np.asarray(coords, dtype=np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                continue
            all_pts.append(arr)
            j = int(poly_idx) % npoly
            ax.fill(
                arr[:, 0], arr[:, 1], color=fill[j], edgecolor=line[j], linewidth=0.9, alpha=0.92
            )
            c = arr.mean(axis=0)
            ax.text(
                c[0],
                c[1],
                str(poly_idx),
                size=7,
                ha="center",
                va="center",
                color="0.15",
                fontweight="medium",
            )
        if all_pts:
            stacked = np.concatenate(all_pts, axis=0)
            min_xy = stacked.min(axis=0)
            max_xy = stacked.max(axis=0)
            center_xy = (min_xy + max_xy) / 2.0
            width = max(float(max_xy[0] - min_xy[0]), 1e-6)
            height = max(float(max_xy[1] - min_xy[1]), 1e-6)
            exterior_lim = 1.2
            ax.set_xlim(
                center_xy[0] - width / 2.0 * exterior_lim, center_xy[0] + width / 2.0 * exterior_lim
            )
            ax.set_ylim(
                center_xy[1] - height / 2.0 * exterior_lim,
                center_xy[1] + height / 2.0 * exterior_lim,
            )
            border = plt.Rectangle(
                center_xy - np.array([width / 2.0, height / 2.0]),
                width,
                height,
                linewidth=1.1,
                edgecolor=sns.color_palette("flare")[2],
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(border)
            ax.set_title(f"Instance {instance_index} best panel | occupancy={float(best_occ):.6f}")
            ax.tick_params(labelsize=7, colors="0.45")
            sns.despine(ax=ax, offset=1, trim=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved best panel plot: {out_path}", flush=True)

    def _plot_best_occ_progression(self, best_occ_values, instance_index):
        vals = np.asarray(best_occ_values, dtype=np.float64).reshape(-1)
        if vals.size == 0:
            return
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_occ_progression.png")
        episodes = np.arange(1, vals.size + 1, dtype=np.int32)
        running = np.maximum.accumulate(vals)
        sns.set_theme(style="ticks", context="notebook", font_scale=0.9)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=episodes, y=vals, ax=ax, label="Per-rollout best occupancy", alpha=0.65)
        sns.lineplot(x=episodes, y=running, ax=ax, label="Running max", linewidth=2)
        ax.set_xlabel("Rollout index")
        ax.set_ylabel("Best occupancy ratio")
        ax.set_title(f"Instance {instance_index} — occupancy progression ({vals.size} rollouts)")
        ax.legend(loc="lower right")
        sns.despine(ax=ax, offset=4, trim=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved best occupancy progression plot: {out_path}", flush=True)

    def _plot_best_occ_vs_step(
        self,
        best_occ_values,
        best_occ_steps,
        instance_index: int,
    ) -> None:
        vals = np.asarray(best_occ_values, dtype=np.float64).reshape(-1)
        steps = np.asarray(best_occ_steps, dtype=np.int64).reshape(-1)
        if vals.size == 0 or steps.size != vals.size:
            return
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_occ_vs_step.png")
        ep_len = int(self.episode_length)

        max_step = int(max(int(steps.max()), ep_len))
        centers = np.arange(0, max_step + 1, dtype=np.int64)
        counts = np.bincount(steps.clip(min=0), minlength=max_step + 1)[: max_step + 1]
        total = int(counts.sum())
        freqs = counts / max(total, 1)

        sns.set_theme(style="ticks", context="notebook", font_scale=0.9)
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_color = sns.color_palette("flare")[2]
        ax.bar(centers, counts, width=0.85, color=bar_color, edgecolor="0.2", linewidth=0.5)
        for c, n, f in zip(centers, counts, freqs):
            if n > 0:
                ax.text(
                    c,
                    n,
                    f"{f * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="0.25",
                )

        ax.set_xlabel("Step index in episode where best occ was found")
        ax.set_ylabel("Number of rollouts")
        ax.set_title(
            f"Instance {instance_index} — best_occ step-index histogram "
            f"(n={total} rollouts, episode_length={ep_len})"
        )
        ax.set_xticks(centers)
        ax.set_xlim(-0.5, max_step + 0.5)
        ax.margins(y=0.12)
        ax.grid(True, axis="y", alpha=0.3)
        sns.despine(ax=ax, offset=4, trim=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved best occ vs step plot: {out_path}", flush=True)

    def _plot_best_occ_rollout_init_markers(self, best: dict, instance_index: int) -> None:
        vals = best.get("best_occ_values") or []
        warm = best.get("started_from_best_per_episode")
        if not vals or warm is None or len(warm) != len(vals):
            return
        elite = best.get("exploit_elite_seed_per_episode")
        if elite is not None and len(elite) == len(vals):
            plot_flags = [bool(w) or bool(e) for w, e in zip(warm, elite)]
        else:
            plot_flags = list(warm)
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_occ_rollout_init.png")
        top_frac = self.plot_rollout_top_frac
        pct = top_frac * 100.0
        save_best_occ_rollout_init_trace_png(
            out_path,
            vals,
            plot_flags,
            title=(
                f"Instance {instance_index} — top {pct:g}% rollouts by best_occ "
                f"(× = warm-start from best panel or elite placement replay, ○ = otherwise)"
            ),
            top_frac=top_frac,
        )
        print(f"Saved rollout-init marker plot: {out_path}", flush=True)

    def _plot_best_occ_vs_log_likelihood(self, best: dict, instance_index: int) -> None:
        peak_ll = best.get("peak_log_likelihood")
        legacy_ll = best.get("peak_reward_log_likelihoods")
        if not peak_ll and not legacy_ll:
            print(
                "[plot] skip best_occ vs log-likelihood: no peak_log_likelihood "
                "(re-run eval with current code)",
                flush=True,
            )
            return
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(
            log_dir, f"instance_{instance_index:04d}_best_occ_vs_log_likelihood.png"
        )
        ok = save_best_occ_vs_log_likelihood_png(
            out_path,
            best.get("best_occ_values") or [],
            peak_reward_log_likelihoods=legacy_ll,
            peak_log_likelihood=peak_ll,
            instance_index=int(instance_index),
            top_frac=float(self.plot_occ_likelihood_top_frac),
            max_plot_points=int(self.max_occ_likelihood_plot_points),
        )
        if ok:
            print(f"Saved best occ vs log-likelihood plot: {out_path}", flush=True)
        else:
            print(
                f"[plot] skip best_occ vs log-likelihood: no finite LL in top "
                f"{self.plot_occ_likelihood_top_frac:g} fraction",
                flush=True,
            )

    def _dump_eval_best_occ_recap(self, instances: list[dict[str, float | int]]) -> None:
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "eval_best_occ_recap.json")
        scores = [float(row["best_occ"]) for row in instances]
        finite = [s for s in scores if np.isfinite(s)]
        mean_best = float(np.mean(finite)) if finite else None
        payload = {
            "num_instances": int(len(instances)),
            "mean_best_occ": mean_best,
            "instances": instances,
            "bench_budget_strat_tag": self._bench_budget_strat_tag,
            "model_folder": str(self.model_config.get("model_folder", "")),
            "checkpoint_file": str(self.model_config.get("checkpoint_file", "")),
            "global_num_instances": int(self.global_num_instances),
            "shard_rank": int(self._shard_rank),
            "shard_world_size": int(self._shard_world_size),
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved eval best-occupancy recap: {out_path}", flush=True)

    def _dump_best_occ_values(
        self,
        instance_index,
        best_occ_values,
        best_occ,
        started_from_best_per_episode: list[bool],
        exploit_elite_seed_per_episode: list[bool],
        best_occ_steps: list[int],
        peak_log_likelihood: list[float | None] | None = None,
    ):
        log_dir = self._bench_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"instance_{instance_index:04d}_best_occ_values.json")
        n = len(best_occ_values)
        flags = list(started_from_best_per_episode)
        elite_flags = list(exploit_elite_seed_per_episode)
        steps = list(best_occ_steps)
        if len(flags) != n:
            raise ValueError(
                f"started_from_best_per_episode length {len(flags)} != num episodes {n}"
            )
        if len(elite_flags) != n:
            raise ValueError(
                f"exploit_elite_seed_per_episode length {len(elite_flags)} != num episodes {n}"
            )
        if len(steps) != n:
            raise ValueError(f"best_occ_steps length {len(steps)} != num episodes {n}")
        peak_ll = list(peak_log_likelihood or [])
        if peak_ll and len(peak_ll) != n:
            raise ValueError(f"peak_log_likelihood length {len(peak_ll)} != num episodes {n}")
        warm_eff = float(exploit_p_normalized(self.warm_start_p))
        payload = {
            "instance_index": int(instance_index),
            "num_episodes": int(n),
            "episode_length": int(self.episode_length),
            "best_occ": float(best_occ),
            "best_occ_values": [float(v) for v in best_occ_values],
            "best_occ_steps": [int(s) for s in steps],
            "peak_log_likelihood_gamma": _BEST_OCC_LOGPROB_GAMMA,
            "peak_log_likelihood_lookback": _BEST_OCC_LOGPROB_LOOKBACK,
            "peak_log_likelihood": peak_ll,
            "init_strategy": str(self.init_strategy),
            "exploit_p": float(self.exploit_p),
            "elite_frac": float(self.elite_frac),
            "warm_start_p": float(self.warm_start_p),
            "warm_start_p_effective_0_1": warm_eff,
            "started_from_best_per_episode": flags,
            "exploit_elite_seed_per_episode": elite_flags,
        }
        json_indent = 2 if n <= _JSON_PRETTY_MAX_EPISODES else None
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=json_indent)
        print(f"Saved best occupancy values: {out_path}", flush=True)

    def _placement_seeds_vector(
        self, batch_start, batch_size, completed, next_seed, rng, root_inst
    ):
        placement_seeds = np.zeros(batch_size, dtype=np.int64)
        elite_flags = np.zeros(batch_size, dtype=np.bool_)

        if self.init_strategy == "exploit":
            ep = exploit_p_normalized(self.exploit_p)
            elite = elite_seeds_from_completed(
                completed,
                self.elite_frac,
                critic_elite_coef=(
                    self.critic_elite_coef if self.use_critic_elite_ranking else 0.0
                ),
            )
            ns = int(next_seed)
            for i in range(batch_size):
                if elite and rng.random() < ep:
                    placement_seeds[i] = int(rng.choice(elite))
                    elite_flags[i] = True
                else:
                    placement_seeds[i] = ns
                    ns += 1
            return placement_seeds, ns, elite_flags

        for i in range(batch_size):
            placement_seeds[i] = int(root_inst) + int(batch_start) + i
        return placement_seeds, int(next_seed), elite_flags

    def _pending_placed_polygons_for_rollout(self, batch_size, active_envs, best_panel_coords, rng):
        warm = exploit_p_normalized(self.warm_start_p)
        pending: list = []
        warm_flags: list[bool] = []
        for i in range(batch_size):
            if i >= active_envs:
                pending.append(None)
                warm_flags.append(False)
                continue
            if best_panel_coords is None or warm <= 0 or rng.random() >= warm:
                pending.append(None)
                warm_flags.append(False)
                continue
            pending.append(_panel_dict_to_placed_list(best_panel_coords))
            warm_flags.append(True)
        return pending, warm_flags


InferenceBench = EvalPipeline
