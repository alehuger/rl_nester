import copy
import os
import random
import threading
import time

import gymnasium as gym
import numpy as np
import psutil
import torch

from nester import PATH_TO_PROJECT
from nester.env import InferenceEnv, TrainingEnv
from nester.agent.agent.agent import PotRLAgent


def load_agent_model(checkpoint_path: str, device: str = "cpu"):
    """Load agent, full config, and environment section from a training checkpoint."""
    try:
        full_path = (
            checkpoint_path
            if os.path.isabs(checkpoint_path)
            else os.path.join(PATH_TO_PROJECT, checkpoint_path)
        )
        checkpoint = torch.load(full_path, map_location=device, weights_only=False)
        config = checkpoint.get("config")

        env_cfg = config.get("environment", {})
        model_cfg = config.get("model", {})

        agent_type = model_cfg.get("agent_type", "pot")
        if agent_type != "pot":
            raise ValueError(f"Unsupported agent_type: {agent_type!r}; only 'pot' is supported")
        model_kwargs = model_cfg.get("model_kwargs", {})

        family_size_range = env_cfg.get("family_size_range", [4, 4])
        encoder_kwargs = model_kwargs.get("encoder_config", {})
        encoder_kwargs["max_candidates"] = family_size_range[1]

        agent = PotRLAgent(**model_kwargs, device=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()

        return agent, config, env_cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def create_environment(
    env_cfg,
    seed=None,
    db_path=None,
    episode_length=20,
):
    """Initialize a TrainingEnv for interactive / analysis scripts."""
    if seed is None:
        seed = random.randint(0, 100_000)

    data_path_config = env_cfg.get("data_path", "")
    if db_path is not None:
        data_path = db_path
    elif data_path_config.startswith("/"):
        data_path = data_path_config
    else:
        data_path = PATH_TO_PROJECT + data_path_config

    env = TrainingEnv(
        db_path=data_path,
        family_size_range=env_cfg.get("family_size_range", [4, 4]),
        episode_length=episode_length or env_cfg.get("episode_length", 20),
        shared_memory_dict=None,
        limit_samples=env_cfg.get("limit_samples", 10_000),
        resample_length=env_cfg.get("resample_length", 0.05),
        seed=seed,
        track_best_panel=True,
    )
    obs, _ = env.reset(seed=seed)
    return env, obs, seed


def _pad_nfp_single(nfp_entry: dict):
    """Pad one env NFP dict (T, P, 2) to batch tensors (B=1)."""
    seq = np.asarray(nfp_entry["nfp_patches"], dtype=np.float32)
    ids = np.asarray(nfp_entry["nfp_poly_ids"], dtype=np.int64)
    vertex_mask = np.asarray(
        nfp_entry.get("nfp_vertex_mask", np.ones(seq.shape[:2], dtype=np.bool_)),
        dtype=np.bool_,
    )
    length = int(seq.shape[0])
    if length <= 0:
        raise ValueError("NFP patches must be non-empty")
    p = int(seq.shape[1])
    padded = np.zeros((1, length, p, 2), dtype=np.float32)
    padded_mask = np.zeros((1, length), dtype=np.bool_)
    padded_vertex_mask = np.zeros((1, length, p), dtype=np.bool_)
    padded_ids = np.zeros((1, length), dtype=np.int64)
    padded[0, :length] = seq
    padded_mask[0, :length] = True
    padded_vertex_mask[0, :length] = vertex_mask
    padded_ids[0, :length] = ids
    return (
        torch.from_numpy(padded),
        torch.from_numpy(padded_mask).bool(),
        torch.from_numpy(padded_vertex_mask).bool(),
        torch.from_numpy(padded_ids).long(),
    )


def _batch_obs(obs) -> dict:
    """Wrap a single-env observation dict as a batch keyed by env index."""
    if isinstance(obs, dict) and obs and all(isinstance(k, int) for k in obs):
        return obs
    return {0: obs}


def perform_step(env, agent, obs, deterministic: bool = False):
    """Run one two-stage placement step on a single env."""
    batch_obs = _batch_obs(obs)
    with torch.no_grad():
        stage1 = agent.get_action_and_value(batch_obs, deterministic=deterministic)
        stage1_action = stage1["stage1"]["action"]
        nfp_entry = env.compute_nfp_for_action(
            stage1_action["polygon_id"].cpu().numpy(),
            stage1_action["theta_abs"].cpu().numpy(),
        )
        nfp_pad, nfp_mask, nfp_vertex_mask, nfp_ids = _pad_nfp_single(nfp_entry)
        rollout_out = agent.get_action_and_value_from_encoded(
            stage1["context"],
            action={"stage1": stage1_action},
            deterministic=deterministic,
            nfp_patches=nfp_pad,
            nfp_mask=nfp_mask,
            nfp_vertex_mask=nfp_vertex_mask,
            nfp_poly_ids=nfp_ids,
        )
        actions = rollout_out["action"]
        env_action = {
            "polygon_id": int(actions["polygon_id"].item()),
            "translation": actions["translation"].detach().cpu().numpy().astype(np.float32),
            "theta": actions["theta"].detach().cpu().numpy().astype(np.float32).reshape(1),
        }
        next_obs, reward, terminated, truncated, info = env.step(env_action)
    return next_obs, reward, terminated, truncated, info, rollout_out


class GPUProfiler:
    def __init__(self, device_index=0):
        self.monitoring = False
        self.device_index = device_index
        self.gpu_util_samples = []
        self.memory_used_samples = []
        self.temperature_samples = []
        self.power_samples = []
        self.monitor_thread = None
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            print(f"Failed to initialize GPUProfiler: {e}")
            pass

    def start(self):
        if self.monitoring or not self.handle:
            return
        self.monitoring = True
        self.gpu_util_samples = []
        self.memory_used_samples = []
        self.temperature_samples = []
        self.power_samples = []

        def monitor():
            import pynvml

            while self.monitoring:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.gpu_util_samples.append(util.gpu)
                self.memory_used_samples.append(mem.used / 1024 / 1024)
                self.temperature_samples.append(temp)
                self.power_samples.append(power)
                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        if not self.monitoring:
            return {}
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=0.2)
        if not self.gpu_util_samples:
            return {}
        return {
            "avg_gpu_util": np.mean(self.gpu_util_samples),
            "max_gpu_util": np.max(self.gpu_util_samples),
            "avg_memory_used_mb": np.mean(self.memory_used_samples),
            "max_memory_used_mb": np.max(self.memory_used_samples),
            "avg_temperature": np.mean(self.temperature_samples),
            "max_temperature": np.max(self.temperature_samples),
            "avg_power_w": np.mean(self.power_samples),
            "max_power_w": np.max(self.power_samples),
        }


class CPUProfiler:
    def __init__(self):
        self.monitoring = False
        self.cpu_util_samples = []
        self.memory_used_samples = []
        self.monitor_thread = None

    def start(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.cpu_util_samples = []
        self.memory_used_samples = []

        def monitor():
            while self.monitoring:
                per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                self.cpu_util_samples.append(float(np.mean(per_cpu)) if per_cpu else 0.0)
                self.memory_used_samples.append(psutil.virtual_memory().used / 1024 / 1024)
                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        if not self.monitoring:
            return {}
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=0.2)
        if not self.cpu_util_samples:
            return {}
        return {
            "avg_cpu_util": np.mean(self.cpu_util_samples),
            "max_cpu_util": np.max(self.cpu_util_samples),
            "avg_memory_used_mb": np.mean(self.memory_used_samples),
            "max_memory_used_mb": np.max(self.memory_used_samples),
        }


class TimingContext:
    def __init__(self, profiler, operation_name, sync_cuda):
        self.profiler = profiler
        self.operation_name = operation_name
        self.sync_cuda = sync_cuda
        self.start_time = None

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time

        if self.operation_name in self.profiler.timings:
            self.profiler.timings[self.operation_name].append(elapsed)
        return False


class TimingProfiler:
    def __init__(self, sync_cuda=True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.reset()

    def reset(self):
        self.timings = {
            "env_step": [],
            "rollout_inference": [],
            "forward_pass": [],
            "backward_pass": [],
        }

    def timer(self, operation_name):
        return TimingContext(self, operation_name, self.sync_cuda)

    def get_stats(self, num_steps, num_envs):
        stats = {}
        total_steps = num_steps * num_envs
        for op_name, times in self.timings.items():
            if times:
                total_time = np.sum(times)
                mean_time = np.mean(times)
                stats[op_name] = {
                    "mean": mean_time,
                    "mean_per_step_ms": (total_time / total_steps) * 1000,
                    "total": total_time,
                    "count": len(times),
                }
        return stats


def apply_env_model_overrides(env_cfg: dict, model_kwargs: dict) -> dict:
    model_kwargs = copy.deepcopy(model_kwargs)
    encoder_kwargs = model_kwargs.setdefault("encoder_config", {})
    family_size_range = env_cfg.get("family_size_range", [4, 4])
    encoder_kwargs["max_candidates"] = max(family_size_range)
    encoder_kwargs["data_patch_size"] = model_kwargs.get("data_patch_size", 10)
    mv = env_cfg.get("max_vertices")
    if mv is not None:
        encoder_kwargs.setdefault("transformer", {})["max_vertices"] = int(mv)
    policy_kwargs = model_kwargs.setdefault("policy_config", {})
    if env_cfg.get("max_nfp_components") is not None:
        policy_kwargs["max_nfp_components"] = int(env_cfg["max_nfp_components"])
    return model_kwargs


def make_env(
    seed: int,
    db_path: str,
    family_size_range: tuple[int, int],
    episode_length: int,
    resample_length: float,
    resample_length_nfp: float,
    limit_samples: int,
    shared_memory_dict: dict[str, str],
    env_index: int = 0,
    max_vertices: int = 1000,
    max_nfp_vertices: int | None = None,
    nfp_patch_size: int = 10,
    max_reward: bool = False,
    track_best_panel: bool = False,
    occupancy_metric: str = "aabb",
    max_nfp_components: int = 3,
    step_warmup: int = 0,
):
    def thunk():
        env = TrainingEnv(
            db_path=db_path,
            family_size_range=family_size_range,
            episode_length=episode_length,
            shared_memory_dict=shared_memory_dict,
            limit_samples=limit_samples,
            resample_length=resample_length,
            resample_length_nfp=resample_length_nfp,
            seed=seed,
            env_index=env_index,
            max_vertices=max_vertices,
            max_nfp_vertices=max_nfp_vertices,
            nfp_patch_size=nfp_patch_size,
            max_reward=max_reward,
            track_best_panel=track_best_panel,
            occupancy_metric=occupancy_metric,
            max_nfp_components=max_nfp_components,
            step_warmup=step_warmup,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_inference_env(
    *,
    shared_memory_dict: dict,
    episode_length: int,
    resample_length: float,
    seed: int,
    env_index: int,
    family_size: int,
    max_vertices: int,
    nfp_patch_size: int,
    max_reward: bool,
    placement_seed: int,
    resample_length_nfp: float | None = None,
    use_nfp_cache: bool = False,
    nfp_cache_bins: int = 720,
    nfp_cache_shm_meta: dict | None = None,
    occupancy_metric: str = "aabb",
    max_nfp_components: int = 3,
    step_warmup: int = 0,
):
    """Thunk factory for one AsyncVectorEnv worker on a fixed bench family (shared memory)."""
    _ = family_size  # family size is implied by SHM; caller keeps arg for layout checks

    def thunk():
        return InferenceEnv(
            episode_length=int(episode_length),
            shared_memory_dict=shared_memory_dict,
            resample_length=float(resample_length),
            seed=int(seed),
            env_index=int(env_index),
            max_vertices=int(max_vertices),
            nfp_patch_size=int(nfp_patch_size),
            max_reward=bool(max_reward),
            use_nfp_cache=bool(use_nfp_cache),
            nfp_cache_bins=int(nfp_cache_bins),
            nfp_cache_shm_meta=nfp_cache_shm_meta,
            placement_seed=int(placement_seed) if placement_seed is not None else None,
            resample_length_nfp=float(resample_length_nfp)
            if resample_length_nfp is not None
            else None,
            occupancy_metric=str(occupancy_metric),
            max_nfp_components=int(max_nfp_components),
            step_warmup=int(step_warmup),
        )

    return thunk


def load_agent_state_dict_merged(agent, state_dict, *, skip_prefixes: tuple[str, ...] = ()):
    """Load checkpoint weights; raise if any tensor is missing or shape-mismatched.

    Allowed exceptions: widening ``encoder.mlp_heads.0.weight`` or
    ``value_network.value_network.0.weight`` by one input column (``prev_best_occupancy``).
    ``skip_prefixes`` keeps current init for matching keys (e.g. frozen layers).
    """
    cur = agent.state_dict()
    merged = {}
    skipped: list[str] = []
    for k, mt in cur.items():
        if any(k.startswith(p) for p in skip_prefixes):
            merged[k] = mt.clone()
            continue
        if k not in state_dict:
            merged[k] = mt.clone()
            skipped.append(f"{k} (missing in checkpoint)")
            continue
        ct = state_dict[k].to(device=mt.device, dtype=mt.dtype)
        if mt.shape == ct.shape:
            merged[k] = ct
        elif (
            k == "encoder.mlp_heads.0.weight"
            and ct.dim() == 2
            and ct.shape[0] == mt.shape[0]
            and ct.shape[1] + 1 == mt.shape[1]
        ):
            w = mt.clone()
            w[:, :-1] = ct
            merged[k] = w
        elif (
            k == "value_network.value_network.0.weight"
            and ct.dim() == 2
            and ct.shape[0] == mt.shape[0]
            and ct.shape[1] + 1 == mt.shape[1]
        ):
            w = mt.clone()
            w[:, :-1] = ct
            merged[k] = w
        else:
            merged[k] = mt.clone()
            skipped.append(f"{k} (checkpoint {tuple(ct.shape)} vs model {tuple(mt.shape)})")
    if skipped:
        preview = "\n  ".join(skipped[:20])
        extra = f"\n  ... and {len(skipped) - 20} more" if len(skipped) > 20 else ""
        raise RuntimeError(
            f"Checkpoint load failed: {len(skipped)} parameter(s) missing or shape-mismatched.\n"
            f"  {preview}{extra}"
        )
    agent.load_state_dict(merged)


def _optimizer_state_shapes_ok(optimizer, state_dict):
    if not state_dict or "state" not in state_dict:
        return False
    flat = [p for g in optimizer.param_groups for p in g["params"]]
    st = state_dict["state"]
    if len(st) != len(flat):
        return False
    for i, p in enumerate(flat):
        blk = st.get(i)
        if blk is None:
            return False
        for key in ("exp_avg", "exp_avg_sq"):
            if key in blk and tuple(blk[key].shape) != tuple(p.shape):
                return False
    return True


def try_load_optimizer_state_dict(optimizer, state_dict):
    if not state_dict:
        return
    if _optimizer_state_shapes_ok(optimizer, state_dict):
        optimizer.load_state_dict(state_dict)
    else:
        raise RuntimeError(
            "Optimizer checkpoint does not match current parameter shapes; "
            "refusing silent skip."
        )
