"""Bench path naming, sharding, and rollout sampling helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

_CKPT_RE = re.compile(r"checkpoint_(\d+)", re.IGNORECASE)


def sampling_flags(mode: str) -> tuple[bool, bool]:
    m = str(mode).lower().replace("-", "_")
    if m == "both":
        return True, True
    if m in ("policy_stochastic_only", "stoch_policy", "seed_variance"):
        return True, False
    if m in ("init_random_only", "scenario_variance", "init_random"):
        return False, True
    raise ValueError(
        f"unknown sampling_mode={mode!r}; use 'both', 'policy_stochastic_only', "
        "or 'init_random_only' (aliases: stoch_policy / seed_variance / scenario_variance)"
    )


def exploit_p_normalized(p: float) -> float:
    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return p


def elite_seeds_from_completed(
    completed: list[tuple[int, float] | tuple[int, float, float]],
    elite_frac: float = 0.01,
    *,
    critic_elite_coef: float = 0.0,
) -> list[int]:
    if not completed:
        return []
    coef = float(critic_elite_coef)
    seeds: list[int] = []
    keys: list[float] = []
    for row in completed:
        seed = int(row[0])
        occ = float(row[1])
        v0 = float(row[2]) if len(row) >= 3 else 0.0
        seeds.append(seed)
        keys.append(occ + coef * v0)
    occs = np.asarray(keys, dtype=np.float64)
    k = max(1, int(np.ceil(len(completed) * float(elite_frac))))
    order = np.argsort(-occs)[:k]
    return [int(seeds[j]) for j in order]


def bench_checkpoint_step_count(checkpoint_file: str | None) -> int | None:
    if not checkpoint_file:
        return None
    m = _CKPT_RE.search(Path(str(checkpoint_file)).name)
    if not m:
        return None
    return int(m.group(1))


def bench_eval_ckpt_subdir_name(checkpoint_file: str | None) -> str:
    n = bench_checkpoint_step_count(checkpoint_file)
    if n is None:
        return "ckpt_unknown"
    if n >= 1_000_000_000:
        g = n / 1_000_000_000
        xi = int(g)
        tenth = int(round((g - xi) * 10))
        if tenth >= 10:
            xi += 1
            tenth = 0
        return f"ckpt_{xi}G{tenth}"
    return f"ckpt_{n}"


def bench_eval_ckpt_job_token(checkpoint_file: str | None) -> str:
    return bench_eval_ckpt_subdir_name(checkpoint_file).replace("_", "-").lower()


def bench_checkpoint_steps_slug(checkpoint_file: str | None) -> str:
    sub = bench_eval_ckpt_subdir_name(checkpoint_file)
    if sub == "ckpt_unknown":
        return "unknown"
    if sub.startswith("ckpt_"):
        return sub[len("ckpt_") :].lower()
    return sub.lower()


def _format_episode_budget(n: int) -> str:
    n = int(n)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def bench_eval_budget_strat_tag(
    *,
    num_instances: int,
    num_episodes: int,
    episode_length: int,
    init_strategy: str,
    exploit_p: float,
    warm_start_p: float,
    placement_varied: bool,
) -> str:
    n_ep = _format_episode_budget(num_episodes)
    parts = [
        f"n{n_ep}",
        f"i{int(num_instances)}",
        f"t{int(episode_length)}",
        f"e{int(round(float(exploit_p)))}",
    ]
    tag = "-".join(parts)
    if float(warm_start_p) >= 1e-9:
        wp = float(warm_start_p)
        wp_s = str(int(round(wp))) if wp > 1.0 else str(int(round(wp * 100)))
        tag += f"-ws{wp_s}"
    if not placement_varied:
        tag += "-npv"
    elif str(init_strategy).lower() == "random":
        tag += "-rnd"
    return tag


def bench_eval_name_segment(eval_name: str | None) -> str | None:
    if eval_name is None:
        return None
    raw = str(eval_name).strip()
    if not raw:
        return None
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-_").lower()
    if not s:
        return None
    return s[:32]


def bench_eval_budget_strat_tag_from_config(settings: dict | None, data: dict | None) -> str:
    settings = settings or {}
    data = data or {}
    _policy_stoch, placement_varied = sampling_flags(str(settings.get("sampling_mode", "both")))
    del _policy_stoch
    init_strategy = str(settings.get("type", "random")).lower()
    warm_start_p = float(settings.get("warm_start_p", 0.0))
    if not placement_varied:
        init_strategy = "random"
        warm_start_p = 0.0
    return bench_eval_budget_strat_tag(
        num_instances=int(data.get("num_instances", 1)),
        num_episodes=int(settings.get("num_episodes", 0)),
        episode_length=int(settings.get("episode_length", 1)),
        init_strategy=init_strategy,
        exploit_p=float(settings.get("exploit_p", 30.0)),
        warm_start_p=warm_start_p,
        placement_varied=bool(placement_varied),
    )


def bench_eval_run_path_segments(
    *,
    checkpoint_file: str | None,
    settings: dict | None,
    data: dict | None,
) -> list[str]:
    settings = settings or {}
    data = data or {}
    segments = [
        bench_eval_ckpt_subdir_name(checkpoint_file),
        bench_eval_budget_strat_tag_from_config(settings, data),
    ]
    name_seg = bench_eval_name_segment(settings.get("eval_name"))
    if name_seg:
        segments.append(name_seg)
    return segments


def bench_eval_shard_triplet() -> tuple[int, int, int]:
    """``(RANK, WORLD_SIZE, LOCAL_RANK)`` under ``torchrun``; ``(0, 1, 0)`` if unset."""
    ws_raw = os.environ.get("WORLD_SIZE")
    if ws_raw is None or str(ws_raw).strip() == "":
        return 0, 1, 0
    ws = max(1, int(ws_raw))
    rank = int(os.environ.get("RANK", "0"))
    if ws > 1:
        rank = rank % ws
    lr_raw = os.environ.get("LOCAL_RANK")
    if lr_raw is None or str(lr_raw).strip() == "":
        lr = 0
    else:
        lr = int(lr_raw)
    return rank, ws, lr


def instance_indices_for_shard(num_instances: int, rank: int, world_size: int) -> list[int]:
    """Balanced contiguous slices of ``range(num_instances)`` (same order as single-process)."""
    n = int(num_instances)
    if n <= 0:
        return []
    if world_size <= 1:
        return list(range(n))
    r = rank % world_size
    base = n // world_size
    rem = n % world_size
    start = r * base + min(r, rem)
    end = start + base + (1 if r < rem else 0)
    return list(range(start, end))
