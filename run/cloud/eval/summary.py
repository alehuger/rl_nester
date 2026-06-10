"""Merge per-rank bench eval artifacts into a single cross-shard score summary."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

BENCH_JOINT_SUMMARY_FILENAME = "eval_joint_summary.json"
# Top 0.1% of rollouts (999th per-mille); matches settings.elite_frac default 0.001.
BENCH_JOINT_OCC_PERCENTILE = 99.9

_INSTANCE_VALUES_RE = re.compile(r"^instance_(\d+)_best_occ_values\.json$")
_SHARD_DIR_RE = re.compile(r"^rank_(\d+)_of_(\d+)$")


def bench_run_root_from_log_dir(log_dir: str | Path, shard_rank: int, shard_world_size: int) -> Path:
    """Eval run directory without the per-rank suffix."""
    root = Path(log_dir)
    if shard_world_size <= 1:
        return root
    suffix = f"rank_{shard_rank:02d}_of_{shard_world_size:02d}"
    if root.name == suffix:
        return root.parent
    return root


def bench_shard_barrier(run_root: Path, shard_rank: int, shard_world_size: int, *, timeout_s: float) -> None:
    """File-based barrier so rank 0 can merge after all ranks finish writing."""
    if shard_world_size <= 1:
        return
    barrier_dir = run_root / ".bench_shard_barrier"
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"rank_{shard_rank:02d}.done").touch()
    if shard_rank != 0:
        return
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if all((barrier_dir / f"rank_{r:02d}.done").is_file() for r in range(shard_world_size)):
            return
        time.sleep(2.0)
    missing = [
        r
        for r in range(shard_world_size)
        if not (barrier_dir / f"rank_{r:02d}.done").is_file()
    ]
    raise TimeoutError(
        f"Bench shard barrier timed out after {timeout_s}s; missing ranks: {missing}"
    )


def occ_at_joint_percentile(values: list[float] | np.ndarray) -> float | None:
    """Occupancy at the 99.9th percentile (top 0.1% / 999th per-mille)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, BENCH_JOINT_OCC_PERCENTILE))


def _shard_dirs(run_root: Path, shard_world_size: int) -> list[Path]:
    if shard_world_size <= 1:
        return [run_root]
    dirs: list[Path] = []
    for rank in range(shard_world_size):
        d = run_root / f"rank_{rank:02d}_of_{shard_world_size:02d}"
        if d.is_dir():
            dirs.append(d)
    if not dirs and run_root.is_dir():
        for child in sorted(run_root.iterdir()):
            if child.is_dir() and _SHARD_DIR_RE.match(child.name):
                dirs.append(child)
    return dirs


def _load_recap_metadata(shard_dirs: list[Path]) -> dict:
    for shard_dir in shard_dirs:
        recap_path = shard_dir / "eval_best_occ_recap.json"
        if not recap_path.is_file():
            continue
        with open(recap_path, encoding="utf-8") as handle:
            recap = json.load(handle)
        return {
            k: recap[k]
            for k in (
                "bench_budget_strat_tag",
                "model_folder",
                "checkpoint_file",
                "global_num_instances",
            )
            if k in recap
        }
    return {}


def build_bench_joint_summary(run_root: Path, shard_world_size: int) -> dict:
    """Per-instance best_occ and 99.9th-percentile occupancy across all shards."""
    shard_dirs = _shard_dirs(run_root, shard_world_size)
    by_instance: dict[int, dict] = {}

    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            continue
        for path in sorted(shard_dir.glob("instance_*_best_occ_values.json")):
            m = _INSTANCE_VALUES_RE.match(path.name)
            if not m:
                continue
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            idx = int(payload.get("instance_index", m.group(1)))
            values = payload.get("best_occ_values") or []
            best = float(payload.get("best_occ", np.nanmax(values) if len(values) else float("nan")))
            p999 = occ_at_joint_percentile(values)
            by_instance[idx] = {
                "instance_index": idx,
                "best_occ": best,
                "occ_p99_9": p999,
            }

        recap_path = shard_dir / "eval_best_occ_recap.json"
        if not recap_path.is_file():
            continue
        with open(recap_path, encoding="utf-8") as handle:
            recap = json.load(handle)
        for row in recap.get("instances") or []:
            idx = int(row["instance_index"])
            if idx in by_instance:
                continue
            best = float(row["best_occ"])
            by_instance[idx] = {
                "instance_index": idx,
                "best_occ": best,
                "occ_p99_9": None,
            }

    instances = [by_instance[k] for k in sorted(by_instance)]
    best_scores = [
        float(row["best_occ"])
        for row in instances
        if row.get("best_occ") is not None and np.isfinite(float(row["best_occ"]))
    ]
    p999_scores = [
        float(row["occ_p99_9"])
        for row in instances
        if row.get("occ_p99_9") is not None and np.isfinite(float(row["occ_p99_9"]))
    ]

    out: dict = {
        "summary_file": BENCH_JOINT_SUMMARY_FILENAME,
        "occupancy_percentile": BENCH_JOINT_OCC_PERCENTILE,
        "occupancy_percentile_note": "99.9th percentile (top 0.1% / 999th per-mille of rollouts)",
        "num_instances": len(instances),
        "mean_best_occ": float(np.mean(best_scores)) if best_scores else None,
        "mean_occ_p99_9": float(np.mean(p999_scores)) if p999_scores else None,
        "instances": instances,
        "shard_world_size": int(shard_world_size),
    }
    out.update(_load_recap_metadata(shard_dirs))
    return out


def write_bench_joint_summary(run_root: Path, shard_world_size: int) -> Path:
    payload = build_bench_joint_summary(run_root, shard_world_size)
    run_root.mkdir(parents=True, exist_ok=True)
    out_path = run_root / BENCH_JOINT_SUMMARY_FILENAME
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"Saved joint eval summary ({payload['num_instances']} instances, "
        f"mean_best_occ={payload['mean_best_occ']}, "
        f"mean_occ_p99_9={payload['mean_occ_p99_9']}): {out_path}",
        flush=True,
    )
    return out_path
