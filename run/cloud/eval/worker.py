#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import torch
import yaml

_CLOUD = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_CLOUD))
sys.path.insert(0, str(_EVAL))

from nester import PATH_TO_PROJECT
from nester.agent.processes.eval.layout import bench_eval_run_path_segments
from gcs import (
    download_blob_from_gcs_if_exists,
    download_data_from_gcs,
    upload_file_to_gcs,
)
from nester.agent.processes.eval.bench import EvalPipeline, bench_eval_shard_triplet
from summary import (
    BENCH_JOINT_SUMMARY_FILENAME,
    bench_run_root_from_log_dir,
    bench_shard_barrier,
    write_bench_joint_summary,
)

_BENCH_DOWNLOAD_LOCK = Path("/tmp/.bench_eval_download.lock")
_BENCH_SHARD_BARRIER_TIMEOUT_S = 86_400.0


def _checkpoint_loadable(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < 4096:
        return False
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        print(f"[bench] checkpoint not readable at {path}: {exc}", flush=True)
        return False


def _sync_download_bench_assets(
    *,
    bucket: str,
    settings: dict,
    val_abs: Path,
    val_db_blob: str,
    run_root: Path,
    ckpt_name: str,
    gcs_prefix: str,
) -> None:
    """Serialize GCS pulls across torchrun ranks (parallel writes corrupt shared .pt files)."""
    ckpt_path = run_root / "models" / ckpt_name
    _BENCH_DOWNLOAD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(_BENCH_DOWNLOAD_LOCK, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            skip_dl = bool(settings.get("skip_download_if_present", False))
            blob = str(val_db_blob or "").strip()
            if blob and (not skip_dl or not val_abs.is_file()):
                download_data_from_gcs(bucket, blob, str(val_abs))
            elif not val_abs.is_file():
                raise FileNotFoundError(
                    f"Validation DB missing at {val_abs} and no GCS blob configured "
                    f"(set cloud.val_db_blob in eval config.yaml)."
                )

            if not skip_dl or not _checkpoint_loadable(ckpt_path):
                if ckpt_path.is_file():
                    print(
                        f"[bench] removing incomplete checkpoint before re-download: {ckpt_path}",
                        flush=True,
                    )
                    ckpt_path.unlink(missing_ok=True)
                ckpt_blob = f"{gcs_prefix.rstrip('/')}/models/{ckpt_name}"
                download_data_from_gcs(bucket, ckpt_blob, str(ckpt_path))

            train_cfg_local = run_root / "train_config.yaml"
            if not skip_dl or not train_cfg_local.is_file():
                train_cfg_blob = f"{gcs_prefix.rstrip('/')}/train_config.yaml"
                download_blob_from_gcs_if_exists(bucket, train_cfg_blob, str(train_cfg_local))

            _ensure_checkpoint_present_after_gcs_pull(run_root, ckpt_name, bucket, gcs_prefix)
            if not _checkpoint_loadable(ckpt_path):
                raise RuntimeError(
                    f"Checkpoint at {ckpt_path} is missing or corrupt after download from "
                    f"gs://{bucket}/{gcs_prefix}/. Delete {ckpt_path} and re-run eval."
                )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _panel_from_family_size_range(fsr):
    if not fsr or len(fsr) < 2:
        return None
    hi = int(fsr[1])
    return f"P{hi}"


def _infer_eval_panel(config):
    data = config.get("data", {})
    model = config.get("model", {})
    if data.get("eval_panel"):
        return str(data["eval_panel"]).strip("/")
    tag = _panel_from_family_size_range(data.get("family_size_range"))
    if tag:
        return tag
    train_cfg = None
    mounted = model.get("mounted_folder", "data/runs/")
    mf = model.get("model_folder", "").strip("/")
    if str(mounted).startswith("/"):
        run_root = Path(mounted) / mf
    else:
        run_root = Path(PATH_TO_PROJECT) / mounted / mf
    train_path = run_root / "train_config.yaml"
    if train_path.is_file():
        with open(train_path) as f:
            train_cfg = yaml.safe_load(f)
        tag = _panel_from_family_size_range(
            (train_cfg or {}).get("environment", {}).get("family_size_range")
        )
        if tag:
            return tag
    ckpt_path = run_root / "models" / model.get("checkpoint_file", "best.pt")
    if ckpt_path.is_file():
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        tag = _panel_from_family_size_range(cfg.get("environment", {}).get("family_size_range"))
        if tag:
            return tag
    val_local = data.get("val_db_local") or ""
    val_blob = data.get("val_db_blob") or ""
    for raw in (val_local, val_blob):
        norm = "/" + str(raw).replace("\\", "/").strip("/") + "/"
        if "/P8/" in norm:
            return "P8"
        if "/P4/" in norm:
            return "P4"
    return "P4"


def run_bench_eval(config):
    cloud = config["cloud"]
    data = config["data"]
    model = config["model"]
    settings = config.get("settings", {})
    bucket = cloud["bucket"]

    val_local = data["val_db_local"]
    val_abs = Path(val_local) if val_local.startswith("/") else Path(PATH_TO_PROJECT) / val_local
    val_abs.parent.mkdir(parents=True, exist_ok=True)

    mounted = model.get("mounted_folder", "data/runs/")
    model_folder = str(model["model_folder"]).strip("/")
    ckpt_name = model.get("checkpoint_file", "best.pt")

    if str(mounted).startswith("/"):
        run_root = Path(mounted) / model_folder
    else:
        run_root = Path(PATH_TO_PROJECT) / str(mounted).strip("/") / model_folder
    run_root.mkdir(parents=True, exist_ok=True)

    gcs_prefix = f"runs/{model_folder}"
    _sync_download_bench_assets(
        bucket=bucket,
        settings=settings,
        val_abs=val_abs,
        val_db_blob=str(cloud.get("val_db_blob") or data.get("val_db_blob") or ""),
        run_root=run_root,
        ckpt_name=ckpt_name,
        gcs_prefix=gcs_prefix,
    )

    train_cfg = run_root / "train_config.yaml"
    if train_cfg.is_file() and not data.get("family_size_range"):
        with open(train_cfg) as f:
            tc = yaml.safe_load(f)
        env = (tc or {}).get("environment", {})
        if "family_size_range" in env:
            data = dict(data)
            data["family_size_range"] = env["family_size_range"]

    cfg_out = dict(config)
    cfg_out.setdefault("data", {})["val_db_local"] = str(val_abs)
    cfg_out.setdefault("model", {})["mounted_folder"] = str(mounted).rstrip("/") + "/"
    cfg_out["model"]["model_folder"] = model_folder
    cfg_out["model"]["checkpoint_file"] = ckpt_name

    bench = EvalPipeline(cfg_out)
    log_dir = bench.run_eval()
    shard_rank, shard_ws, _ = bench_eval_shard_triplet()
    run_root = bench_run_root_from_log_dir(log_dir, shard_rank, shard_ws)

    eval_yaml_path = os.path.join(log_dir, "bench_config.yaml")
    with open(eval_yaml_path, "w") as yf:
        yaml.safe_dump(cfg_out, yf, default_flow_style=False, sort_keys=False)

    bench_shard_barrier(run_root, shard_rank, shard_ws, timeout_s=_BENCH_SHARD_BARRIER_TIMEOUT_S)
    joint_summary_path: Path | None = None
    if shard_rank == 0:
        joint_summary_path = write_bench_joint_summary(run_root, shard_ws)

    if data.get("upload_eval_results", cloud.get("upload_eval_results", True)):
        panel = _infer_eval_panel(cfg_out)
        path_segs = bench_eval_run_path_segments(
            checkpoint_file=cfg_out.get("model", {}).get("checkpoint_file"),
            settings=cfg_out.get("settings", {}),
            data=cfg_out.get("data", {}),
        )
        gcs_eval_prefix = f"runs/{model_folder}/eval/{panel}/{'/'.join(path_segs)}"
        gcs_shard_prefix = (
            f"rank_{shard_rank:02d}_of_{shard_ws:02d}/" if shard_ws > 1 else ""
        )
        for root, _, files in os.walk(log_dir):
            for name in files:
                lp = os.path.join(root, name)
                rel = os.path.relpath(lp, log_dir)
                blob_path = f"{gcs_eval_prefix}/{gcs_shard_prefix}{rel.replace(os.sep, '/')}"
                upload_file_to_gcs(bucket, blob_path, lp)
        if shard_rank == 0 and joint_summary_path is not None and joint_summary_path.is_file():
            blob_path = f"{gcs_eval_prefix}/{BENCH_JOINT_SUMMARY_FILENAME}"
            upload_file_to_gcs(bucket, blob_path, str(joint_summary_path))
        print(
            f"Uploaded eval artifacts (rank {shard_rank}/{shard_ws}) to "
            f"gs://{bucket}/{gcs_eval_prefix}/",
            flush=True,
        )


def _models_pt_listing(models_dir: Path) -> str:
    if not models_dir.is_dir():
        return f"no directory at {models_dir}"
    names = sorted(p.name for p in models_dir.glob("*.pt"))
    return ", ".join(names) if names else "no *.pt under models/"


def _ensure_checkpoint_present_after_gcs_pull(
    run_root: Path,
    ckpt_name: str,
    bucket: str,
    gcs_prefix: str,
) -> None:
    ckpt_path = run_root / "models" / ckpt_name
    if ckpt_path.is_file():
        return
    models_dir = run_root / "models"
    hint = ""
    train_path = run_root / "train_config.yaml"
    if train_path.is_file():
        try:
            with open(train_path, encoding="utf-8") as f:
                tc = yaml.safe_load(f)
            exp = (tc or {}).get("training", {}).get("exp_name")
            rid = run_root.name
            if exp and rid:
                hint = (
                    f"\nHint: train_config.yaml has training.exp_name={exp!r}. "
                    f"Training uploads to gs://{bucket}/runs/{exp}/{rid}/. "
                    f"Set model.model_folder to \"{exp}/{rid}\" (exp_name + run timestamp folder)."
                )
        except Exception:
            pass
    raise FileNotFoundError(
        f"Checkpoint missing at {ckpt_path} after gs://{bucket}/{gcs_prefix}/*. "
        f"models/: {_models_pt_listing(models_dir)}.{hint}"
    )


def _load_config_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suf = path.suffix.lower()
    with open(path, encoding="utf-8") as f:
        if suf in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def _load_bench_config_from_cli() -> dict:
    """GCP Batch sets CONFIG_JSON; local runs use --config path to eval config.yaml."""
    raw_json = os.environ.get("CONFIG_JSON")
    if raw_json is not None and str(raw_json).strip():
        return json.loads(raw_json)

    parser = argparse.ArgumentParser(description="Bench eval")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="run/cloud/eval/config.yaml",
        help="Path to bench eval config.yaml (or .json)",
    )
    args = parser.parse_args()
    return _load_config_file(Path(args.config))


if __name__ == "__main__":
    run_bench_eval(_load_bench_config_from_cli())
