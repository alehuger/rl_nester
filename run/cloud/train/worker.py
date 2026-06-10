#!/usr/bin/env python3
"""Cloud training worker (torchrun). Rank 0 pulls GCS data; all ranks run PPOTrainer."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

_CLOUD = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_CLOUD))

from gcs import download_data_from_gcs, download_dir_from_gcs  # noqa: E402
from nester import PATH_TO_PROJECT  # noqa: E402
from nester.agent.processes.train.ppo import PPOTrainer  # noqa: E402

_DATA_READY_MARKER = Path("/tmp/.data_ready")


def run_training(config: dict) -> None:
    cloud = config["cloud"]
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        download_data_from_gcs(
            cloud["bucket"], cloud["data_path"], config["environment"]["data_path"]
        )

        pretrained_path = cloud.get("pretrained_model_path")
        if pretrained_path is not None:
            model_cfg = config.get("model", {})
            mounted_folder = model_cfg.get("mounted_folder", "data/runs/")
            model_pretrained_path = model_cfg.get("pretrained_model_path", pretrained_path)
            bucket_prefix = f"runs/{pretrained_path.rstrip('/')}"
            local_base = (
                Path(PATH_TO_PROJECT) / mounted_folder
                if not str(mounted_folder).startswith("/")
                else Path(mounted_folder)
            )
            local_dir = local_base / model_pretrained_path.lstrip("/") / "models"
            download_dir_from_gcs(cloud["bucket"], f"{bucket_prefix}/models", str(local_dir))

        config_path = Path("/tmp/train_config.yaml")
        model_section = dict(config.get("model", {}))
        if cloud.get("pretrained_checkpoint") is not None:
            model_section["pretrained_checkpoint"] = cloud["pretrained_checkpoint"]
        with open(config_path, "w") as f:
            yaml.dump(
                {
                    "cloud": config.get("cloud", {}),
                    "training": config.get("training", {}),
                    "model": model_section,
                    "environment": config.get("environment", {}),
                },
                f,
            )
        _DATA_READY_MARKER.touch()
    else:
        while not _DATA_READY_MARKER.exists():
            time.sleep(0.5)

    trainer = PPOTrainer(Path("/tmp/train_config.yaml"))
    trainer.setup()
    trainer.train()

    if rank == 0:
        _DATA_READY_MARKER.unlink(missing_ok=True)
        print("Training completed!")


if __name__ == "__main__":
    raw = os.environ.get("CONFIG_JSON") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not raw:
        raise ValueError("Set CONFIG_JSON or pass config JSON as argv[1]")
    run_training(json.loads(raw))
