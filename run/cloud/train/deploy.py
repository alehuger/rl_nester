#!/usr/bin/env python3
import sys
from pathlib import Path

_CLOUD = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CLOUD))

from batch import get_docker_image, load_config, submit_training_job


def main() -> None:
    if len(sys.argv) <= 1:
        raise SystemExit("Usage: python run/cloud/train/deploy.py [run/cloud/train/config.yaml]")
    config = load_config(sys.argv[1])
    image = get_docker_image(config)
    c = config["cloud"]
    print("Deploying training job…")
    print("Image:", image)
    print(f"Machine: {c['machine_type']} + {c['gpu_count']}x {c['gpu_type']}")
    submit_training_job(config, image)


if __name__ == "__main__":
    main()
