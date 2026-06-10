#!/usr/bin/env python3
"""Local PPO training."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT = Path(__file__).resolve().parent / "config.yaml"
sys.path.insert(0, str(_REPO))

from nester.agent.processes.train.ppo import PPOTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Local training")
    p.add_argument("config", nargs="?", default=str(_DEFAULT))
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("cloud", None)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        path = Path(tmp.name)

    trainer = PPOTrainer(path)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
