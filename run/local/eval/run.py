#!/usr/bin/env python3
"""Local bench eval."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CLOUD = _REPO / "run" / "cloud"
_EVAL = _CLOUD / "eval"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_CLOUD))
sys.path.insert(0, str(_EVAL))

from worker import _load_bench_config_from_cli, run_bench_eval  # noqa: E402


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", str(Path(__file__).resolve().parent / "config.yaml")])
    run_bench_eval(_load_bench_config_from_cli())


if __name__ == "__main__":
    main()
