#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$DIR/config.yaml}"
ROOT="$(cd "$DIR/../../.." && pwd)"
IMAGE="$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG'))['cloud']; print(f\"{c['region']}-docker.pkg.dev/{c['project_id']}/{c['repository_name']}/{c['docker_tag']}\")")"
docker build -t "$IMAGE" -f "$ROOT/run/cloud/Dockerfile" "$ROOT"
docker push "$IMAGE"
python3 "$ROOT/run/cloud/eval/deploy.py" "$CONFIG"
