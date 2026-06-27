#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="tokenscope-linux-builder"

docker build -f "$ROOT/packaging/Dockerfile.linux" -t "$IMAGE" "$ROOT"
mkdir -p "$ROOT/dist"
container_id="$(docker create "$IMAGE")"
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
docker cp "$container_id:/out/tokenscope-linux-x86_64" "$ROOT/dist/tokenscope-linux-x86_64"

echo "Built dist/tokenscope-linux-x86_64"
