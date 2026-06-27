#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  scripts/build-linux-docker.sh
else
  echo "Docker is unavailable or not connected; falling back to local PyInstaller." >&2
  python -m PyInstaller --clean --noconfirm tokenscope.spec
  mkdir -p dist
  if [[ -f dist/tokenscope ]]; then
    mv dist/tokenscope dist/tokenscope-linux-x86_64
  fi
  echo "Built dist/tokenscope-linux-x86_64"
fi
