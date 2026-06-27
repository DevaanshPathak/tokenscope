#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS binaries must be built on a macOS host or CI runner." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="$(uname -m)"

cd "$ROOT"
python -m pip install -r requirements.txt
pyinstaller --clean --noconfirm tokenscope.spec
mv dist/tokenscope "dist/tokenscope-macos-${ARCH}"

echo "Built dist/tokenscope-macos-${ARCH}"
