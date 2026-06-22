#!/usr/bin/env bash
set -euo pipefail

pyinstaller --onefile --name tokenscope main.py

echo "Built dist/tokenscope"
