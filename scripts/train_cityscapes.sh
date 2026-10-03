#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/cityscapes.yaml --data-root "${1:?Usage: $0 /path/to/cityscapes}"
