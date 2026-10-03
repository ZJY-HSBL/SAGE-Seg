#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/voc.yaml --data-root "${1:?Usage: $0 /path/to/VOC2012}"
