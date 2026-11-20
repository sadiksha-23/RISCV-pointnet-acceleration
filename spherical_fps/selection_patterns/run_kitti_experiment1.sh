#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python3 "spherical_fps/selection_patterns/kitti_fps_structure.py" \
  --bin "datasets/kitti/0000000000.bin" \
  --n 100000 --m 128 \
  --out "spherical_fps/selection_patterns/results/kitti_100k_128"
