#!/usr/bin/env bash
set -euo pipefail
cd "/mnt/c/Users/lsadi/Documents/PCD Research"
python3 "Polar/experiment/kitti_fps_structure.py" \
  --bin "datasets/kitti/0000000000.bin" \
  --n 100000 --m 128 \
  --out "Polar/experiment_result/KITTI_N100000_M128"
