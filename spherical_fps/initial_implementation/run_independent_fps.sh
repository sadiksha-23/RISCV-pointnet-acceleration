#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
OUT="$SCRIPT_DIR/results"
mkdir -p "$OUT"
DATA="datasets/kitti/0000000000.bin"
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Dataset not found: $DATA" >&2
  exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "ERROR: g++ required; run from Ubuntu/WSL" >&2
  exit 1
fi
printf 'Compiling independent RadiusFPS versus SphericalFPS benchmark...\n'
mkdir -p "$ROOT/bin"
g++ -O3 -std=c++17 -Wall -Wextra -o "$ROOT/bin/compare_independent_fps" "$SCRIPT_DIR/compare_independent_fps.cpp"
printf '\nTesting centroid-origin standalone SphericalFPS versus original RadiusFPS...\n'
"$ROOT/bin/compare_independent_fps" --dataset "$DATA" --n 100000 --m 128 --repeats 5 --origin centroid | tee "$OUT/centroid_comparison.txt"
printf '\nTesting sensor-origin standalone SphericalFPS versus original RadiusFPS...\n'
"$ROOT/bin/compare_independent_fps" --dataset "$DATA" --n 100000 --m 128 --repeats 5 --origin sensor | tee "$OUT/sensor_comparison.txt"
printf '\nSaved logs to %s\n' "$OUT"
