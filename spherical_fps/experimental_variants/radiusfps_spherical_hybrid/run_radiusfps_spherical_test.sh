#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

if command -v g++ >/dev/null 2>&1; then
  CXX=g++
elif command -v clang++ >/dev/null 2>&1; then
  CXX=clang++
else
  echo "ERROR: No C++ compiler on PATH. Run in Ubuntu/WSL with g++ installed (sudo apt install g++)." >&2
  exit 1
fi

EXE="$SCRIPT_DIR/radiusfps_spherical_test"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE="$EXE.exe" ;;
esac

mkdir -p "$SCRIPT_DIR/radius_spherical_results"
OUT="$SCRIPT_DIR/radius_spherical_results"
DATA="datasets/kitti/0000000000.bin"
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Cannot find $ROOT/$DATA" >&2
  exit 1
fi

printf '\nCompiling native host scalar RadiusFPS comparison with %s...\n' "$CXX"
"$CXX" -O3 -std=c++17 -Wall -Wextra -o "$EXE" "$SCRIPT_DIR/radiusfps_spherical_test.cpp"

printf '\n=== BENCHMARK (correctness verified by exact index equality; no bound-scan overhead) ===\n'
"$EXE" --dataset "$DATA" --n 100000 --m 128 --voxel-size 2 --origin centroid --repeats 3 \
  | tee "$OUT/centroid_timing.txt"

printf '\n=== SPHERICAL BOUND VALIDATION (separate from benchmark timings) ===\n'
"$EXE" --dataset "$DATA" --n 100000 --m 128 --voxel-size 2 --origin centroid --repeats 1 --verify-bounds \
  | tee "$OUT/centroid_bound_validation.txt"

printf '\n=== OPTIONAL SENSOR-ORIGIN BENCHMARK ===\n'
"$EXE" --dataset "$DATA" --n 100000 --m 128 --voxel-size 2 --origin sensor --repeats 3 \
  | tee "$OUT/sensor_timing.txt"

printf '\nDONE. Send centroid_timing.txt and sensor_timing.txt back for comparison.\n'
printf 'Logs saved in %s\n' "$OUT"
