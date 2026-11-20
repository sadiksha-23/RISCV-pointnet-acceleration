#!/usr/bin/env bash
# From project root: bash benchmark/scripts/run_nsfps_kitti.sh [X_BLOCKS Y_BLOCKS Z_BLOCKS]
# First tests native x86 for quick correctness sanity, then runs unmodified C++ NS-FPS in RISC-V gem5.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
XB="${1:-16}"
YB="${2:-16}"
ZB="${3:-16}"
N=100000
M=128
DATASET="$ROOT/datasets/kitti/0000000000.bin"
UPSTREAM="$ROOT/benchmark/third_party/ns-fps"
WRAPPER="$ROOT/benchmark/src/fps/NSFPS/kitti/ns_fps_kitti.cpp"
BIN_DIR="$ROOT/benchmark/bin"
OUT="$ROOT/benchmark/analysis/nsfps_cpu_gem5/N${N}_M${M}_B${XB}x${YB}x${ZB}"
GEM5="$ROOT/gem5"
mkdir -p "$BIN_DIR" "$OUT" "$ROOT/benchmark/third_party"
for cmd in git g++ riscv64-linux-gnu-g++ python3; do
  command -v "$cmd" >/dev/null || { echo "ERROR: Missing $cmd"; exit 1; }
done
[ -f "$DATASET" ] || { echo "ERROR: Missing KITTI file $DATASET"; exit 1; }
[ -f "$WRAPPER" ] || { echo "ERROR: Missing wrapper $WRAPPER"; exit 1; }
[ -f "$GEM5/custom/riscv_benchmark.py" ] || { echo "ERROR: Missing gem5 config file"; exit 1; }
[ -f "$GEM5/util/m5/build/riscv/out/libm5.a" ] || { echo "ERROR: Missing gem5 libm5.a"; exit 1; }
if [ ! -f "$UPSTREAM/src/yuezu_fps.cpp" ]; then
  if [ -e "$UPSTREAM" ]; then echo "ERROR: Directory exists but source is missing: $UPSTREAM"; exit 1; fi
  echo "===== Cloning upstream NS-FPS (original code) ====="
  git clone --depth 1 https://github.com/satreeby/ns-fps.git "$UPSTREAM"
fi
REV="$(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null || echo unknown)"
printf 'UPSTREAM_GIT_COMMIT=%s\n' "$REV" | tee "$OUT/metadata.txt"
printf 'CONFIG=N=%s M=%s X_BLOCKS=%s Y_BLOCKS=%s Z_BLOCKS=%s\n' "$N" "$M" "$XB" "$YB" "$ZB" | tee -a "$OUT/metadata.txt"

echo '===== Compiling original NS-FPS CPU core with native g++ ====='
g++ -O3 -std=c++17 -fno-tree-vectorize \
  -I"$UPSTREAM/src" "$WRAPPER" "$UPSTREAM/src/yuezu_fps.cpp" \
  -o "$BIN_DIR/ns_fps_kitti.native"

echo '===== Native quick test (NOT comparable to gem5 timings) ====='
"$BIN_DIR/ns_fps_kitti.native" "$DATASET" "$N" "$M" "$XB" "$YB" "$ZB" \
  > "$OUT/native_output.txt"
grep -E '^(METHOD|N=|M=|BLOCKS|SAMPLES_RETURNED|NATIVE_MS|FPS_INDEX\[[0-4]\])' "$OUT/native_output.txt"
SECOND="$(grep '^FPS_INDEX\[1\]=' "$OUT/native_output.txt" | head -1 | cut -d= -f2 || true)"
if [ "$SECOND" != 3052 ]; then
  echo "WARNING: Second FPS index is '$SECOND', not independent KITTI-reference index 3052."
  echo 'Investigate correctness/tie-breaking BEFORE interpreting any benchmark speedup.'
  echo 'Skipping expensive gem5 simulation by default after failed first-step sanity check.'
  echo "Native output: $OUT/native_output.txt"
  exit 3
fi

echo '===== Compiling original NS-FPS C++ core for RISC-V gem5 ====='
riscv64-linux-gnu-g++ -O3 -fno-tree-vectorize -std=c++17 -static \
  -march=rv64gcv -mabi=lp64d -DNSFPS_GEM5 \
  -DNSFPS_XBLOCKS="$XB" -DNSFPS_YBLOCKS="$YB" -DNSFPS_ZBLOCKS="$ZB" \
  -I"$UPSTREAM/src" -I"$GEM5/include" \
  "$WRAPPER" "$UPSTREAM/src/yuezu_fps.cpp" \
  "$GEM5/util/m5/build/riscv/out/libm5.a" \
  -o "$BIN_DIR/ns_fps_kitti.riscv"

echo '===== Running gem5 N=100000 M=128 ====='
mkdir -p "$OUT/m5out"
"$GEM5/build/RISCV/gem5.opt" \
   --outdir="$OUT/m5out" \
   "$GEM5/custom/riscv_benchmark.py" \
   "$BIN_DIR/ns_fps_kitti.riscv" \
   > "$OUT/gem5.log" 2>&1
if [ -f "$OUT/m5out/stats.txt" ]; then
  echo '===== gem5 timing and instruction counts ====='
  grep -E '(^|\.)simSeconds[[:space:]]|(^|\.)simInsts[[:space:]]|(^|\.)numCycles[[:space:]]' \
    "$OUT/m5out/stats.txt" | tail -15 || true
fi
# Depending on the project config, stdout might go to gem5.log or m5out/simout.
for f in "$OUT/gem5.log" "$OUT/m5out/simout" "$OUT/m5out/system.out"; do
 if [ -f "$f" ]; then
  echo "===== FPS output: $f ====="
  grep -E '^(METHOD|N=|M=|BLOCKS|SAMPLES_RETURNED|FPS_INDEX\[[0-9]+\])' "$f" | head -15 || true
 fi
done
echo "OUTPUT_DIR=$OUT"
