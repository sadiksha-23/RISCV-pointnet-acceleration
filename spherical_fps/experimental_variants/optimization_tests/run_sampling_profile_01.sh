#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/spherical_fps/experimental_variants/optimization_tests/spherical_fps_01_sampling_profile.cpp"
GEM5="$ROOT/gem5/build/RISCV/gem5.opt"
CPU="$ROOT/benchmark/cpu/riscv_fps_comparison.py"
M5_LIB="$ROOT/gem5/util/m5/build/riscv/out/libm5.a"
BIN="$ROOT/bin/spherical_fps_01_sampling_profile.riscv"
OUT="$ROOT/spherical_fps/experimental_variants/optimization_tests/results/riscv_gem5/01_sampling_diagnostic_N100000_M128_A24_E8_R6"
DATA="${1:-$ROOT/datasets/kitti/0000000000.bin}"

for path in "$SRC" "$GEM5" "$CPU" "$M5_LIB" "$DATA"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: file missing: $path" >&2
        echo "If KITTI data is in a different location, pass its full path as the first argument." >&2
        exit 1
    fi
done

mkdir -p "$ROOT/bin" "$OUT"

echo 'Compiling SphericalFPS 01 (sampling diagnostics) for RISC-V...'
riscv64-linux-gnu-g++ -O3 -std=c++17 -static -ffp-contract=off \
    -DENABLE_GEM5 -DAZ_BINS=24 -DEL_BINS=8 -DRAD_BINS=6 \
    -I"$ROOT/gem5/include" "$SRC" "$M5_LIB" -o "$BIN"

echo 'Running one gem5 experiment: N=100000 M=128...'
"$GEM5" --outdir="$OUT" "$CPU" "$BIN" "$DATA" 100000 128 \
    2>&1 | tee "$OUT/output.txt"

python3 - "$OUT" <<'PY'
from pathlib import Path
import sys,re
folder=Path(sys.argv[1]); stats=(folder/'stats.txt').read_text()
log=(folder/'output.txt').read_text()
ticks=[int(x) for x in re.findall(r'^simTicks\s+(\d+)',stats,re.M)]
if len(ticks)<2:
    raise SystemExit('ERROR: did not find two gem5 ROI intervals in stats.txt')
vals={k:int(v) for k,v in re.findall(r'^PROFILE_(\w+)=([0-9]+)$',log,re.M)}
unit=re.search(r'^PROFILE_UNIT=(\S+)$',log,re.M)
if not unit or unit.group(1) != 'cycles':
    raise SystemExit('ERROR: expected RISC-V cycle counters')
if not all(k in vals for k in ('BOUND_TICKS','POINT_TICKS','WINNER_TICKS','TOTAL_TICKS')):
    raise SystemExit('ERROR: diagnostic cycle counts missing from output.txt')
print('\n==== SPHERICALFPS 01 SAMPLING DIAGNOSTICS ====')
print(f'Preparation, gem5 ROI: {ticks[0]/1e9:.4f} ms')
print(f'Sampling, gem5 ROI (instrumented): {ticks[1]/1e9:.4f} ms')
print('\nRISC-V rdcycle breakdown (diagnostic, includes timer overhead):')
for label,key in [('Region bound + skip test','BOUND_TICKS'),('Point updates + local best','POINT_TICKS'),('Global winner selection','WINNER_TICKS')]:
    cycles=vals[key]
    print(f'  {label:<30} {cycles:>12,} cycles  ({cycles/2e6:.4f} ms at 2 GHz)')
accounted=sum(vals[k] for k in ('BOUND_TICKS','POINT_TICKS','WINNER_TICKS'))
rest=vals['TOTAL_TICKS']-accounted
print(f'  {"Other + instrumentation":<30} {rest:>12,} cycles  ({rest/2e6:.4f} ms at 2 GHz)')
print(f'  {"Full rdcycle sampling":<30} {vals["TOTAL_TICKS"]:>12,} cycles  ({vals["TOTAL_TICKS"]/2e6:.4f} ms at 2 GHz)')
for label,key in [('Bound checks','BOUND_CALLS'),('Regions with point updates','POINT_REGION_VISITS'),('Visited points','POINTS_VISITED')]:
    print(f'  {label}: {vals.get(key,"unknown")}')
print('\nReference uninstrumented 01: 24.1185 ms total (12.5039 prep + 11.6146 sampling).')
print('Reference RadiusFPS: 23.5523 ms total.')
print('Do NOT compare instrumented run time as a speedup: fine-grained rdcycle calls add overhead.')
print(f'Raw output: {folder}')
PY
