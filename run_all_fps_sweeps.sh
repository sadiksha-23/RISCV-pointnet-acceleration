#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/lsadi/Documents/GitHub/RISCV-pointnet-acceleration"
cd "$ROOT"

GEM5="./gem5/build/RISCV/gem5.opt"
CONFIG="benchmark/cpu/riscv_fps_comparison.py"
DATA="datasets/kitti/0000000000.bin"
LIBM5="./gem5/util/m5/build/riscv/out/libm5.a"

SPH="spherical_fps/main_implementation/results"
RAD="benchmark/stats/experiments/fps/RadiusFPS/kitti/RVV_sweeps"

N_VALUES=(1024 2048 4096 10000 20000 50000 100000)
M_VALUES=(32 64 128 256 512 1024)

mkdir -p bin "$SPH" "$RAD"

for file in "$GEM5" "$CONFIG" "$DATA" "$LIBM5" \
    bin/spherical_fps.riscv bin/spherical_fps_rvv.riscv; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing $file"
        exit 1
    fi
done

echo "======================================"
echo "Preparing RadiusFPS RVV sweep version"
echo "======================================"

python3 - <<'PY'
from pathlib import Path
import re

src = Path("benchmark/src/fps/RadiusFPS/kitti/radius_fps_vector.cpp")
dst = Path("benchmark/src/fps/RadiusFPS/kitti/radius_fps_rvv_sweep.cpp")

s = src.read_text()

# Remove internal profiling checkpoints from the sweep copy.
s = re.sub(r'\bm5_(?:reset|dump)_stats\s*\(\s*0\s*,\s*0\s*\)\s*;', '', s)

# Accept command-line arguments.
assert "int main()" in s
s = s.replace(
    "int main() {",
    '''int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <KITTI> <N> <M>\\n", argv[0]);
        return 2;
    }''',
    1
)

# Replace the hardcoded KITTI path.
s, count = re.subn(
    r'const char\s*\*\s*filename\s*=\s*"[^"]+";',
    'const char *filename = argv[1];',
    s, count=1
)
assert count == 1

# Replace hardcoded point and sample counts.
s, count_n = re.subn(r'\bint n = 100000;', 'int n = atoi(argv[2]);', s, count=1)
s, count_m = re.subn(r'\bint m = 128;', 'int m = atoi(argv[3]);', s, count=1)
assert count_n == 1 and count_m == 1

# Validate arguments.
assert "float voxel_size = 2.0f;" in s
s = s.replace(
    "float voxel_size = 2.0f;",
    '''float voxel_size = 2.0f;
    if (n < 1 || m < 1 || m > n) {
        fprintf(stderr, "Invalid N or M\\n");
        return 2;
    }''',
    1
)

# One measurement interval covering preparation and sampling.
marker = "const ptrdiff_t dataset_stride = 3 * sizeof(float);"
assert marker in s
s = s.replace(marker, marker + "\n    m5_reset_stats(0, 0);", 1)

marker = 'printf("POINT_COUNT=%d\\n", n);'
assert marker in s
s = s.replace(marker, 'm5_dump_stats(0, 0);\n    ' + marker, 1)

assert len(re.findall(r'\bm5_reset_stats\s*\(', s)) == 1
assert len(re.findall(r'\bm5_dump_stats\s*\(', s)) == 1

dst.write_text(s)
print(f"Created {dst}")
print("Original RadiusFPS RVV source unchanged.")
PY

riscv64-linux-gnu-g++ -O3 -std=c++17 -static \
    -march=rv64gcv -mabi=lp64d -ffp-contract=off \
    -I./gem5/include \
    benchmark/src/fps/RadiusFPS/kitti/radius_fps_rvv_sweep.cpp \
    "$LIBM5" \
    -o bin/radius_fps_rvv_sweep.riscv

echo "RadiusFPS RVV compiled successfully."

run_one() {
    local name="$1"
    local binary="$2"
    local dir="$3"
    local n="$4"
    local m="$5"

    # Reuse SphericalFPS baseline measured earlier.
    if [[ "$name" != "RadiusFPS_RVV" && "$n" == 100000 && "$m" == 128 ]]; then
        echo "SKIP $name N=$n M=$m — existing baseline"
        return
    fi

    if [[ -d "$dir" ]]; then
        if [[ -s "$dir/stats.txt" ]] &&
           grep -q "End Simulation Statistics" "$dir/stats.txt" &&
           [[ -s "$dir/run_output.txt" ]] &&
           grep -Fq "FPS_INDEX[$((m-1))]=" "$dir/run_output.txt"; then
            echo "SKIP $name N=$n M=$m — completed"
            return
        fi
        echo "ERROR: Incomplete or unverified folder: $dir"
        echo "Inspect it before retrying. Nothing was overwritten."
        exit 1
    fi

    mkdir -p "$(dirname "$dir")"

    echo
    echo "======================================"
    echo "$name | N=$n | M=$m"
    echo "======================================"

    "$GEM5" \
        --outdir="$dir" \
        "$CONFIG" "$binary" "$DATA" "$n" "$m" \
        > "${dir}_run.log" 2>&1

    cp "${dir}_run.log" "$dir/run_output.txt"

    if [[ ! -s "$dir/stats.txt" ]] ||
       ! grep -q "End Simulation Statistics" "$dir/stats.txt" ||
       ! grep -Fq "FPS_INDEX[$((m-1))]=" "$dir/run_output.txt"; then
        echo "ERROR: Run did not produce complete results: $dir"
        exit 1
    fi

    awk '/^simTicks[[:space:]]/ {
        printf "Time: %.4f ms\n", $2/1e9
        exit
    }' "$dir/stats.txt"
}

run_sweeps() {
    local name="$1"
    local binary="$2"
    local base="$3"

    echo
    echo "######## $name — N SWEEP ########"

    for n in "${N_VALUES[@]}"; do
        run_one "$name" "$binary" \
            "$base/N_sweep/N${n}_M128" "$n" 128
    done

    echo
    echo "######## $name — M SWEEP ########"

    for m in "${M_VALUES[@]}"; do
        run_one "$name" "$binary" \
            "$base/M_sweep/N100000_M${m}" 100000 "$m"
    done
}

# Run sequentially, never in parallel.
run_sweeps "SphericalFPS_Scalar" \
    "bin/spherical_fps.riscv" \
    "$SPH/spherical_fps"

run_sweeps "SphericalFPS_RVV" \
    "bin/spherical_fps_rvv.riscv" \
    "$SPH/spherical_fps_rvv"

run_sweeps "RadiusFPS_RVV" \
    "bin/radius_fps_rvv_sweep.riscv" \
    "$RAD"

echo
echo "======================================"
echo "ALL THREE SWEEPS COMPLETED"
echo "======================================"
