from pathlib import Path
import subprocess
import re
import statistics

# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
ORIGINAL = FOLDER / "compare_lower_bound_fps.cpp"
OPTIMIZED = FOLDER / "compare_cached_bounds.cpp"

BIN = ROOT / "bin"
BIN.mkdir(exist_ok=True)

# ============================================================
# 2. CREATE OPTIMIZED C++ VERSION
# ============================================================

code = ORIGINAL.read_text(encoding="utf-8")

# Change the function to accept precomputed selected radius.
old_signature = """int newest,
    const double origin[3]
) {"""

new_signature = """int newest,
    const double origin[3],
    double selected_radius
) {"""

assert code.count(old_signature) == 1, "Function signature not found"
code = code.replace(old_signature, new_signature, 1)

# Remove repeated radial-distance calculation.
old_block = """    double ox = sx - origin[0];
    double oy = sy - origin[1];
    double oz = sz - origin[2];

    double selected_radius = std::sqrt(
        ox*ox + oy*oy + oz*oz
    );

"""

assert code.count(old_block) == 1, "Radial calculation not found"
code = code.replace(old_block, "", 1)

# Compute selected radius once per iteration.
old_iteration = """        int newest = result.indices[step - 1];

        //"""

new_iteration = """        int newest = result.indices[step - 1];

        size_t p = (size_t)newest * 3;
        double ox = (double)xyz[p] - origin[0];
        double oy = (double)xyz[p + 1] - origin[1];
        double oz = (double)xyz[p + 2] - origin[2];

        double selected_radius = std::sqrt(
            ox*ox + oy*oy + oz*oz
        );

        //"""

assert code.count(old_iteration) == 1, "Iteration location not found"
code = code.replace(old_iteration, new_iteration, 1)

# Pass the cached value to the bound function.
old_call = """                g, xyz, newest, origin
            );"""

new_call = """                g, xyz, newest, origin, selected_radius
            );"""

assert code.count(old_call) == 1, "Bound call not found"
code = code.replace(old_call, new_call, 1)

OPTIMIZED.write_text(code, encoding="utf-8")

print("Created optimized C++ source:", OPTIMIZED)

# ============================================================
# 3. COMPILE BOTH VERSIONS
# ============================================================

def compile_cpp(source, executable):
    subprocess.run(
        [
            "g++", "-O3", "-std=c++17",
            "-Wall", "-Wextra",
            str(source), "-o", str(executable)
        ],
        check=True
    )

original_bin = BIN / "lower_bound_original"
optimized_bin = BIN / "lower_bound_cached"

compile_cpp(ORIGINAL, original_bin)
compile_cpp(OPTIMIZED, optimized_bin)

# ============================================================
# 4. RUN HEAD-TO-HEAD TEST
# ============================================================

results = {"Original": [], "Cached": []}
sampling = {"Original": [], "Cached": []}

for repeat in range(7):
    order = [
        ("Original", original_bin),
        ("Cached", optimized_bin),
    ]

    if repeat % 2:
        order.reverse()

    for name, executable in order:
        output = subprocess.check_output(
            [
                str(executable),
                str(ROOT / "datasets/kitti/0000000000.bin"),
                "100000", "128", "1"
            ],
            text=True,
            cwd=ROOT
        )

        total = re.search(
            r"LowerBoundFPS total:\s*([\d.]+)", output
        )
        sample = re.search(
            r"LowerBoundFPS sampling:\s*([\d.]+)", output
        )

        if not total or not sample:
            raise RuntimeError("Could not read timing results")

        if "CORRECTNESS: PASS" not in output:
            raise RuntimeError("FPS correctness failed")

        results[name].append(float(total.group(1)))
        sampling[name].append(float(sample.group(1)))

# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n=== CACHED BOUND EXPERIMENT ===")
print("N=100000, M=128, grid=24x8x6")
print("7 runs per version\n")

for name in ["Original", "Cached"]:
    print(f"{name}:")
    print(f"  Sampling: {statistics.median(sampling[name]):.3f} ms")
    print(f"  Total:    {statistics.median(results[name]):.3f} ms")

old_total = statistics.median(results["Original"])
new_total = statistics.median(results["Cached"])

old_sample = statistics.median(sampling["Original"])
new_sample = statistics.median(sampling["Cached"])

print("\nSampling speedup:", round(old_sample / new_sample, 3), "x")
print("Total speedup:", round(old_total / new_total, 3), "x")
print("Correctness: PASS")