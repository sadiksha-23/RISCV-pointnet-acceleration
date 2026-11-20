import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# RadiusFPS N sweep
# Fixed M = 128
# ============================================================

N = np.array([1024, 10000, 20000, 50000, 100000])

voxel_n = np.array([
    121176,
    1199715,
    2411587,
    6011879,
    12016940
], dtype=float)

x_n = np.array([
    2304,
    9849,
    12289,
    34747,
    65995
], dtype=float)

y_n = np.array([
    913,
    4752,
    9166,
    21458,
    50268
], dtype=float)

z_n = np.array([
    0,
    1,
    148,
    173,
    301
], dtype=float)

full_n = np.array([
    5655,
    55683,
    106810,
    281743,
    566496
], dtype=float)

total_n = voxel_n + x_n + y_n + z_n + full_n

voxel_pct_n = voxel_n / total_n * 100
x_pct_n = x_n / total_n * 100
y_pct_n = y_n / total_n * 100
z_pct_n = z_n / total_n * 100
full_pct_n = full_n / total_n * 100

fig, ax = plt.subplots(figsize=(10, 6))

labels_n = [f"{x:,}" for x in N]
pos = np.arange(len(N))

ax.bar(pos, voxel_pct_n, label="Voxel-pruned")
ax.bar(pos, x_pct_n, bottom=voxel_pct_n, label="X-pruned")
ax.bar(pos, y_pct_n, bottom=voxel_pct_n+x_pct_n, label="Y-pruned")
ax.bar(pos, z_pct_n, bottom=voxel_pct_n+x_pct_n+y_pct_n, label="Z-pruned")
ax.bar(
    pos,
    full_pct_n,
    bottom=voxel_pct_n+x_pct_n+y_pct_n+z_pct_n,
    label="Full distance calculated"
)

ax.set_xticks(pos)
ax.set_xticklabels(labels_n)
ax.set_xlabel("Number of Input Points (N)")
ax.set_ylabel("Percentage of Point Opportunities (%)")
ax.set_title("RadiusFPS Pruning Breakdown vs N (M = 128)")
ax.set_ylim(0, 100)
ax.legend(loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "radiusfps_pruning_N.png", dpi=300)
plt.close()


# ============================================================
# RadiusFPS M sweep
# Fixed N = 100000
# ============================================================

M = np.array([32, 64, 128, 256, 512, 1024])

voxel_m = np.array([
    2614746,
    5751068,
    12016940,
    24572504,
    49766795,
    100347857
], dtype=float)

x_m = np.array([
    21521,
    30137,
    65995,
    164985,
    393932,
    790814
], dtype=float)

y_m = np.array([
    13655,
    27287,
    50268,
    104146,
    179858,
    292018
], dtype=float)

z_m = np.array([
    0,
    0,
    301,
    4289,
    19185,
    47753
], dtype=float)

full_m = np.array([
    450078,
    491508,
    566496,
    654076,
    740230,
    821558
], dtype=float)

total_m = voxel_m + x_m + y_m + z_m + full_m

voxel_pct_m = voxel_m / total_m * 100
x_pct_m = x_m / total_m * 100
y_pct_m = y_m / total_m * 100
z_pct_m = z_m / total_m * 100
full_pct_m = full_m / total_m * 100

fig, ax = plt.subplots(figsize=(10, 6))

pos = np.arange(len(M))

ax.bar(pos, voxel_pct_m, label="Voxel-pruned")
ax.bar(pos, x_pct_m, bottom=voxel_pct_m, label="X-pruned")
ax.bar(pos, y_pct_m, bottom=voxel_pct_m+x_pct_m, label="Y-pruned")
ax.bar(pos, z_pct_m, bottom=voxel_pct_m+x_pct_m+y_pct_m, label="Z-pruned")
ax.bar(
    pos,
    full_pct_m,
    bottom=voxel_pct_m+x_pct_m+y_pct_m+z_pct_m,
    label="Full distance calculated"
)

ax.set_xticks(pos)
ax.set_xticklabels(M)
ax.set_xlabel("Number of Sampled Points (M)")
ax.set_ylabel("Percentage of Point Opportunities (%)")
ax.set_title("RadiusFPS Pruning Breakdown vs M (N = 100,000)")
ax.set_ylim(0, 100)
ax.legend(loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "radiusfps_pruning_M.png", dpi=300)
plt.close()


# ============================================================
# Full-distance reduction vs BasicFPS
# Fixed N = 100000
# ============================================================

basic_full = np.array([
    3100000,
    6300000,
    12700000,
    25500000,
    51100000,
    102300000
], dtype=float)

radius_full = full_m

reduction = (1.0 - radius_full / basic_full) * 100.0

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(pos, reduction, color="#2E86C1")

ax.set_xticks(pos)
ax.set_xticklabels(M)
ax.set_xlabel("Number of Sampled Points (M)")
ax.set_ylabel("Full-Distance Calculations Eliminated (%)")
ax.set_title("RadiusFPS Reduction in Full-Distance Calculations")
ax.set_ylim(80, 100)
ax.grid(axis="y", linestyle="--", alpha=0.3)

for bar, value in zip(bars, reduction):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.25,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(OUT / "radiusfps_distance_reduction_M.png", dpi=300)
plt.close()


# ============================================================
# Print numerical summary
# ============================================================

print()
print("===== N SWEEP PRUNING =====")
for i in range(len(N)):
    print(
        f"N={N[i]:6d}: "
        f"voxel={voxel_pct_n[i]:6.2f}%  "
        f"x={x_pct_n[i]:5.2f}%  "
        f"y={y_pct_n[i]:5.2f}%  "
        f"z={z_pct_n[i]:5.2f}%  "
        f"full={full_pct_n[i]:5.2f}%"
    )

print()
print("===== M SWEEP PRUNING =====")
for i in range(len(M)):
    print(
        f"M={M[i]:4d}: "
        f"voxel={voxel_pct_m[i]:6.2f}%  "
        f"x={x_pct_m[i]:5.2f}%  "
        f"y={y_pct_m[i]:5.2f}%  "
        f"z={z_pct_m[i]:5.2f}%  "
        f"full={full_pct_m[i]:5.2f}%"
    )

print()
print("===== FULL DISTANCE REDUCTION =====")
for i in range(len(M)):
    print(
        f"M={M[i]:4d}: "
        f"Basic={int(basic_full[i]):10d}  "
        f"Radius={int(radius_full[i]):8d}  "
        f"Reduction={reduction[i]:6.2f}%"
    )

print()
print("Graphs saved to:")
print(OUT / "radiusfps_pruning_N.png")
print(OUT / "radiusfps_pruning_M.png")
print(OUT / "radiusfps_distance_reduction_M.png")
