import matplotlib.pyplot as plt
import numpy as np

# Ordered benchmark names
kernels = [
    "3NN\n(3N)",
    "Get Weights\n(weight)",
    "Interpolate\n(interpolation)",
    "Upsampling\n(pipeline)",
    "FPS\n(fps)",
    "Ball Query\n(ball_query)",
    "Max Pooling\n(maxpooling)",
    "Downsampling\n(pipeline)",
    "MLP / GN\n(mlp_gn)"
]

# Benchmark Data
scalar_seconds = [1.972e-6, 0.720e-6, 5.6405e-6, 7.781e-6, 1.9445e-6, 5.9565e-6, 36.7405e-6, 8.644e-6, 6.0636e-3]
rvv_seconds    = [1.8735e-6, 0.1845e-6, 1.564e-6,  4.064e-6, 1.4995e-6, 3.2355e-6, 3.014e-6,  5.0745e-6, 0.4405e-3]

scalar_insts = [6193, 869, 25434, 32480, 4869, 24583, 121507, 30995, 13538123]
rvv_insts    = [5121, 149, 6482,  11896, 3083, 9653,  14189,  13821, 1669207]

y = np.arange(len(kernels))
height = 0.35

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# -------------------------------------------------------------
# Chart 1: Execution Time (simSeconds) - Log Scale
# -------------------------------------------------------------
ax1.barh(y - height/2, scalar_seconds, height, label='Scalar', color='#4A90E2', alpha=0.9)
ax1.barh(y + height/2, rvv_seconds, height, label='RVV (Vector)', color='#50E3C2', alpha=0.9)

ax1.set_xscale('log')
ax1.set_xlabel('Simulated Time (Seconds, Log Scale)', fontsize=11, fontweight='bold')
ax1.set_title('Execution Time: Scalar vs. RVV', fontsize=12, fontweight='bold', pad=10)
ax1.set_yticks(y)
ax1.set_yticklabels(kernels, fontsize=10)
ax1.invert_yaxis()
ax1.legend(frameon=True, facecolor='white', edgecolor='none')
ax1.grid(True, which="both", ls="--", alpha=0.5)

# -------------------------------------------------------------
# Chart 2: Dynamic Instruction Count - Log Scale
# -------------------------------------------------------------
ax2.barh(y - height/2, scalar_insts, height, label='Scalar', color='#4A90E2', alpha=0.9)
ax2.barh(y + height/2, rvv_insts, height, label='RVV (Vector)', color='#F5A623', alpha=0.9)

ax2.set_xscale('log')
ax2.set_xlabel('Dynamic Instruction Count (Log Scale)', fontsize=11, fontweight='bold')
ax2.set_title('Instruction Count: Scalar vs. RVV', fontsize=12, fontweight='bold', pad=10)
ax2.legend(frameon=True, facecolor='white', edgecolor='none')
ax2.grid(True, which="both", ls="--", alpha=0.5)

# Visual category dividers
for ax in (ax1, ax2):
    ax.axhline(3.5, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axhline(7.5, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)

plt.tight_layout()
plt.savefig('rvv_vs_scalar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()