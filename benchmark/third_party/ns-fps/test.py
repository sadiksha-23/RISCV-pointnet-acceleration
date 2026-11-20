import numpy as np
import time
import yuezu_fps.yuezu_fps_module as yf


# =============================================================================
# SpaceRange Calculation on Python Side
# =============================================================================

def compute_bbox_python(points, x_blocks=16, y_blocks=16, z_blocks=16, margin=0.01):
    """
    Calculate SpaceRange on the Python side to ensure all points are included
    """
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    
    # Add margin to prevent boundary issues
    x_pad = (max_x - min_x) * margin + 1e-4
    y_pad = (max_y - min_y) * margin + 1e-4
    z_pad = (max_z - min_z) * margin + 1e-4
    
    return yf.make_range(
        float(min_x - x_pad), float(max_x + x_pad),
        float(min_y - y_pad), float(max_y + y_pad),
        float(min_z - z_pad), float(max_z + z_pad),
        x_blocks, y_blocks, z_blocks
    )


# =============================================================================
# Python Reference Implementation (Brute-force FPS, No Spatial Constraints)
# =============================================================================

def fps_numpy(points: np.ndarray, n_samples: int):
    """Traditional FPS, iterating over all points"""
    N = len(points)
    if n_samples >= N:
        return np.arange(N), np.zeros(N)
    
    indices = np.zeros(n_samples, dtype=np.int64)
    indices[0] = 0
    
    min_distances = np.full(N, np.inf, dtype=np.float32)
    points_f = points.astype(np.float32)
    max_dists = np.zeros(n_samples, dtype=np.float32)
    max_dists[0] = np.inf
    
    for i in range(1, n_samples):
        last_point = points_f[indices[i - 1]]
        dists = np.sum((points_f - last_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dists)
        
        max_dists[i] = np.max(min_distances)
        indices[i] = np.argmax(min_distances)
        
        if max_dists[i] == 0:
            indices = indices[:i]
            max_dists = max_dists[:i]
            break
    
    return indices, max_dists


def compute_distance_sequence(points: np.ndarray, indices: np.ndarray):
    """Compute maximum distance per step from index sequence"""
    N = len(points)
    n = len(indices)
    
    min_distances = np.full(N, np.inf, dtype=np.float32)
    points_f = points.astype(np.float32)
    max_dists = np.zeros(n, dtype=np.float32)
    max_dists[0] = np.inf
    
    for i in range(1, n):
        last_point = points_f[indices[i - 1]]
        dists = np.sum((points_f - last_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dists)
        max_dists[i] = np.max(min_distances)
    
    return max_dists


# =============================================================================
# Core Verification Functions
# =============================================================================

def verify_correctness():
    """Verify consistency between C++ implementation and Python reference"""
    print("=" * 70)
    print("FPS Correctness Verification (SpaceRange Calculated by Python)")
    print("=" * 70)
    
    np.random.seed(42)
    
    test_cases = [
        ("Small scale: 100 points/10 samples", np.random.randn(100, 3).astype(np.float32), 10),
        ("Medium scale: 1k points/100 samples", np.random.randn(1000, 3).astype(np.float32), 100),
        ("Large scale: 10k points/1k samples", np.random.randn(10000, 3).astype(np.float32), 1000),
        ("Non-uniform distribution", np.random.randn(5000, 3).astype(np.float32) * [10, 1, 0.1], 500),
    ]
    
    all_passed = True
    
    for name, points, n_samples in test_cases:
        print(f"\n【{name}】")
        
        # 1. Compute SpaceRange on Python side
        range_obj = compute_bbox_python(points, 16, 16, 16)
        
        # 2. Python brute-force implementation
        t0 = time.time()
        py_idx, py_dists = fps_numpy(points, n_samples)
        py_time = (time.time() - t0) * 1000
        
        # 3. C++ implementation (passing range computed by Python)
        t0 = time.time()
        cpp_idx = yf.fps(points, n_samples, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        # 4. Compute C++ distance sequence
        cpp_dists = compute_distance_sequence(points, cpp_idx)
        
        # 5. Compare
        min_len = min(len(py_dists), len(cpp_dists))
        max_diff = 0.0
        first_diff_idx = -1
        
        for i in range(min_len):
            diff = abs(py_dists[i] - cpp_dists[i])
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-3 and first_diff_idx == -1:
                first_diff_idx = i
        
        length_ok = abs(len(py_dists) - len(cpp_dists)) <= 1
        dist_ok = max_diff < 1e-3
        passed = length_ok and dist_ok
        
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"  Python: {py_time:7.2f}ms | C++: {cpp_time:6.2f}ms | Speedup {py_time / cpp_time:5.1f}x")
        print(f"  Sample count: {len(py_idx)}(Py) vs {len(cpp_idx)}(C++) | Max distance diff: {max_diff:.2e} [{status}]")
        
        if not passed:
            all_passed = False
            if first_diff_idx >= 0:
                print(f"  First discrepancy @ step{first_diff_idx}: Py={py_dists[first_diff_idx]:.6f}, C++={cpp_dists[first_diff_idx]:.6f}")
    
    print("\n" + "=" * 70)
    print(f"Verification result: {'All passed ✓' if all_passed else 'Some failed ✗'}")
    print("=" * 70)
    
    return all_passed


def verify_granularity():
    """Verify different spatial granularity configurations"""
    print("\n" + "=" * 70)
    print("Spatial Granularity Configuration Test (SpaceRange Calculated by Python)")
    print("=" * 70)
    
    np.random.seed(42)
    points = np.random.randn(5000, 3).astype(np.float32) * 100
    n_samples = 500
    
    # Python reference
    py_idx, py_dists = fps_numpy(points, n_samples)
    print(f"Python reference: {len(py_dists)} samples")
    
    # Granularity configurations
    configs = [
        (8, 8, 8, "Coarse 8x8x8"),
        (16, 16, 16, "Standard 16x16x16"),
        (32, 32, 32, "Fine 32x32x32"),
        (64, 64, 64, "Ultra-fine 64x64x64"),
        (16, 8, 4, "Non-uniform 16x8x4"),
        (4, 16, 64, "Non-uniform 4x16x64"),
        (32, 16, 8, "Non-uniform 32x16x8"),
    ]
    
    print(f"\n{'Config':<20} | {'Total Blocks':>12} | {'Py Samples':>10} | {'C++ Samples':>12} | {'Max Dist Diff':>14} | {'Time(ms)':>10} | {'Result':>8}")
    print("-" * 100)
    
    all_passed = True
    
    for xb, yb, zb, desc in configs:
        # Compute SpaceRange in Python
        range_obj = compute_bbox_python(points, xb, yb, zb)
        
        t0 = time.time()
        cpp_idx = yf.fps(points, n_samples, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        cpp_dists = compute_distance_sequence(points, cpp_idx)
        
        min_len = min(len(py_dists), len(cpp_dists))
        max_diff = 0.0
        for i in range(min_len):
            diff = abs(py_dists[i] - cpp_dists[i])
            if diff > max_diff:
                max_diff = diff
        
        total_blocks = xb * yb * zb
        passed = max_diff < 1e-3 and abs(len(py_dists) - len(cpp_dists)) <= 1
        status = "✓" if passed else "✗"
        
        print(f"{desc:<20} | {total_blocks:>12} | {len(py_idx):>10} | {len(cpp_idx):>12} | "
              f"{max_diff:>14.2e} | {cpp_time:>10.2f} | {status:>8}")
        
        if not passed:
            all_passed = False
    
    print("-" * 100)
    print(f"Granularity test: {'All passed ✓' if all_passed else 'Some failed ✗'}")
    
    return all_passed


def verify_manual_range():
    """Demonstrate risks of manual range specification (insufficient range leads to insufficient sampling)"""
    print("\n" + "=" * 70)
    print("Manual Range Test (Demonstrate Consequences of Insufficient Range)")
    print("=" * 70)
    
    np.random.seed(123)
    points = np.random.randn(3000, 3).astype(np.float32)
    points[:, 0] = points[:, 0] * 50 + 100   # X: [50, 150]
    points[:, 1] = points[:, 1] * 20 - 30    # Y: [-50, -10]
    points[:, 2] = points[:, 2] * 5 + 10     # Z: [5, 15]
    
    n_samples = 300
    
    print(f"Actual data range: X[{points[:, 0].min():.1f},{points[:, 0].max():.1f}], "
          f"Y[{points[:, 1].min():.1f},{points[:, 1].max():.1f}], "
          f"Z[{points[:, 2].min():.1f},{points[:, 2].max():.1f}]")
    
    # Comparison: correct range vs incorrect range
    test_cases = [
        ("Correct range (computed by Python)", compute_bbox_python(points, 16, 16, 8)),
        ("Manual - includes all", yf.make_range(0, 200, -100, 0, 0, 20, 16, 16, 8)),
        ("Manual - Y insufficient", yf.make_range(0, 200, -40, -20, 0, 20, 16, 16, 8)),  # Y too tight
        ("Manual - Z insufficient", yf.make_range(0, 200, -100, 0, 10, 15, 16, 16, 8)),  # Z too tight
    ]
    
    print(f"\n{'Range Config':<20} | {'X Range':>16} | {'Y Range':>16} | {'Z Range':>12} | {'C++ Samples':>12} | {'Status':>10}")
    print("-" * 105)
    
    for label, range_obj in test_cases:
        cpp_idx = yf.fps(points, n_samples, range_obj)
        
        # Count points within range
        in_range = (
            (points[:, 0] >= range_obj.min_x) & (points[:, 0] <= range_obj.max_x) &
            (points[:, 1] >= range_obj.min_y) & (points[:, 1] <= range_obj.max_y) &
            (points[:, 2] >= range_obj.min_z) & (points[:, 2] <= range_obj.max_z)
        )
        n_in_range = np.sum(in_range)
        
        if len(cpp_idx) == n_samples:
            status = "✓ OK"
        elif n_in_range < n_samples:
            status = f"✗ Only {n_in_range} pts"
        else:
            status = f"⚠️ Sampled {len(cpp_idx)}"
        
        x_range = f"[{range_obj.min_x:.0f},{range_obj.max_x:.0f}]"
        y_range = f"[{range_obj.min_y:.0f},{range_obj.max_y:.0f}]"
        z_range = f"[{range_obj.min_z:.0f},{range_obj.max_z:.0f}]"
        
        print(f"{label:<20} | {x_range:>16} | {y_range:>16} | {z_range:>12} | {len(cpp_idx):>12} | {status:>10}")
    
    print("\nConclusion: Recommend using compute_bbox_python() to automatically compute the range to avoid missing data due to manual specification")


def verify_step_by_step(points=None, n_samples=20):
    """Detailed step-by-step comparison"""
    print("\n" + "=" * 70)
    print("Detailed Step-by-Step Comparison")
    print("=" * 70)
    
    if points is None:
        np.random.seed(123)
        points = np.random.randn(100, 3).astype(np.float32)
    
    # Compute SpaceRange in Python
    range_obj = compute_bbox_python(points, 16, 16, 16)
    
    py_idx, py_dists = fps_numpy(points, n_samples)
    cpp_idx = yf.fps(points, n_samples, range_obj)
    cpp_dists = compute_distance_sequence(points, cpp_idx)
    
    n = min(len(py_idx), len(cpp_idx), n_samples)
    
    print(f"{'Step':>5} | {'PyIdx':>6} | {'CppIdx':>6} | {'IdxMatch':>8} | {'PyDist':>10} | {'CppDist':>10} | {'DistDiff':>10}")
    print("-" * 85)
    
    for i in range(n):
        idx_match = "✓" if py_idx[i] == cpp_idx[i] else "✗"
        dist_diff = abs(py_dists[i] - cpp_dists[i]) if i < len(cpp_dists) else float('nan')
        dist_diff_str = f"{dist_diff:.2e}" if dist_diff < 1e6 else "N/A"
        marker = " <--" if dist_diff > 1e-3 else ""
        
        print(f"{i:>5} | {py_idx[i]:>6} | {cpp_idx[i]:>6} | {idx_match:>8} | "
              f"{py_dists[i]:>10.4f} | {cpp_dists[i]:>10.4f} | {dist_diff_str:>10}{marker}")


def benchmark_speed():
    """Performance benchmark"""
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    sizes = [1000, 5000, 10000, 50000]
    sample_ratio = 0.25
    
    print(f"{'Size':>8} | {'Samples':>8} | {'Python(ms)':>12} | {'C++(ms)':>10} | {'Speedup':>8}")
    print("-" * 65)
    
    for n in sizes:
        np.random.seed(42)
        points = np.random.randn(n, 3).astype(np.float32)
        k = int(n * sample_ratio)
        
        # Python
        t0 = time.time()
        fps_numpy(points, k)
        py_time = (time.time() - t0) * 1000
        
        # C++ (range computed by Python)
        range_obj = compute_bbox_python(points, 16, 16, 16)
        t0 = time.time()
        yf.fps(points, k, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
        
        print(f"{n:>8} | {k:>8} | {py_time:>12.2f} | {cpp_time:>10.2f} | {speedup:>8.1f}x")


def visualize_distance_decay(save_path="figs/fps_decay.png"):
    """Visualize distance decay curve"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping visualization")
        return
    
    print("\nGenerating distance decay comparison plot...")
    
    np.random.seed(42)
    points = np.random.randn(5000, 3).astype(np.float32)
    n_samples = 500
    
    # Compute range in Python
    range_obj = compute_bbox_python(points, 16, 16, 16)
    
    py_idx, py_dists = fps_numpy(points, n_samples)
    cpp_idx = yf.fps(points, n_samples, range_obj)
    cpp_dists = compute_distance_sequence(points, cpp_idx)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left plot: distance decay curve
    ax1 = axes[0]
    steps = range(1, min(len(py_dists), len(cpp_dists)))
    ax1.semilogy(steps, py_dists[1:len(steps) + 1], 'b-', label='Python', alpha=0.7, linewidth=2)
    ax1.semilogy(steps, cpp_dists[1:len(steps) + 1], 'r--', label='C++', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Sample Step', fontsize=11)
    ax1.set_ylabel('Max Distance (log scale)', fontsize=11)
    ax1.legend()
    ax1.set_title('Distance Decay Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: distance difference
    ax2 = axes[1]
    min_len = min(len(py_dists), len(cpp_dists))
    diffs = np.abs(py_dists[:min_len] - cpp_dists[:min_len])
    ax2.semilogy(range(min_len), diffs, 'g-', alpha=0.7)
    ax2.axhline(y=1e-3, color='r', linestyle='--', label='threshold 1e-3')
    ax2.set_xlabel('Sample Step', fontsize=11)
    ax2.set_ylabel('|Distance Difference|', fontsize=11)
    ax2.set_title(f'Max Diff: {np.max(diffs):.2e}', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {save_path}")


# =============================================================================
# Main Entry
# =============================================================================

if __name__ == "__main__":
    results = []
    
    # Core verification
    results.append(("Basic correctness", verify_correctness()))
    results.append(("Granularity config", verify_granularity()))
    
    # Demonstrate risks of manual range
    verify_manual_range()
    
    # Other tests
    verify_step_by_step()
    benchmark_speed()
    visualize_distance_decay()
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("-" * 70)
    if all_passed:
        print("✓ All core verifications passed!")
        print("=" * 70)
        print("Usage:")
        print("  1. Compute SpaceRange in Python:")
        print("     range_obj = compute_bbox_python(points, x_blocks, y_blocks, z_blocks)")
        print("  2. Or manually specify:")
        print("     range_obj = yf.make_range(min_x, max_x, ..., x_blocks, y_blocks, z_blocks)")
        print("  3. Call FPS:")
        print("     indices = yf.fps(points, n_samples, range_obj)")
    else:
        print("✗ Some verifications failed")
    print("=" * 70)