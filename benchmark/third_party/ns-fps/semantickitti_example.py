import numpy as np
import os
import yuezu_fps.yuezu_fps_module as yf
import time

def load_kitti_point_cloud(bin_path: str) -> np.ndarray:
    """
    📂 Load SemanticKITTI point cloud from .bin file
    Format: [x, y, z, intensity] as float32
    """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # Return xyz coordinates only (discard intensity)
    return point_cloud[:, :3]

def sample_kitti_point_cloud(points: np.ndarray, n_samples: int = 1024) -> np.ndarray:
    """
    ⚡ Run NS-FPS sampling on KITTI point cloud
    Automatically calculates space range from point cloud bounds
    """
    # Calculate auto range from point cloud
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()

    # Create space range with appropriate granularity
    space_range = yf.make_range(
        float(min_x), float(max_x),
        float(min_y), float(max_y),
        float(min_z), float(max_z),
        128, 128, 2  # KITTI has smaller Z range
    )

    # Run FPS sampling
    start_time = time.time()
    indices = yf.fps(points, n_samples, space_range)
    end_time = time.time()
    print(f"⏱️  FPS sampling time: {end_time - start_time:.4f} seconds")
    return indices

if __name__ == "__main__":
    DATA_DIR = "simple_semantickitti_data"

    # 📁 List all bin files
    bin_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".bin")])
    print(f"🔍 Found {len(bin_files)} SemanticKITTI point cloud files")

    # Process first file as example
    if bin_files:
        bin_path = os.path.join(DATA_DIR, bin_files[0])
        points = load_kitti_point_cloud(bin_path)
        print(f"\n📥 Loaded point cloud: {points.shape[0]} points")
        print(f"📏 Point cloud bounds: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] "
              f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] "
              f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

        # Run sampling
        indices = sample_kitti_point_cloud(points, n_samples=1024)
        sampled_points = points[indices]
        print(f"\n🎯 Sampled {sampled_points.shape[0]} points")
        print(f"🗺️  Sampled bounds: X[{sampled_points[:,0].min():.2f}, {sampled_points[:,0].max():.2f}] "
              f"Y[{sampled_points[:,1].min():.2f}, {sampled_points[:,1].max():.2f}] "
              f"Z[{sampled_points[:,2].min():.2f}, {sampled_points[:,2].max():.2f}]")
        print("\n✅ Sampling completed successfully!")
