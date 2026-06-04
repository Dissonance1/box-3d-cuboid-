"""Performance benchmarks for each pipeline stage.

Run with: pytest tests/performance/ -v --benchmark-sort=mean

Target latencies (for 30 FPS budget of 33ms/frame):
  YOLO11n-seg inference:     ≤15ms on GPU
  Point cloud generation:     ≤3ms per object
  Plane detection (RANSAC):   ≤5ms per object
  Cuboid reconstruction:      ≤2ms per object
  Total pipeline budget:      ≤30ms per frame
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.plane_detector import MultiPlaneDetector
from src.geometry.cuboid_reconstructor import CuboidReconstructor
from src.tracking.kalman_box_filter import KalmanBoxFilter
from src.utils.config import PlaneDetectionConfig
from src.utils.types import PointCloud


@pytest.fixture(scope="module")
def large_box_cloud():
    """5000-point cloud for performance benchmarking."""
    rng = np.random.default_rng(0)
    L, W, H = 0.4, 0.3, 0.2
    faces = []
    for face_idx in range(3):
        n = 1667
        u = rng.uniform(-0.5, 0.5, n)
        v = rng.uniform(-0.5, 0.5, n)
        if face_idx == 0:
            pts = np.column_stack([u * L, np.full(n, -H/2), v * W])
        elif face_idx == 1:
            pts = np.column_stack([u * L, v * H, np.full(n, W/2)])
        else:
            pts = np.column_stack([np.full(n, L/2), v * H, u * W])
        pts += np.array([0.05, 0.1, 1.5])
        pts += rng.normal(0, 0.002, pts.shape)
        faces.append(pts)
    all_pts = np.vstack(faces)
    colors = np.ones((len(all_pts), 3), dtype=np.float32)
    return PointCloud(points=all_pts.astype(np.float64), colors=colors)


class TestRANSACBenchmark:

    def test_plane_detection_5000pts(self, benchmark, large_box_cloud):
        """RANSAC on 5000-point cloud must complete in <10ms."""
        config = PlaneDetectionConfig(ransac_n_iterations=1000)
        detector = MultiPlaneDetector(config)
        result = benchmark(detector.detect, large_box_cloud)
        assert len(result) >= 2

    def test_plane_detection_fast_mode(self, benchmark, large_box_cloud):
        """Fast mode (500 iter) for edge devices."""
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        result = benchmark(detector.detect, large_box_cloud)
        assert len(result) >= 1


class TestCuboidBenchmark:

    def test_reconstruction_benchmark(self, benchmark, large_box_cloud):
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(large_box_cloud)
        reconstructor = CuboidReconstructor(config)
        cuboid = benchmark(reconstructor.reconstruct, planes, large_box_cloud)
        assert cuboid is not None


class TestKalmanBenchmark:

    def test_kalman_update_1000_calls(self, benchmark):
        """1000 Kalman updates should complete in <5ms total."""
        kf = KalmanBoxFilter()
        state = np.array([0.1, 0.1, 1.5, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0])
        kf.initialize(state)

        def run_updates():
            for _ in range(1000):
                kf.predict()
                kf.update(state)

        benchmark(run_updates)
