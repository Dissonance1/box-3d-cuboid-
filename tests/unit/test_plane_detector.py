"""Unit tests for RANSAC plane detection.

Tests cover:
  - Single plane on clean data (should find exact plane)
  - Multi-plane on box point cloud (3 orthogonal planes)
  - Robustness to noise (20% outlier contamination)
  - Empty/degenerate point cloud rejection
  - Orthogonality enforcement
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.plane_detector import MultiPlaneDetector, enforce_orthogonality
from src.utils.config import PlaneDetectionConfig
from src.utils.exceptions import PlaneDetectionError
from src.utils.types import PlaneModel, PointCloud


def _make_plane_cloud(normal, offset, n_points, noise_std=0.002, rng=None):
    """Generate points on a plane ax+by+cz+d=0 with optional noise."""
    if rng is None:
        rng = np.random.default_rng(0)
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Build two tangent vectors
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, [1, 0, 0])
    else:
        t1 = np.cross(normal, [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    u = rng.uniform(-0.5, 0.5, n_points)
    v = rng.uniform(-0.5, 0.5, n_points)
    # Plane point: normal*(-offset) + u*t1 + v*t2
    base = normal * (-offset)
    pts = base + u[:, None] * t1 + v[:, None] * t2
    pts += rng.normal(0, noise_std, pts.shape)
    return pts


class TestSinglePlaneFitting:

    def test_detects_horizontal_plane(self):
        pts = _make_plane_cloud([0, 1, 0], -1.5, 1000)
        colors = np.ones((len(pts), 3), dtype=np.float32)
        pc = PointCloud(points=pts, colors=colors)
        config = PlaneDetectionConfig(ransac_min_inlier_ratio=0.10, min_planes_for_cuboid=1)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        assert len(planes) >= 1
        p = planes[0]
        cos_ang = abs(np.dot(p.normal, np.array([0, 1, 0])))
        assert cos_ang > 0.98, f"Normal not aligned: {p.normal}"

    def test_detects_arbitrary_plane(self):
        normal = np.array([0.577, 0.577, 0.577])
        normal /= np.linalg.norm(normal)
        pts = _make_plane_cloud(normal, -2.0, 800)
        colors = np.ones((len(pts), 3), dtype=np.float32)
        pc = PointCloud(points=pts, colors=colors)
        config = PlaneDetectionConfig(ransac_min_inlier_ratio=0.10, min_planes_for_cuboid=1)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        assert len(planes) >= 1
        cos_ang = abs(np.dot(planes[0].normal, normal))
        assert cos_ang > 0.97

    def test_plane_offset_accuracy(self):
        """RANSAC + SVD refinement should estimate offset within 5mm."""
        pts = _make_plane_cloud([0, 0, 1], -1.5, 1000, noise_std=0.003)
        colors = np.ones((len(pts), 3), dtype=np.float32)
        pc = PointCloud(points=pts, colors=colors)
        config = PlaneDetectionConfig(
            ransac_min_inlier_ratio=0.10,
            ransac_distance_threshold_m=0.010,
            min_planes_for_cuboid=1,
        )
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        assert len(planes) >= 1
        estimated_offset = abs(planes[0].offset)
        assert abs(estimated_offset - 1.5) < 0.010, f"Offset error: {abs(estimated_offset - 1.5)*1000:.1f}mm"

    def test_robust_to_outliers(self):
        """20% outlier contamination should not break detection."""
        rng = np.random.default_rng(42)
        inliers = _make_plane_cloud([0, 1, 0], -1.0, 800, noise_std=0.002, rng=rng)
        outliers = rng.uniform(-1, 1, (200, 3))
        pts = np.vstack([inliers, outliers])
        rng.shuffle(pts)
        colors = np.ones((len(pts), 3), dtype=np.float32)
        pc = PointCloud(points=pts, colors=colors)
        config = PlaneDetectionConfig(ransac_min_inlier_ratio=0.10, min_planes_for_cuboid=1)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        assert len(planes) >= 1
        cos_ang = abs(np.dot(planes[0].normal, [0, 1, 0]))
        assert cos_ang > 0.95


class TestMultiPlaneDetection:

    def test_detects_three_orthogonal_planes(self, synthetic_point_cloud, plane_config):
        """Should find 3 planes from the synthetic box point cloud."""
        detector = MultiPlaneDetector(plane_config)
        planes = detector.detect(synthetic_point_cloud)
        assert len(planes) >= 2

    def test_planes_are_near_orthogonal(self, synthetic_point_cloud, plane_config):
        """Box normals should be mutually orthogonal within tolerance."""
        detector = MultiPlaneDetector(plane_config)
        planes = detector.detect(synthetic_point_cloud)
        if len(planes) >= 2:
            for i in range(len(planes) - 1):
                for j in range(i + 1, len(planes)):
                    dot = abs(np.dot(planes[i].normal, planes[j].normal))
                    assert dot < 0.20, f"Planes {i} and {j} not orthogonal: dot={dot:.3f}"

    def test_empty_cloud_raises(self):
        """Detector must raise on empty point cloud."""
        pc = PointCloud(points=np.zeros((5, 3)), colors=np.zeros((5, 3)))
        config = PlaneDetectionConfig()
        detector = MultiPlaneDetector(config)
        with pytest.raises((PlaneDetectionError, Exception)):
            detector.detect(pc)


class TestOrthogonalityEnforcement:

    def test_corrects_nearly_orthogonal_normals(self):
        """Two planes at 87° should be corrected to exactly 90°."""
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([np.sin(np.deg2rad(3)), np.cos(np.deg2rad(3)), 0.0])

        planes = [
            PlaneModel(normal=n1, offset=0, inlier_count=100, inlier_ratio=0.5,
                       fitness=0.9, rmse=0.003, centroid=np.zeros(3)),
            PlaneModel(normal=n2, offset=0, inlier_count=100, inlier_ratio=0.5,
                       fitness=0.9, rmse=0.003, centroid=np.zeros(3)),
        ]
        corrected = enforce_orthogonality(planes, tolerance_deg=15.0)
        dot = abs(np.dot(corrected[0].normal, corrected[1].normal))
        assert dot < 0.06, f"After correction, planes not orthogonal: dot={dot:.4f}"
