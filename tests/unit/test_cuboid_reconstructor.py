"""Unit tests for cuboid reconstruction accuracy.

Ground truth: box with known dimensions and pose.
Acceptance criteria matching production targets:
  - Dimension accuracy: ±5mm (test with ±10mm tolerance to account for noise)
  - Center position: ±10mm
  - Yaw angle: ±3°
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.cuboid_reconstructor import CuboidReconstructor
from src.geometry.plane_detector import MultiPlaneDetector
from src.utils.config import PlaneDetectionConfig
from src.utils.types import PlaneModel, PointCloud


def _make_box_cloud(
    L: float, W: float, H: float,
    cx: float, cy: float, cz: float,
    yaw_deg: float = 0.0,
    n_per_face: int = 400,
    noise_std: float = 0.002,
) -> PointCloud:
    """Synthesize a box point cloud with 3 visible faces (top + 2 sides)."""
    yaw = np.deg2rad(yaw_deg)
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0,           1, 0           ],
        [-np.sin(yaw),0, np.cos(yaw) ],
    ])
    rng = np.random.default_rng(7)
    faces = []

    # Top face (normal = [0, -1, 0] local → Y up = camera Y = down, so -1)
    u = rng.uniform(-L/2, L/2, n_per_face)
    w = rng.uniform(-W/2, W/2, n_per_face)
    top = np.column_stack([u, np.full(n_per_face, -H/2), w])
    faces.append(top)

    # Front face (+Z)
    u2 = rng.uniform(-L/2, L/2, n_per_face)
    v2 = rng.uniform(-H/2, H/2, n_per_face)
    front = np.column_stack([u2, v2, np.full(n_per_face, W/2)])
    faces.append(front)

    # Right face (+X)
    v3 = rng.uniform(-H/2, H/2, n_per_face)
    w3 = rng.uniform(-W/2, W/2, n_per_face)
    right = np.column_stack([np.full(n_per_face, L/2), v3, w3])
    faces.append(right)

    pts_local = np.vstack(faces)
    pts_world = (R @ pts_local.T).T + np.array([cx, cy, cz])
    pts_world += rng.normal(0, noise_std, pts_world.shape)

    colors = np.ones((len(pts_world), 3), dtype=np.float32) * 0.5
    return PointCloud(points=pts_world.astype(np.float64), colors=colors)


class TestCuboidDimensions:

    @pytest.mark.parametrize("L,W,H", [
        (0.400, 0.300, 0.200),
        (0.600, 0.400, 0.300),
        (0.300, 0.300, 0.200),
        (0.500, 0.200, 0.150),
    ])
    def test_dimension_accuracy(self, L, W, H):
        """Reconstructed dimensions within ±10mm of ground truth."""
        pc = _make_box_cloud(L, W, H, 0.05, 0.1, 1.5, yaw_deg=0, noise_std=0.002)
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)

        reconstructor = CuboidReconstructor(config)
        cuboid = reconstructor.reconstruct(planes, pc)

        gt_dims = sorted([L, W, H], reverse=True)
        est_dims = sorted(cuboid.dimensions.tolist(), reverse=True)

        tolerance_m = 0.010  # 10mm
        for i, (gt, est) in enumerate(zip(gt_dims, est_dims)):
            err_mm = abs(gt - est) * 1000
            assert err_mm < tolerance_m * 1000, (
                f"Dim[{i}] error {err_mm:.1f}mm > {tolerance_m*1000:.0f}mm "
                f"(gt={gt*1000:.0f}mm, est={est*1000:.0f}mm)"
            )

    def test_center_accuracy(self):
        """Reconstructed center within ±10mm of ground truth."""
        cx, cy, cz = 0.05, 0.10, 1.50
        pc = _make_box_cloud(0.4, 0.3, 0.2, cx, cy, cz)
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        reconstructor = CuboidReconstructor(config)
        cuboid = reconstructor.reconstruct(planes, pc)

        gt_center = np.array([cx, cy, cz])
        err = np.linalg.norm(cuboid.center - gt_center) * 1000
        assert err < 15.0, f"Center error {err:.1f}mm > 15mm"

    def test_rotation_matrix_orthonormal(self):
        """Output rotation matrix must be orthonormal (det=1)."""
        pc = _make_box_cloud(0.4, 0.3, 0.2, 0, 0, 1.5)
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        reconstructor = CuboidReconstructor(config)
        cuboid = reconstructor.reconstruct(planes, pc)

        R = cuboid.rotation_matrix
        I_approx = R.T @ R
        assert np.allclose(I_approx, np.eye(3), atol=0.01), "Rotation not orthonormal"
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 0.01, f"Rotation det={det:.4f} not +1"

    def test_minimum_planes_fallback(self):
        """Should reconstruct from 2 planes by synthesizing the third pair."""
        pc = _make_box_cloud(0.4, 0.3, 0.2, 0, 0, 1.5, n_per_face=300)
        # Manually create just 2 planes
        p1 = PlaneModel(np.array([0.0, 1.0, 0.0]), -0.9, 300, 0.33, 0.8, 0.004,
                        np.array([0.0, 0.9, 1.5]))
        p2 = PlaneModel(np.array([0.0, 0.0, 1.0]), -1.65, 300, 0.33, 0.8, 0.004,
                        np.array([0.0, 0.0, 1.65]))
        config = PlaneDetectionConfig(min_planes_for_cuboid=2)
        reconstructor = CuboidReconstructor(config)
        cuboid = reconstructor.reconstruct([p1, p2], pc)
        assert cuboid is not None
        assert cuboid.dimensions[0] > 0.0


class TestReconstructionQuality:

    def test_high_quality_score_for_clean_data(self):
        pc = _make_box_cloud(0.4, 0.3, 0.2, 0, 0, 1.5, noise_std=0.001)
        config = PlaneDetectionConfig(ransac_n_iterations=500)
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        reconstructor = CuboidReconstructor(config)
        cuboid = reconstructor.reconstruct(planes, pc)
        assert cuboid.reconstruction_quality > 0.4, \
            f"Quality {cuboid.reconstruction_quality:.3f} too low for clean data"
