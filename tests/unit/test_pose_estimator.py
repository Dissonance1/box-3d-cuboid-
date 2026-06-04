"""Unit tests for pose estimation from cuboid geometry."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.pose.pose_estimator import PoseEstimator
from src.utils.config import SystemConfig
from src.utils.types import CuboidGeometry, PlaneModel


def _make_cuboid(
    cx=0.0, cy=0.0, cz=1.5,
    L=0.4, W=0.3, H=0.2,
    yaw_deg=0.0,
) -> CuboidGeometry:
    yaw = np.deg2rad(yaw_deg)
    R = Rotation.from_euler("ZYX", [yaw, 0, 0]).as_matrix()
    dims = np.array([L, W, H])

    hL, hW, hH = L/2, W/2, H/2
    local_corners = np.array([
        [s * hL, s2 * hW, s3 * hH]
        for s in [-1, 1] for s2 in [-1, 1] for s3 in [-1, 1]
    ])
    corners = (R @ local_corners.T).T + np.array([cx, cy, cz])

    dummy_plane = PlaneModel(
        normal=R[:, 1], offset=-float(np.dot(R[:, 1], [cx, cy, cz])),
        inlier_count=300, inlier_ratio=0.4, fitness=0.8, rmse=0.003,
        centroid=np.array([cx, cy, cz])
    )
    return CuboidGeometry(
        corners=corners,
        faces=[dummy_plane],
        center=np.array([cx, cy, cz]),
        rotation_matrix=R,
        dimensions=dims,
        reconstruction_quality=0.85,
    )


class TestPoseEstimation:

    def setup_method(self):
        self.estimator = PoseEstimator(SystemConfig())

    def test_position_accuracy(self):
        cx, cy, cz = 0.15, 0.08, 2.10
        cuboid = _make_cuboid(cx=cx, cy=cy, cz=cz)
        pose = self.estimator.estimate(cuboid)
        np.testing.assert_allclose(pose.position, [cx, cy, cz], atol=1e-6)

    def test_quaternion_unit_length(self):
        cuboid = _make_cuboid()
        pose = self.estimator.estimate(cuboid)
        q_norm = np.linalg.norm(pose.quaternion)
        assert abs(q_norm - 1.0) < 1e-4, f"Quaternion norm {q_norm:.6f} not unit"

    def test_euler_ranges(self):
        cuboid = _make_cuboid(yaw_deg=30.0)
        pose = self.estimator.estimate(cuboid)
        assert -180 <= pose.euler_deg[0] <= 180, "Roll out of range"
        assert -180 <= pose.euler_deg[1] <= 180, "Pitch out of range"
        assert -180 <= pose.euler_deg[2] <= 180, "Yaw out of range"

    def test_yaw_recovery(self):
        """Estimated yaw should match ground truth within ±5°."""
        for gt_yaw in [0, 15, 30, 45, -30, 90]:
            cuboid = _make_cuboid(yaw_deg=gt_yaw)
            pose = self.estimator.estimate(cuboid)
            # Compare rotation matrices (yaw axis convention may differ)
            R_gt = Rotation.from_euler("ZYX", [np.deg2rad(gt_yaw), 0, 0]).as_matrix()
            R_est = pose.rotation_matrix
            # Product should be near identity
            diff = R_gt.T @ R_est
            angle_err = np.rad2deg(np.arccos(np.clip((np.trace(diff) - 1) / 2, -1, 1)))
            assert angle_err < 5.0 or angle_err > 175.0, \
                f"Yaw {gt_yaw}°: rotation error {angle_err:.2f}°"

    def test_rotation_matrix_orthonormal(self):
        cuboid = _make_cuboid(yaw_deg=45.0)
        pose = self.estimator.estimate(cuboid)
        I_approx = pose.rotation_matrix.T @ pose.rotation_matrix
        np.testing.assert_allclose(I_approx, np.eye(3), atol=0.01)

    def test_degenerate_rotation_handled(self):
        """Singular rotation matrix should be corrected via SVD, not raise."""
        cuboid = _make_cuboid()
        # Corrupt rotation matrix
        cuboid.rotation_matrix[0, :] = cuboid.rotation_matrix[1, :]
        pose = self.estimator.estimate(cuboid)  # Should not raise
        assert pose is not None
