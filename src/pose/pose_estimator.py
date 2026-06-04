"""6DoF pose estimation from cuboid geometry.

Coordinate conventions:
  Camera frame:  X right, Y down, Z forward (standard OpenCV/RealSense)
  World frame:   X east, Y north, Z up (standard robotics/ENU)

Rotation representation:
  - Rotation matrix R (3x3, det=+1, R^T = R^-1)
  - Quaternion [qw, qx, qy, qz] (scalar-first, unit quaternion)
  - Euler angles [roll, pitch, yaw] in degrees, ZYX convention (yaw applied first)

Yaw convention: 0° = length axis aligned with camera Z (pointing forward).
Positive yaw rotates counter-clockwise when viewed from above.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.config import SystemConfig
from src.utils.exceptions import PoseEstimationError
from src.utils.logger import get_logger
from src.utils.types import CuboidGeometry, PoseEstimate

logger = get_logger(__name__)

# Camera-to-world rotation: rotate from camera (X right, Y down, Z forward)
#   to world (X east, Y north, Z up)
# Standard transformation:
#   Xw = Xc          (east = right)
#   Yw = -Zc         (north = -forward... depends on setup; configurable)
#   Zw = -Yc         (up = -down)
# This is set from config; the default below is identity (camera = world).
_DEFAULT_C2W = np.eye(3)


class PoseEstimator:

    def __init__(self, config: SystemConfig) -> None:
        R_list = config.camera_to_world_rotation
        t_list = config.camera_to_world_translation
        self._R_c2w = np.array(R_list, dtype=np.float64)
        self._t_c2w = np.array(t_list, dtype=np.float64)

    def estimate(self, cuboid: CuboidGeometry) -> PoseEstimate:
        """
        Compute 6DoF pose from a reconstructed cuboid.

        Steps:
          1. Position = center of cuboid in camera frame
          2. Rotation matrix from cuboid principal axes
          3. Convert to quaternion + euler angles
          4. Transform to world frame if extrinsics provided
        """
        try:
            position_cam = cuboid.center.copy()
            R_cam = cuboid.rotation_matrix.copy()

            # Validate rotation matrix
            det = np.linalg.det(R_cam)
            if abs(abs(det) - 1.0) > 0.05:
                # Force proper rotation via SVD
                U, _, Vt = np.linalg.svd(R_cam)
                R_cam = U @ Vt
                if np.linalg.det(R_cam) < 0:
                    U[:, -1] *= -1
                    R_cam = U @ Vt

            rot = Rotation.from_matrix(R_cam)
            quaternion = rot.as_quat()  # [qx, qy, qz, qw] scipy convention
            # Convert to [qw, qx, qy, qz]
            quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
            euler_rad = rot.as_euler("ZYX")  # yaw, pitch, roll
            euler_deg = np.rad2deg(euler_rad)[[2, 1, 0]]  # → [roll, pitch, yaw]

            # World frame transform
            position_world = None
            rotation_world = None
            if not np.allclose(self._R_c2w, np.eye(3)):
                position_world = self._R_c2w @ position_cam + self._t_c2w
                R_world = self._R_c2w @ R_cam
                rotation_world = R_world

            return PoseEstimate(
                position=position_cam,
                rotation_matrix=R_cam,
                quaternion=quat_wxyz,
                euler_deg=euler_deg,
                position_world=position_world,
                rotation_world=rotation_world,
            )
        except Exception as e:
            raise PoseEstimationError(f"Pose estimation failed: {e}") from e

    def camera_to_world(self, position_cam: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to world frame."""
        return self._R_c2w @ position_cam + self._t_c2w

    def world_to_camera(self, position_world: np.ndarray) -> np.ndarray:
        """Transform a 3D point from world frame to camera frame."""
        return self._R_c2w.T @ (position_world - self._t_c2w)
