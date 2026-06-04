"""Camera calibration validation.

Validates intrinsic parameters at startup and warns when values fall outside
expected ranges for the known RealSense D455/D435 hardware specifications.
"""

from __future__ import annotations

import numpy as np

from src.utils.exceptions import CalibrationError
from src.utils.logger import get_logger
from src.utils.types import CameraIntrinsics

logger = get_logger(__name__)

# Approximate focal lengths for RealSense sensors at common resolutions
_EXPECTED_FOCAL_RANGES = {
    (1280, 720): (900, 1000),
    (848, 480): (600, 700),
    (640, 480): (380, 660),   # D435I at 640x480: fx≈606 (color), varies by firmware
    (640, 360): (350, 500),
}

_MAX_PRINCIPAL_OFFSET_RATIO = 0.15  # principal point within 15% of centre


class CalibrationValidator:
    """Validates that camera intrinsics are physically plausible."""

    def validate(self, intrinsics: CameraIntrinsics) -> None:
        self._check_focal_length(intrinsics)
        self._check_principal_point(intrinsics)
        self._check_depth_scale(intrinsics)
        self._check_distortion(intrinsics)
        logger.info(
            "Calibration validated",
            fx=round(intrinsics.fx, 2),
            fy=round(intrinsics.fy, 2),
            cx=round(intrinsics.cx, 2),
            cy=round(intrinsics.cy, 2),
            depth_scale=intrinsics.depth_scale,
        )

    def _check_focal_length(self, intr: CameraIntrinsics) -> None:
        key = (intr.width, intr.height)
        if key in _EXPECTED_FOCAL_RANGES:
            lo, hi = _EXPECTED_FOCAL_RANGES[key]
            for name, f in [("fx", intr.fx), ("fy", intr.fy)]:
                if not (lo <= f <= hi):
                    raise CalibrationError(
                        f"{name}={f:.1f} outside expected [{lo},{hi}] for "
                        f"{intr.width}x{intr.height} — calibration may be corrupt"
                    )
        else:
            if not (200 < intr.fx < 3000):
                raise CalibrationError(f"Implausible fx={intr.fx:.1f}")
            if not (200 < intr.fy < 3000):
                raise CalibrationError(f"Implausible fy={intr.fy:.1f}")

    def _check_principal_point(self, intr: CameraIntrinsics) -> None:
        cx_offset = abs(intr.cx - intr.width / 2) / intr.width
        cy_offset = abs(intr.cy - intr.height / 2) / intr.height
        if cx_offset > _MAX_PRINCIPAL_OFFSET_RATIO:
            logger.warning(
                "Principal point cx far from image center",
                cx=intr.cx,
                center_x=intr.width / 2,
                offset_ratio=round(cx_offset, 3),
            )
        if cy_offset > _MAX_PRINCIPAL_OFFSET_RATIO:
            logger.warning(
                "Principal point cy far from image center",
                cy=intr.cy,
                center_y=intr.height / 2,
                offset_ratio=round(cy_offset, 3),
            )

    def _check_depth_scale(self, intr: CameraIntrinsics) -> None:
        # RealSense units are in mm (scale=0.001) or 0.1mm (scale=0.0001)
        if not (1e-5 < intr.depth_scale < 0.01):
            raise CalibrationError(
                f"Unexpected depth_scale={intr.depth_scale} — expected 0.0001–0.001"
            )

    def _check_distortion(self, intr: CameraIntrinsics) -> None:
        k1 = intr.distortion_coeffs[0] if len(intr.distortion_coeffs) > 0 else 0.0
        if abs(k1) > 1.5:
            logger.warning("Large radial distortion k1=%f — verify calibration", k1)


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float:
    """Compute mean reprojection error for a set of 3D-2D correspondences."""
    K = intrinsics.matrix
    projected = (K @ points_3d.T).T
    projected_2d = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected_2d - points_2d, axis=1)
    return float(errors.mean())
