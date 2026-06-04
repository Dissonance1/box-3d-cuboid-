"""Depth map processing for ROI extraction and validation.

Extracts valid depth within a segmentation mask, applies edge-noise rejection,
and computes depth statistics used for confidence scoring downstream.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.config import DepthConfig
from src.utils.exceptions import InsufficientDepthError
from src.utils.logger import get_logger
from src.utils.types import CameraFrame, DepthROI, DetectionResult

logger = get_logger(__name__)


class DepthProcessor:
    """Extract and validate depth data within a segmentation mask ROI."""

    def __init__(self, config: DepthConfig) -> None:
        self._config = config

    def extract_roi(
        self,
        frame: CameraFrame,
        detection: DetectionResult,
        refined_mask: np.ndarray,
    ) -> DepthROI:
        """
        Extract depth pixels within the refined segmentation mask.

        Depth validity rules (per pixel):
          1. Pixel is within the eroded mask
          2. Depth is within [min_valid, max_valid]
          3. Not a depth edge (high local gradient)

        Raises InsufficientDepthError if valid coverage < min_depth_coverage.
        """
        depth = frame.depth  # HxW float32 meters

        # Mask pixels within ROI
        mask_bool = refined_mask.astype(bool)
        depth_roi = depth.copy()
        depth_roi[~mask_bool] = 0.0

        # Range filter
        valid = (depth_roi > self._config.min_valid_depth_m) & \
                (depth_roi < self._config.max_valid_depth_m) & \
                mask_bool

        # Edge noise rejection: remove pixels at sharp depth discontinuities
        if self._config.gradient_threshold_m > 0:
            valid = valid & ~self._depth_edge_mask(depth_roi, self._config.gradient_threshold_m)

        depth_roi[~valid] = 0.0

        total_mask_pixels = mask_bool.sum()
        valid_pixels = valid.sum()
        valid_ratio = float(valid_pixels) / float(total_mask_pixels) if total_mask_pixels > 0 else 0.0

        if valid_ratio < self._config.min_depth_coverage:
            raise InsufficientDepthError(valid_ratio, self._config.min_depth_coverage)

        valid_depths = depth_roi[valid]
        return DepthROI(
            mask=refined_mask,
            depth_map=depth_roi,
            valid_ratio=valid_ratio,
            mean_depth=float(valid_depths.mean()),
            min_depth=float(valid_depths.min()),
            max_depth=float(valid_depths.max()),
        )

    @staticmethod
    def _depth_edge_mask(depth: np.ndarray, threshold_m: float) -> np.ndarray:
        """
        Return True for pixels at sharp depth transitions.
        Uses Sobel gradient magnitude on the depth image.
        """
        # Convert 0-invalid to NaN for gradient computation
        d32 = depth.copy()
        d32[d32 == 0] = np.nan

        # Sobel gradient (handle NaN by replacing with 0 for gradient)
        d_filled = np.where(np.isnan(d32), 0.0, d32).astype(np.float32)
        gx = cv2.Sobel(d_filled, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(d_filled, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx ** 2 + gy ** 2)
        return gradient_mag > threshold_m

    def estimate_box_height_overhead(
        self,
        depth_roi: DepthROI,
        camera_height_m: float,
    ) -> float | None:
        """
        For overhead cameras: estimate box height from camera height minus
        the depth to the top face.

          box_height = camera_height - mean_top_face_depth

        Returns None if camera_height_m is 0.0 (not configured).
        """
        if camera_height_m <= 0.0:
            return None
        top_face_depth = depth_roi.mean_depth
        height = camera_height_m - top_face_depth
        if height < 0.010 or height > camera_height_m * 0.95:
            return None
        return height

    def compute_depth_confidence(self, depth_roi: DepthROI) -> float:
        """
        Confidence metric based on depth completeness and consistency.

        Score components:
          - Coverage ratio: valid_pixels / mask_pixels
          - Uniformity: low std-dev relative to mean depth → consistent surface
        """
        coverage_score = min(depth_roi.valid_ratio / 0.70, 1.0)

        valid_depths = depth_roi.depth_map[depth_roi.depth_map > 0]
        if len(valid_depths) == 0:
            return 0.0

        # Relative standard deviation; flat box surface should have low dispersion
        rel_std = np.std(valid_depths) / (np.mean(valid_depths) + 1e-6)
        uniformity_score = max(0.0, 1.0 - rel_std / 0.1)

        return float(0.6 * coverage_score + 0.4 * uniformity_score)
