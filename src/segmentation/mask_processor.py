"""Segmentation mask processing and refinement.

After YOLO produces raw binary masks, this module:
  1. Morphologically cleans the mask (removes small holes and noise)
  2. Erodes the boundary to avoid depth edge contamination
  3. Computes mask quality metrics
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.config import DepthConfig
from src.utils.logger import get_logger
from src.utils.types import DetectionResult

logger = get_logger(__name__)


class MaskProcessor:
    """Refine raw YOLO segmentation masks for downstream depth extraction."""

    def __init__(self, config: DepthConfig) -> None:
        self._config = config
        # Erosion kernel: removes edge pixels that have mixed depth / foreground
        self._erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.depth_dilation_px + 1, 2 * config.depth_dilation_px + 1),
        )
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, detection: DetectionResult) -> np.ndarray:
        """
        Return a refined binary mask (HxW uint8, values 0 or 1).

        Steps:
          1. Morphological closing: fill small holes inside object
          2. Morphological opening: remove small noise blobs
          3. Largest connected component: select primary object blob
          4. Boundary erosion: pull mask inward by depth_dilation_px
        """
        mask = detection.mask.copy().astype(np.uint8)
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        # Fill small interior holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

        # Remove small exterior noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)

        # Keep only the largest connected component
        mask = self._largest_component(mask)

        # Erode to avoid depth edge contamination
        if self._config.depth_dilation_px > 0:
            mask = cv2.erode(mask, self._erosion_kernel)

        return mask

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if n_labels <= 1:
            return mask
        # Label 0 is background; find largest foreground component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.argmax()) + 1
        return (labels == largest).astype(np.uint8)

    @staticmethod
    def compute_mask_quality(mask: np.ndarray) -> float:
        """
        Quality score 0-1 based on:
          - Compactness (4π·area / perimeter²) — 1.0 = perfect circle
          - Area as fraction of bbox area
        """
        if mask.sum() == 0:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            return 0.0
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        # Normalize: rectangular boxes have compactness ~0.785 (π/4)
        # Penalize very thin or irregular masks
        score = min(compactness / (np.pi / 4), 1.0)
        return float(score)
