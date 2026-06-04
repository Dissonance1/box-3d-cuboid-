"""Point cloud quality validation."""

from __future__ import annotations

import numpy as np

from src.utils.config import PointCloudConfig
from src.utils.exceptions import InsufficientPointsError
from src.utils.logger import get_logger
from src.utils.types import PointCloud

logger = get_logger(__name__)


class PointCloudValidator:

    def __init__(self, config: PointCloudConfig) -> None:
        self._config = config

    def validate(self, pc: PointCloud) -> None:
        if pc.size < self._config.min_points_for_reconstruction:
            raise InsufficientPointsError(pc.size, self._config.min_points_for_reconstruction)
        self._check_spatial_extent(pc)

    def _check_spatial_extent(self, pc: PointCloud) -> None:
        """Warn if point cloud bounding box is suspiciously small or large."""
        extents = pc.points.max(axis=0) - pc.points.min(axis=0)
        if any(e < 0.005 for e in extents[:2]):
            logger.warning("Point cloud very flat in XY — may be floor plane leak", extents=extents.tolist())
        if any(e > 3.0 for e in extents):
            logger.warning("Point cloud extent > 3m — possible filter failure", extents=extents.tolist())

    def compute_density_score(self, pc: PointCloud) -> float:
        """
        Score 0-1 based on point density per cubic centimetre.

        At 5mm voxels after downsampling, a 40x30x20 cm box has ~9600 voxels.
        Scale to produce 1.0 at ≥5000 points.
        """
        score = min(pc.size / 5000.0, 1.0)
        return float(score)
