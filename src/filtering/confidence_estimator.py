"""Composite confidence score computation.

Aggregates 6 independent quality signals into a single 0-1 score:

  1. Segmentation confidence    (YOLO output score)
  2. Depth completeness         (valid depth pixel ratio)
  3. Point density              (points relative to expected)
  4. Plane quality              (RANSAC fitness + RMSE)
  5. Cuboid quality             (reconstruction completeness)
  6. Tracking stability         (Kalman filter NIS ratio)

Weights are tuned for the warehouse box dimensioning use case where
dimension accuracy is the primary objective.
"""

from __future__ import annotations

from src.utils.types import (
    ConfidenceBreakdown,
    CuboidGeometry,
    DepthROI,
    DetectionResult,
    PlaneModel,
    PointCloud,
)

_WEIGHTS = {
    "segmentation": 0.15,
    "depth": 0.20,
    "density": 0.15,
    "plane": 0.20,
    "cuboid": 0.20,
    "tracking": 0.10,
}


class ConfidenceEstimator:

    def compute(
        self,
        detection: DetectionResult,
        depth_roi: DepthROI,
        point_cloud: PointCloud,
        cuboid: CuboidGeometry,
        tracking_stability: float,
    ) -> ConfidenceBreakdown:

        seg_conf = float(detection.confidence)
        depth_conf = self._depth_score(depth_roi)
        density_score = self._density_score(point_cloud)
        plane_score = self._plane_score(cuboid.faces)
        cuboid_score = float(cuboid.reconstruction_quality)
        tracking_score = float(tracking_stability)

        overall = (
            _WEIGHTS["segmentation"] * seg_conf
            + _WEIGHTS["depth"] * depth_conf
            + _WEIGHTS["density"] * density_score
            + _WEIGHTS["plane"] * plane_score
            + _WEIGHTS["cuboid"] * cuboid_score
            + _WEIGHTS["tracking"] * tracking_score
        )
        overall = max(0.0, min(1.0, overall))

        return ConfidenceBreakdown(
            segmentation=round(seg_conf, 3),
            depth_completeness=round(depth_conf, 3),
            point_density=round(density_score, 3),
            plane_quality=round(plane_score, 3),
            cuboid_quality=round(cuboid_score, 3),
            tracking_stability=round(tracking_score, 3),
            overall=round(overall, 3),
        )

    @staticmethod
    def _depth_score(depth_roi: DepthROI) -> float:
        coverage = min(depth_roi.valid_ratio / 0.80, 1.0)
        depth_range = depth_roi.max_depth - depth_roi.min_depth
        flatness = max(0.0, 1.0 - depth_range / 0.30)
        return float(0.7 * coverage + 0.3 * flatness)

    @staticmethod
    def _density_score(pc: PointCloud) -> float:
        return min(pc.size / 5000.0, 1.0)

    @staticmethod
    def _plane_score(planes: list[PlaneModel]) -> float:
        if not planes:
            return 0.0
        fitnesses = [p.fitness for p in planes if p.inlier_count > 0]
        rmses = [p.rmse for p in planes if p.inlier_count > 0]
        if not fitnesses:
            return 0.1

        mean_fitness = sum(fitnesses) / len(fitnesses)
        mean_rmse = sum(rmses) / len(rmses) if rmses else 0.0
        rmse_score = max(0.0, 1.0 - mean_rmse / 0.015)
        count_score = min(len([p for p in planes if p.inlier_count > 0]) / 3.0, 1.0)
        return float(0.4 * mean_fitness + 0.4 * rmse_score + 0.2 * count_score)
