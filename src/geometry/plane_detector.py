"""RANSAC-based multi-plane detection for box surface reconstruction.

Architecture decision: we do NOT use Open3D's built-in RANSAC plane segmentation
because it provides only the best single plane. This module implements iterative
RANSAC with orthogonality constraints to detect up to 6 planes simultaneously.

Algorithm:
  1. Run RANSAC to find the dominant plane → refine with SVD → remove inliers
  2. Repeat on residual points to find the next plane
  3. After finding N planes, apply orthogonality grouping and enforce constraints
  4. Pair anti-parallel planes (opposite box faces)

Coordinate convention: camera frame (X right, Y down, Z forward)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from src.utils.config import PlaneDetectionConfig
from src.utils.exceptions import PlaneDetectionError
from src.utils.logger import get_logger
from src.utils.metrics import GEOMETRY_ERRORS, GEOMETRY_LATENCY
from src.utils.types import PlaneModel, PointCloud

logger = get_logger(__name__)


class MultiPlaneDetector:
    """
    Iterative RANSAC plane detector with orthogonality enforcement.

    For a box, we expect at most 3 visible face directions (the 3 orthogonal
    axis-aligned normals as seen from the camera). Hidden faces inferred later.

    Usage:
        detector = MultiPlaneDetector(config)
        planes = detector.detect(point_cloud)
    """

    def __init__(self, config: PlaneDetectionConfig) -> None:
        self._config = config

    def detect(self, pc: PointCloud) -> list[PlaneModel]:
        """
        Detect up to max_planes planar surfaces in the point cloud.

        Returns list of PlaneModel sorted by inlier count (largest first).
        Raises PlaneDetectionError if no planes found.
        """
        t0 = time.perf_counter()
        points = pc.points.copy()
        found: list[PlaneModel] = []

        total_points = len(points)
        for iteration in range(self._config.max_planes):
            if len(points) < 3:
                break
            # Require that remaining points are a meaningful fraction of original
            if len(points) < total_points * 0.05:
                break

            plane, inlier_idx = self._ransac_plane(points)
            if plane is None:
                break

            inlier_ratio = len(inlier_idx) / total_points
            if inlier_ratio < self._config.ransac_min_inlier_ratio:
                break

            # Refine plane using SVD on all inliers
            inlier_pts = points[inlier_idx]
            refined_plane = self._refine_plane_svd(inlier_pts, plane)
            refined_plane.inlier_count = len(inlier_idx)
            refined_plane.inlier_ratio = inlier_ratio

            # Compute fitness and RMSE
            all_dists = np.abs(refined_plane.distance(points))
            rmse = float(np.sqrt(np.mean(
                all_dists[inlier_idx] ** 2
            )))
            refined_plane.rmse = rmse

            if not self._check_plane_area(inlier_pts):
                # Too small to be a box face — skip and continue
                points = np.delete(points, inlier_idx, axis=0)
                continue

            found.append(refined_plane)
            points = np.delete(points, inlier_idx, axis=0)

        if not found:
            GEOMETRY_ERRORS.inc()
            raise PlaneDetectionError("No valid planes found in point cloud")

        GEOMETRY_LATENCY.observe(time.perf_counter() - t0)
        logger.debug("Planes detected", count=len(found), iterations=len(found))
        return found

    def _ransac_plane(
        self, points: np.ndarray
    ) -> tuple[Optional[PlaneModel], np.ndarray]:
        """
        RANSAC plane fitting.

        Returns (PlaneModel, inlier_indices) or (None, empty) if failed.
        """
        n = len(points)
        best_inlier_count = 0
        best_normal = None
        best_offset = None
        best_inliers = np.array([], dtype=int)

        threshold = self._config.ransac_distance_threshold_m
        n_iter = self._config.ransac_n_iterations

        # Adaptive RANSAC: stop early if we found a good model
        required_inliers = int(n * 0.20)

        rng = np.random.default_rng(seed=42)  # deterministic for reproducibility

        for i in range(n_iter):
            # Sample 3 points
            try:
                idx = rng.choice(n, size=3, replace=False)
            except ValueError:
                break
            p1, p2, p3 = points[idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            offset = -float(np.dot(normal, p1))

            dists = np.abs(points @ normal + offset)
            inliers = np.where(dists < threshold)[0]

            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_normal = normal
                best_offset = offset
                best_inliers = inliers
                if best_inlier_count >= required_inliers:
                    break

        if best_normal is None or best_inlier_count < 3:
            return None, np.array([], dtype=int)

        centroid = points[best_inliers].mean(axis=0)
        plane = PlaneModel(
            normal=best_normal,
            offset=best_offset,
            inlier_count=best_inlier_count,
            inlier_ratio=0.0,
            fitness=0.0,
            rmse=0.0,
            centroid=centroid,
        )
        return plane, best_inliers

    @staticmethod
    def _refine_plane_svd(inlier_pts: np.ndarray, initial: PlaneModel) -> PlaneModel:
        """
        Refine plane normal using SVD on inlier point set.

        The plane normal is the eigenvector corresponding to the smallest
        eigenvalue of the covariance matrix, equivalent to the last row of V^T.
        """
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]
        # Ensure normal consistency with initial estimate
        if np.dot(normal, initial.normal) < 0:
            normal = -normal
        offset = -float(np.dot(normal, centroid))
        return PlaneModel(
            normal=normal,
            offset=offset,
            inlier_count=initial.inlier_count,
            inlier_ratio=initial.inlier_ratio,
            fitness=0.0,
            rmse=0.0,
            centroid=centroid,
        )

    def _check_plane_area(self, points: np.ndarray) -> bool:
        """Check that inlier set spans at least min_plane_area_m2."""
        if len(points) < 4:
            return False
        # Project to plane's 2D coordinate system to compute area
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # First two rows of Vt span the plane
        pts_2d = centered @ Vt[:2].T
        try:
            hull = ConvexHull(pts_2d)
            area = hull.volume  # In 2D, volume = area
            return area >= self._config.min_plane_area_m2
        except Exception:
            return False


def enforce_orthogonality(planes: list[PlaneModel], tolerance_deg: float = 15.0) -> list[PlaneModel]:
    """
    Adjust plane normals to enforce strict orthogonality among the
    dominant plane directions of a box.

    Strategy:
      1. Cluster normals into 1–3 direction groups (anti-parallel = same group)
      2. Find consensus normal per group via circular mean
      3. Orthogonalize the 3 axis directions using Gram-Schmidt
      4. Assign each plane to its corrected normal
    """
    if len(planes) < 2:
        return planes

    tol_rad = np.deg2rad(tolerance_deg)
    normals = np.array([p.normal for p in planes])

    groups = _cluster_normals(normals, tol_rad)
    if len(groups) < 2:
        return planes

    # Compute mean direction per group
    axes = []
    for group_idx in groups:
        group_normals = normals[group_idx]
        # Flip normals that point opposite to the first
        ref = group_normals[0]
        aligned = [n if np.dot(n, ref) >= 0 else -n for n in group_normals]
        axis = np.mean(aligned, axis=0)
        axis /= np.linalg.norm(axis)
        axes.append(axis)

    # Orthogonalize with Gram-Schmidt if we have 3 axes
    if len(axes) >= 3:
        axes = _gram_schmidt(axes[:3])

    # Assign each plane to its nearest axis
    corrected = []
    for plane in planes:
        best_axis = max(axes, key=lambda a: abs(np.dot(a, plane.normal)))
        sign = np.sign(np.dot(best_axis, plane.normal))
        corrected_normal = sign * best_axis
        corrected.append(PlaneModel(
            normal=corrected_normal,
            offset=-float(np.dot(corrected_normal, plane.centroid)),
            inlier_count=plane.inlier_count,
            inlier_ratio=plane.inlier_ratio,
            fitness=plane.fitness,
            rmse=plane.rmse,
            centroid=plane.centroid,
        ))
    return corrected


def _cluster_normals(normals: np.ndarray, tol_rad: float) -> list[list[int]]:
    """Simple greedy clustering of normals by angular distance."""
    groups: list[list[int]] = []
    assigned = [False] * len(normals)

    for i, n in enumerate(normals):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(normals)):
            if assigned[j]:
                continue
            cos_angle = abs(np.dot(n, normals[j]))
            angle = np.arccos(np.clip(cos_angle, 0, 1))
            if angle < tol_rad:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def _gram_schmidt(axes: list[np.ndarray]) -> list[np.ndarray]:
    """Orthogonalize 3 vectors using modified Gram-Schmidt."""
    u = [a.copy() for a in axes]
    for i in range(len(u)):
        u[i] /= (np.linalg.norm(u[i]) + 1e-12)
        for j in range(i + 1, len(u)):
            u[j] = u[j] - np.dot(u[j], u[i]) * u[i]
    return [v / (np.linalg.norm(v) + 1e-12) for v in u]
