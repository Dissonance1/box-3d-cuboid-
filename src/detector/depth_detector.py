"""Depth-only box detector — no ML model required.

Segments box-like objects purely from the depth frame:
  1. RANSAC finds the dominant support plane (floor / table / conveyor)
  2. Removes the support plane; remaining points = objects above it
  3. DBSCAN clusters each object
  4. Filters clusters by volume and aspect ratio consistent with a box
  5. Returns a DetectionResult for each cluster, with a pixel-level mask

Why this works better than YOLO for plain cardboard boxes:
  - COCO-trained YOLO11 has no "box" class — it cannot detect plain cardboard
  - Depth segmentation is purely geometric — texture, color, lighting irrelevant
  - Works in low light, works on glossy labels, works on any box color

Limitations vs YOLO:
  - Cannot distinguish box from other rectangular objects (laptop, book)
  - No class name (always returns "box")
  - Less precise mask on box edges

Use this detector when YOLO is not trained on your specific box type.
"""

from __future__ import annotations

import numpy as np
import cv2

from src.detector.base_detector import BaseDetector
from src.utils.config import DetectorConfig
from src.utils.logger import get_logger
from src.utils.types import DetectionResult

logger = get_logger(__name__)


class DepthDetector(BaseDetector):
    """
    Segments box-shaped objects from the depth frame using RANSAC + DBSCAN.

    Requires the camera frame to be passed differently from the RGB-only
    BaseDetector interface. Wrap it via DepthAwareDetector which feeds
    the depth frame alongside the color frame.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._intrinsics = None

    def set_intrinsics(self, fx, fy, cx, cy) -> None:
        self._intrinsics = (fx, fy, cx, cy)

    def load(self) -> None:
        logger.info("DepthDetector loaded (no model weights needed)")

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        # DepthDetector can't work on RGB alone — use detect_from_depth()
        return []

    def detect_from_depth(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        min_depth: float = 0.25,
        max_depth: float = 2.0,
    ) -> list[DetectionResult]:
        """
        Detect box-like objects from the depth map using closest-layer approach.

        Strategy: boxes are usually the CLOSEST object to the camera.
        We segment the nearest depth layer (min_depth to min_depth + 0.55m),
        cluster it with DBSCAN, and return the best box-shaped cluster.
        This avoids the brittle "above floor plane" approach which fails when
        the box top IS the dominant plane in the scene.
        """
        if self._intrinsics is None:
            logger.warning("DepthDetector: intrinsics not set, using defaults")
            fx, fy, cx, cy = 607.08, 605.97, 314.09, 242.41
        else:
            fx, fy, cx, cy = self._intrinsics

        h, w = depth_m.shape

        # ── Step 1: Focus on foreground layer (closest 0.6m of valid range) ─
        # The target box is almost always the nearest large object.
        all_valid = depth_m[(depth_m > min_depth) & (depth_m < max_depth)]
        if len(all_valid) < 500:
            return []

        # Capture only the nearest 0.30m layer.
        # Boxes are always the closest foreground object; people/background
        # are typically >0.40m further. This narrow window excludes them.
        near_edge = float(np.percentile(all_valid, 3))
        far_limit  = min(near_edge + 0.30, max_depth)

        fg_mask = (depth_m > near_edge - 0.02) & (depth_m < far_limit)
        v_coords, u_coords = np.where(fg_mask)
        z = depth_m[v_coords, u_coords].astype(np.float64)
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy
        points = np.column_stack([x, y, z])

        if len(points) < 200:
            return []

        # ── Step 2: DBSCAN clustering ──────────────────────────────────────
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(0.008)
            labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=15))
        except Exception as e:
            logger.debug("DepthDetector DBSCAN failed: %s", e)
            return []

        candidates = []
        unique_labels = np.unique(labels[labels >= 0])

        for label in unique_labels:
            cluster_pts = np.asarray(pcd.points)[labels == label]
            if len(cluster_pts) < 30:
                continue

            # ── Step 5: Filter by box-like geometry ───────────────────────
            extents = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            dims_sorted = sorted(extents, reverse=True)
            L, W, H_est = dims_sorted

            # Minimum box size: 60mm per side
            if L < 0.060 or H_est < 0.040:
                continue
            # Maximum: single box ≤ 1.2m
            if L > 1.200:
                continue
            # Aspect ratio: longest/shortest ≤ 6
            if extents.min() > 0 and extents.max() / extents.min() > 6.0:
                continue

            # ── Step 6: Build pixel mask ───────────────────────────────────
            cluster_z = cluster_pts[:, 2]
            cluster_u = (cluster_pts[:, 0] * fx / cluster_z + cx).astype(int)
            cluster_v = (cluster_pts[:, 1] * fy / cluster_z + cy).astype(int)

            mask = np.zeros((h, w), dtype=np.uint8)
            valid_px = (cluster_u >= 0) & (cluster_u < w) & \
                       (cluster_v >= 0) & (cluster_v < h)
            mask[cluster_v[valid_px], cluster_u[valid_px]] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.dilate(mask, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

            if mask.sum() < 200:
                continue

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = np.array([x1, y1, x2, y2])

            # ── Step 7: Score — prefer large, centered, compact objects ───
            bbox_area = (x2 - x1) * (y2 - y1)
            fill_ratio = mask.sum() / max(bbox_area, 1)

            # Distance of bbox center from image center (normalised 0–1)
            img_cx, img_cy = w / 2, h / 2
            bbox_cx = (x1 + x2) / 2
            bbox_cy = (y1 + y2) / 2
            center_dist = np.sqrt(((bbox_cx - img_cx) / w) ** 2 +
                                  ((bbox_cy - img_cy) / h) ** 2)
            center_score = max(0.0, 1.0 - center_dist * 2)

            # Volume score: larger boxes score higher (penalise tiny noise)
            volume = float(extents[0] * extents[1] * extents[2])
            vol_score = min(volume / 0.002, 1.0)  # 0.002 m³ ≈ 25cm³ box

            conf = float(np.clip(
                0.30 * fill_ratio + 0.40 * center_score + 0.30 * vol_score,
                0.40, 0.92,
            ))

            candidates.append(DetectionResult(
                class_id=0,
                class_name="box",
                confidence=conf,
                bbox_xyxy=bbox,
                mask=mask,
                mask_score=conf,
            ))
            logger.debug(
                "DepthDetector candidate",
                dims_mm=[round(e * 1000) for e in extents],
                center_dist=round(center_dist, 2),
                conf=round(conf, 2),
            )

        # Return only the best candidate to avoid flooding the tracker
        # with background clutter. The highest-confidence one is the most
        # centered, largest, and compact object — most likely the target box.
        if not candidates:
            logger.debug("DepthDetector: no valid candidates")
            return []

        best = max(candidates, key=lambda d: d.confidence)
        logger.debug("DepthDetector: best candidate conf=%.2f", best.confidence)
        return [best]

    def unload(self) -> None:
        pass

    @staticmethod
    def _ransac_plane(
        points: np.ndarray,
        threshold: float = 0.015,
        n_iter: int = 500,
    ):
        """Fast RANSAC plane fitting. Returns (normal, offset) or (None, None)."""
        n = len(points)
        best_count = 0
        best_normal = None
        best_offset = None
        rng = np.random.default_rng(0)

        for _ in range(n_iter):
            idx = rng.choice(n, 3, replace=False)
            p1, p2, p3 = points[idx]
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            offset = -float(np.dot(normal, p1))
            dists = np.abs(points @ normal + offset)
            count = (dists < threshold).sum()
            if count > best_count:
                best_count = count
                best_normal = normal
                best_offset = offset
                if count > n * 0.3:
                    break

        if best_count < 100:
            return None, None

        # Refine with SVD on inliers
        inliers = points[np.abs(points @ best_normal + best_offset) < threshold]
        centroid = inliers.mean(axis=0)
        _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
        best_normal = Vt[-1]
        if np.dot(best_normal, best_normal) < 0.5:
            best_normal = -best_normal
        best_offset = -float(np.dot(best_normal, centroid))
        return (best_normal, best_offset), None
