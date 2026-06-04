"""Real-time visualization using OpenCV.

Renders into the BGR frame:
  1. Segmentation mask overlay (alpha-blended colored regions)
  2. 3D bounding cuboid (projected wireframe with depth-sorted edges)
  3. Coordinate axes at box center (RGB = XYZ)
  4. Dimension labels
  5. Track ID and confidence badge
  6. FPS counter and system status

Projection: 3D camera-frame points → 2D pixel using camera intrinsics.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.utils.types import BoxMeasurement, CameraIntrinsics, FrameResult

# Color palette for up to 32 concurrent tracks (BGR)
_TRACK_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 100, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (100, 255, 100), (100, 100, 255),
    (255, 150, 0), (0, 255, 150), (150, 0, 255), (0, 150, 255),
    (200, 200, 0), (0, 200, 200), (200, 0, 200), (150, 255, 0),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


class Visualizer:

    def __init__(self, config) -> None:
        self._config = config
        self._fps_history: list[float] = []
        self._last_time = time.perf_counter()

    def render(
        self,
        frame_bgr: np.ndarray,
        result: FrameResult,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """
        Render all detections onto the frame. Returns annotated BGR image.
        Does not modify the input frame (works on a copy).
        """
        canvas = frame_bgr.copy()

        for m in result.measurements:
            color = _track_color(m.track_id)
            if self._config.show_masks:
                canvas = self._render_mask(canvas, m, color)
            canvas = self._render_bbox(canvas, m, color)          # always drawn
            if self._config.show_cuboid:
                canvas = self._render_cuboid(canvas, m, intrinsics, color)
            if self._config.show_axes:
                canvas = self._render_axes(canvas, m, intrinsics)
            if self._config.show_dimensions:
                canvas = self._render_labels(canvas, m, intrinsics)

        if self._config.show_fps:
            canvas = self._render_hud(canvas, result)

        return canvas

    def _render_mask(
        self, canvas: np.ndarray, m: BoxMeasurement, color: tuple
    ) -> np.ndarray:
        mask = m.detection.mask
        if mask is None or mask.sum() == 0:
            return canvas
        overlay = canvas.copy()
        overlay[mask.astype(bool)] = color
        return cv2.addWeighted(overlay, self._config.mask_alpha, canvas, 1 - self._config.mask_alpha, 0)

    def _render_bbox(
        self, canvas: np.ndarray, m: BoxMeasurement, color: tuple
    ) -> np.ndarray:
        """Draw the 2D YOLO bounding box with a label chip."""
        bbox = m.detection.bbox_xyxy
        if bbox is None or len(bbox) != 4:
            return canvas

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = canvas.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Outer dark border for contrast, then colored rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label chip: filled background rectangle + white text
        d = m.dimensions
        label = (f"ID:{m.track_id} {m.detection.class_name}  "
                 f"{int(d.length_m*1000)}x{int(d.width_m*1000)}x{int(d.height_m*1000)}mm  "
                 f"{m.confidence.overall:.2f}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self._config.font_scale
        (tw, th), baseline = cv2.getTextSize(label, font, fs, 1)

        chip_y1 = max(0, y1 - th - baseline - 6)
        chip_y2 = y1
        chip_x2 = min(w - 1, x1 + tw + 8)
        cv2.rectangle(canvas, (x1, chip_y1), (chip_x2, chip_y2), color, -1)
        cv2.putText(canvas, label, (x1 + 4, y1 - baseline - 2),
                    font, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1 + 4, y1 - baseline - 2),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def _render_cuboid(
        self, canvas: np.ndarray, m: BoxMeasurement, intrinsics: CameraIntrinsics, color: tuple
    ) -> np.ndarray:
        corners_3d = m.cuboid.corners  # 8x3
        corners_2d = self._project_points(corners_3d, intrinsics)
        if corners_2d is None:
            return canvas

        # Standard cuboid edge connectivity
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        thick = self._config.cuboid_line_thickness
        for i, j in edges:
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[j].astype(int))
            if self._point_in_frame(pt1, canvas) and self._point_in_frame(pt2, canvas):
                cv2.line(canvas, pt1, pt2, color, thick, cv2.LINE_AA)

        return canvas

    def _render_axes(
        self, canvas: np.ndarray, m: BoxMeasurement, intrinsics: CameraIntrinsics
    ) -> np.ndarray:
        center = m.pose.position
        R = m.pose.rotation_matrix
        L = self._config.axis_length_m

        for i, color_bgr in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            endpoint = center + R[:, i] * L
            pts = self._project_points(np.array([center, endpoint]), intrinsics)
            if pts is None:
                continue
            p1, p2 = tuple(pts[0].astype(int)), tuple(pts[1].astype(int))
            if self._point_in_frame(p1, canvas) and self._point_in_frame(p2, canvas):
                cv2.arrowedLine(canvas, p1, p2, color_bgr, 2, cv2.LINE_AA, tipLength=0.3)

        return canvas

    def _render_labels(
        self, canvas: np.ndarray, m: BoxMeasurement, intrinsics: CameraIntrinsics
    ) -> np.ndarray:
        center_2d = self._project_points(m.pose.position[None], intrinsics)
        if center_2d is None:
            return canvas

        px, py = int(center_2d[0, 0]), int(center_2d[0, 1])
        if not self._point_in_frame((px, py), canvas):
            return canvas

        d = m.dimensions
        L_mm = int(d.length_m * 1000)
        W_mm = int(d.width_m * 1000)
        H_mm = int(d.height_m * 1000)

        lines = [
            f"ID:{m.track_id} {m.detection.class_name}",
            f"{L_mm}x{W_mm}x{H_mm}mm",
            f"Z:{m.pose.position[2]:.2f}m",
            f"Y:{m.pose.euler_deg[2]:.1f}deg",
            f"Conf:{m.confidence.overall:.2f}",
        ]
        color = _track_color(m.track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self._config.font_scale
        for li, line in enumerate(lines):
            y = py - 20 + li * int(18 * fs)
            x = max(5, px - 50)
            cv2.putText(canvas, line, (x, y), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (x, y), font, fs, color, 1, cv2.LINE_AA)

        return canvas

    def _render_hud(self, canvas: np.ndarray, result: FrameResult) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"FPS: {result.fps:.1f}",
            f"Objects: {result.count}",
            f"Frame: {result.frame_number}",
        ]
        for i, line in enumerate(lines):
            y = 20 + i * 22
            cv2.putText(canvas, line, (10, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _project_points(
        points_3d: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Optional[np.ndarray]:
        """Project Nx3 camera-frame 3D points to Nx2 pixel coordinates."""
        if points_3d is None or len(points_3d) == 0:
            return None
        z = points_3d[:, 2]
        if np.any(z <= 0):
            return None
        u = points_3d[:, 0] * intrinsics.fx / z + intrinsics.cx
        v = points_3d[:, 1] * intrinsics.fy / z + intrinsics.cy
        return np.column_stack([u, v])

    @staticmethod
    def _point_in_frame(pt: tuple[int, int], frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        return 0 <= pt[0] < w and 0 <= pt[1] < h
