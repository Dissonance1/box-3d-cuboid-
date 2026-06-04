"""Synthetic camera for testing without physical hardware.

Generates realistic-looking RGB + depth frames of a synthetic scene
containing 1–3 boxes at known positions. Used for:
  - CI/CD pipeline testing
  - Development without a RealSense connected
  - Accuracy validation against known ground truth

Activate via config:  camera.type: dummy
Or via env var:       BOX_POSE__CAMERA__TYPE=dummy
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.camera.base_camera import BaseCamera
from src.utils.config import CameraConfig
from src.utils.types import CameraFrame, CameraIntrinsics


# Synthetic D455 intrinsics at 1280x720
_DUMMY_INTRINSICS = CameraIntrinsics(
    fx=909.15,
    fy=908.94,
    cx=641.20,
    cy=358.80,
    width=1280,
    height=720,
    depth_scale=0.001,
)

# Ground-truth boxes for validation: (cx, cy, cz, L, W, H, yaw_deg)
GROUND_TRUTH_BOXES = [
    (0.05,  0.10, 1.20, 0.400, 0.300, 0.200, 15.0),
    (-0.25, 0.05, 1.00, 0.250, 0.200, 0.150,  0.0),
    (0.30,  0.15, 1.40, 0.500, 0.350, 0.250, -20.0),
]


class DummyCamera(BaseCamera):
    """
    Synthetic RGB-D camera producing frames with rendered box scenes.

    The depth map is computed analytically from box geometry — no noise model
    for clean testing. Add noise_std_m > 0 for realistic sensor simulation.
    """

    def __init__(self, config: CameraConfig, noise_std_m: float = 0.0015) -> None:
        super().__init__()
        self._config = config
        self._noise_std = noise_std_m
        self._intrinsics = _DUMMY_INTRINSICS
        self._rng = np.random.default_rng(seed=42)
        self._frame_count = 0

    def open(self) -> None:
        self._running = True
        print("[DummyCamera] Synthetic camera started — no physical hardware needed.")
        print(f"[DummyCamera] Scene: {len(GROUND_TRUTH_BOXES)} boxes")
        for i, (cx, cy, cz, L, W, H, yaw) in enumerate(GROUND_TRUTH_BOXES):
            print(f"  Box {i+1}: {int(L*1000)}x{int(W*1000)}x{int(H*1000)} mm  "
                  f"at ({cx:.2f}, {cy:.2f}, {cz:.2f}) m  yaw={yaw}°")

    def close(self) -> None:
        self._running = False

    def get_frame(self, timeout_ms: int = 1000) -> CameraFrame:
        color, depth = self._render_scene()
        ts_ns = int(time.time() * 1e9)
        self._frame_count += 1
        return CameraFrame(
            color=color,
            depth=depth,
            timestamp_ns=ts_ns,
            frame_number=self._frame_count,
            intrinsics=self._intrinsics,
        )

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def is_connected(self) -> bool:
        return True

    def _render_scene(self) -> tuple[np.ndarray, np.ndarray]:
        H, W = self._intrinsics.height, self._intrinsics.width
        color = np.full((H, W, 3), (60, 55, 50), dtype=np.uint8)   # dark floor background
        depth = np.full((H, W), self._config.depth_max_m, dtype=np.float32)

        intr = self._intrinsics

        for box_idx, (cx, cy, cz, L, bW, bH, yaw_deg) in enumerate(GROUND_TRUTH_BOXES):
            yaw = np.deg2rad(yaw_deg)
            R = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0,           1, 0           ],
                [-np.sin(yaw),0, np.cos(yaw) ],
            ])
            box_color_bgr = [
                (60, 130, 200),   # blue-ish
                (60, 180,  80),   # green-ish
                (200, 100, 50),   # orange-ish
            ][box_idx % 3]

            # Render top face (overhead camera)
            # Sample a dense grid over the top face in box local coords
            n = 80
            u_vals = np.linspace(-L/2, L/2, n)
            w_vals = np.linspace(-bW/2, bW/2, n)
            uu, ww = np.meshgrid(u_vals, w_vals)
            top_local = np.column_stack([
                uu.ravel(),
                np.full(n*n, -bH/2),   # top face at -H/2 in Y (camera Y=down)
                ww.ravel(),
            ])
            top_world = (R @ top_local.T).T + np.array([cx, cy, cz])

            for pt in top_world:
                px_x, px_y, px_z = pt
                if px_z <= 0:
                    continue
                u = int(px_x * intr.fx / px_z + intr.cx)
                v = int(px_y * intr.fy / px_z + intr.cy)
                if 0 <= u < W and 0 <= v < H:
                    if px_z < depth[v, u]:
                        depth[v, u] = float(px_z)
                        color[v, u] = box_color_bgr

        # Floor plane at camera_max - 0.1 (flat surface)
        floor_depth = self._config.depth_max_m * 0.9
        depth[depth >= self._config.depth_max_m] = floor_depth

        # Add D455-like depth noise
        if self._noise_std > 0:
            noise = self._rng.normal(0, self._noise_std, depth.shape).astype(np.float32)
            depth = np.clip(depth + noise,
                            self._config.depth_min_m,
                            self._config.depth_max_m)

        # Draw a simple box outline on color image for visual reference
        for box_idx, (cx, cy, cz, L, bW, bH, yaw_deg) in enumerate(GROUND_TRUTH_BOXES):
            proj_cx = int(cx * intr.fx / cz + intr.cx)
            proj_cy = int(cy * intr.fy / cz + intr.cy)
            label = f"GT Box {box_idx+1}: {int(L*1000)}x{int(bW*1000)}x{int(bH*1000)}mm"
            cv2.putText(color, label, (proj_cx - 60, proj_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return color, depth
