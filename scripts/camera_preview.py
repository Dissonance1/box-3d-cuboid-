"""Live RGB + depth preview for any configured camera.

usage: python scripts/camera_preview.py [config/ae400.yaml]
Keys: Q/Esc=quit  S=save snapshot to output/
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.camera.camera_factory import create_camera
from src.utils.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/ae400.yaml")
win = f"{cfg.camera.type} preview  -  Q=quit  S=save"
snap = 0

with create_camera(cfg.camera) as cam:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 450)
    t_prev = time.time()
    while True:
        f = cam.get_frame(2000)
        now = time.time()
        fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(f.depth, alpha=255.0 / cfg.camera.depth_max_m), cv2.COLORMAP_JET
        )
        depth_vis[f.depth == 0] = 0
        cy, cx = f.height // 2, f.width // 2
        cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)

        out = np.hstack([f.color, depth_vis])
        valid = 100.0 * np.count_nonzero(f.depth) / f.depth.size
        text = f"{fps:4.1f} fps   center {f.depth[cy, cx] * 1000:.0f} mm   valid depth {valid:.0f}%"
        cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, out)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("s"):
            snap += 1
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(f"output/preview_{snap}.jpg", out)
            print(f"Saved output/preview_{snap}.jpg")

cv2.destroyAllWindows()
