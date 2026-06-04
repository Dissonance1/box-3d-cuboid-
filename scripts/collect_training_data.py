"""
Collect training images from the RealSense camera.

Usage:
    python scripts/collect_training_data.py

Controls:
    SPACE = save current frame
    Q     = quit

Tip: place the box at many different angles and distances.
Save 50-100 images per angle variation.
"""

import os, sys, time
import cv2, numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("pip install pyrealsense2"); sys.exit(1)

SERIAL    = "033422072388"
SAVE_DIR  = "datasets/carton_corners/images/train"
W, H      = 640, 480

os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_device(SERIAL)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
pipeline.start(cfg)

for _ in range(30): pipeline.wait_for_frames()   # warmup

count   = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
print(f"Existing images in {SAVE_DIR}: {count}")
print("SPACE = save frame   Q = quit")

while True:
    fs    = pipeline.wait_for_frames()
    frame = np.asanyarray(fs.get_color_frame().get_data())
    disp  = frame.copy()

    # Guide overlay
    cv2.putText(disp, f"Saved: {count}  | SPACE=save  Q=quit",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,0,0), 3)
    cv2.putText(disp, f"Saved: {count}  | SPACE=save  Q=quit",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,255,0), 1)

    cv2.imshow("Collect Training Data", disp)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'): break
    elif k == ord(' '):
        path = os.path.join(SAVE_DIR, f"frame_{count:05d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
        print(f"Saved: {path}")

pipeline.stop()
cv2.destroyAllWindows()
print(f"\nTotal images collected: {count}")
print(f"Next step: annotate with Roboflow or CVAT")
print(f"Images are in: {SAVE_DIR}")
