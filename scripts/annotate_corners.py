"""
Manual corner annotation tool.

Click 4 corners of each box IN ORDER:
  1 = Top-Left
  2 = Top-Right
  3 = Bottom-Right
  4 = Bottom-Left

Controls:
  Left click = place corner
  R          = reset current box
  SPACE      = save and next image
  Q          = quit

Usage:
    python scripts/annotate_corners.py

Produces YOLO-keypoint label files in:
    datasets/carton_corners/labels/train/
"""

import os, sys, glob
import cv2, numpy as np

IMG_DIR  = "datasets/carton_corners/images/train"
LBL_DIR  = "datasets/carton_corners/labels/train"
os.makedirs(LBL_DIR, exist_ok=True)

CORNER_NAMES  = ["TL","TR","BR","BL"]
CORNER_COLORS = [(0,200,255),(0,255,0),(255,100,0),(255,0,200)]

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                glob.glob(os.path.join(IMG_DIR, "*.png")))

if not images:
    print(f"No images found in {IMG_DIR}")
    print("Run collect_training_data.py first.")
    sys.exit(0)

# Skip already-labeled images
def is_labeled(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.exists(os.path.join(LBL_DIR, base + ".txt"))

images = [img for img in images if not is_labeled(img)]
print(f"Images to annotate: {len(images)}")
if not images:
    print("All images already labeled!")
    sys.exit(0)

W_IMG, H_IMG = 640, 480
clicks = []     # list of (x, y) pixel coords for current box
img_idx = 0

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))

cv2.namedWindow("Annotate Corners", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotate Corners", mouse_cb)

while img_idx < len(images):
    img_path = images[img_idx]
    img      = cv2.imread(img_path)
    h, w     = img.shape[:2]

    clicks.clear()
    print(f"\n[{img_idx+1}/{len(images)}] {os.path.basename(img_path)}")
    print("  Click 4 corners: TL → TR → BR → BL")
    print("  R=reset  SPACE=save  Q=quit")

    while True:
        disp = img.copy()

        # Draw guide
        guide = ["Click 4 corners: TL → TR → BR → BL",
                 "R=reset  SPACE=save & next  Q=quit",
                 f"Image {img_idx+1}/{len(images)}"]
        for gi, g in enumerate(guide):
            cv2.putText(disp, g, (8, 20+gi*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
            cv2.putText(disp, g, (8, 20+gi*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

        # Draw placed corners
        for i, (px, py) in enumerate(clicks):
            cv2.circle(disp, (px,py), 8, (0,0,0), -1)
            cv2.circle(disp, (px,py), 6, CORNER_COLORS[i], -1)
            cv2.putText(disp, CORNER_NAMES[i], (px+10,py-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        CORNER_COLORS[i], 2)

        # Draw box outline if all 4 placed
        if len(clicks) == 4:
            for i in range(4):
                cv2.line(disp, clicks[i], clicks[(i+1)%4],
                         (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(disp, "SPACE to save", (8, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        else:
            nxt = CORNER_NAMES[len(clicks)]
            cv2.putText(disp, f"Next: click {nxt}",
                        (8, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        CORNER_COLORS[len(clicks)], 2)

        cv2.imshow("Annotate Corners", disp)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('q'):
            cv2.destroyAllWindows(); sys.exit(0)
        elif k == ord('r'):
            clicks.clear()
            print("  Reset")
        elif k == ord(' ') and len(clicks) == 4:
            # Save YOLO keypoint label
            # class cx cy bw bh  kp1x kp1y 2  kp2x kp2y 2  kp3x kp3y 2  kp4x kp4y 2
            xs = [c[0] for c in clicks]; ys = [c[1] for c in clicks]
            cx = (min(xs)+max(xs))/2/w; cy = (min(ys)+max(ys))/2/h
            bw = (max(xs)-min(xs))/w;   bh = (max(ys)-min(ys))/h
            kp_str = "  ".join(f"{c[0]/w:.6f} {c[1]/h:.6f} 2"
                                for c in clicks)
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kp_str}"

            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(LBL_DIR, base + ".txt")
            with open(lbl_path, "w") as f:
                f.write(line + "\n")
            print(f"  Saved: {lbl_path}")
            img_idx += 1
            break

cv2.destroyAllWindows()
print("\nAnnotation complete!")
print(f"Labels saved to: {LBL_DIR}")
print("Next: python scripts/train_corners.py")
