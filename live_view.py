"""
Multi-box 3D cuboid detection using depth histogram peaks.

How it works:
  1. Depth histogram → find all significant peaks
     Each peak = one flat surface at that depth (box top face)
  2. For each peak → threshold depth ± tolerance → mask
  3. Connected components → individual box blobs
  4. For each box blob → corners → floor depth → 3D cuboid

Works for any number of boxes at any different depths:
  Box 1 top at 200mm, Box 2 top at 500mm → 2 clear histogram peaks
  → 2 independent masks → 2 independent cuboids

Keys: Q=quit  S=save  +/-=tolerance
"""

import sys, os
import cv2
import numpy as np
from scipy.signal import find_peaks

try:
    import pyrealsense2 as rs
except ImportError:
    print("pip install pyrealsense2"); sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("pip install ultralytics"); sys.exit(1)

MODEL_PATH = "models/carton_seg_best.pt"

SERIAL        = "033422072388"
W, H          = 640, 480
FX, FY        = 607.08, 605.97
CX, CY        = 314.09, 242.41
TOLERANCE_MM      = 30    # ±mm around each peak depth — wider catches small/thin boxes
Z_PUSH_MM         = 50    # push top face slightly deeper so it sits on box
MASK_ERODE_PX     = 8     # erode blob by this many pixels before corner fit
                           # trims 1–3mm of noisy edges from the bounding box

# ── Known camera setup ──────────────────────────────────────────────
# Set CAMERA_HEIGHT_MM to your actual camera-to-floor distance.
# When set, H = CAMERA_HEIGHT_MM - z_top_of_blob  (exact, no guessing).
# Set to 0 to use automatic floor detection from depth data.
CAMERA_HEIGHT_MM  = 990    # exact camera-to-floor distance in mm

# Box search range — only look for objects between these depths
# Boxes are between floor and camera, leave some margin
BOX_MIN_MM        = 800    # ignore anything closer than 800mm (not a box)
BOX_MAX_MM        = 985    # ignore floor at 1000mm (stop 15mm before)

# Histogram settings (used only when CAMERA_HEIGHT_MM = 0)
BIN_MM        = 10     # depth bin size in mm
MIN_PEAK_PX   = 300    # minimum pixels in a peak to count as a surface
MIN_BOX_PX    = 80     # minimum blob area — lowered to detect small objects
PEAK_MIN_DIST = 50     # mm — peaks closer than this merge into one
FLOOR_SKIP    = True   # skip deepest peak (floor/table)

COLORS = [
    (0, 255,  60),
    (0, 200, 255),
    (255, 100,  0),
    (200,   0, 255),
    (255, 200,   0),
]


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — Find depth peaks in scene
# ════════════════════════════════════════════════════════════════════════

def find_surface_depths(depth_m: np.ndarray) -> list[float]:
    """
    Find box-top surface depths.

    Mode 1 (CAMERA_HEIGHT_MM > 0):
      Use the known camera height. Search only within
      [BOX_MIN_MM, BOX_MAX_MM]. Returns a single 'virtual' depth
      that covers the entire box-search band — connected components
      will separate individual boxes spatially.

    Mode 2 (CAMERA_HEIGHT_MM = 0):
      Histogram peak detection — finds multiple depth peaks
      automatically. Works when boxes are >50mm apart in height.
    """
    if CAMERA_HEIGHT_MM > 0:
        # Known setup: just confirm there are pixels in the box range
        lo = BOX_MIN_MM / 1000.0;  hi = BOX_MAX_MM / 1000.0
        count = int(((depth_m > lo) & (depth_m < hi)).sum())
        if count < MIN_BOX_PX:
            return []
        # Return the midpoint of the box search range as a representative depth
        # (the actual per-blob z_top is computed from the blob pixels)
        return [(lo + hi) / 2.0]

    # ── Automatic histogram mode ─────────────────────────────────────
    valid = depth_m[(depth_m > 0.10) & (depth_m < 2.50)]
    if len(valid) < 500:
        return []

    d_min  = float(valid.min())
    d_max  = float(valid.max())
    n_bins = max(10, int((d_max - d_min) * 1000 / BIN_MM))
    hist, edges = np.histogram(valid, bins=n_bins, range=(d_min, d_max))

    kernel = np.ones(3) / 3.0
    hist_s = np.convolve(hist, kernel, mode='same')

    min_dist_bins = max(1, int(PEAK_MIN_DIST / BIN_MM))
    peaks, _      = find_peaks(hist_s, height=MIN_PEAK_PX,
                                distance=min_dist_bins)
    if len(peaks) == 0:
        return []

    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    depths      = sorted([float(bin_centres[p]) for p in peaks])

    if FLOOR_SKIP and len(depths) > 1:
        depths = depths[:-1]   # drop deepest peak = floor

    return depths


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — Mask for one surface depth
# ════════════════════════════════════════════════════════════════════════

def surface_mask(depth_m: np.ndarray,
                 z_surface: float,
                 tol_m: float) -> np.ndarray:
    """
    Binary mask: keep pixels within z_surface ± tol_m.

    TOLERANCE_MM (+/- keys) always controls this directly.
    Increase if the box has depth holes (white surface needs more).
    Decrease if the floor/background leaks into the mask.
    """
    lo = z_surface - tol_m
    hi = z_surface + tol_m

    mask = ((depth_m > lo) & (depth_m < hi) & (depth_m > 0.10)
            ).astype(np.uint8) * 255

    k5 = np.ones((9, 9), np.uint8)
    k3 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    return mask


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — Split mask into individual box blobs
# ════════════════════════════════════════════════════════════════════════

def split_blobs(mask: np.ndarray) -> list[np.ndarray]:
    """
    Connected components → list of individual box masks.
    Rejects blobs smaller than MIN_BOX_PX (noise / reflections).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):   # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BOX_PX:
            blob = (labels == i).astype(np.uint8) * 255
            blobs.append(blob)
    # Sort largest first
    blobs.sort(key=lambda b: b.sum(), reverse=True)
    return blobs


# ════════════════════════════════════════════════════════════════════════
# STEP 4 — Corner detection from a blob mask
# ════════════════════════════════════════════════════════════════════════

def corners_from_blob(blob: np.ndarray):
    """
    Find the INNERMOST boundary of the blob to define the box corners.

    The depth mask boundary is noisy/wavy.  Instead of using the
    outermost noisy edge (minAreaRect), we:
      1. Get the orientation from minAreaRect (accurate even on noisy contour)
      2. Rotate all contour points to align with the box axes
      3. Take the INNER percentile of points on each side
         (e.g. 15th percentile = line where 85% of points are inside)
      4. Construct a clean rectangle from those inner lines
      5. Rotate back

    This clips off all the bumpy outer noise and gives the tightest
    rectangle that fits inside the actual box boundary.
    """
    INNER_PCT = 15   # percentile from each edge — clips noisy bumps, keeps real boundary

    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None

    # 1. Get orientation from minAreaRect
    rect  = cv2.minAreaRect(cnt)
    angle = rect[2]   # degrees

    # 2. Rotate all contour pixels to align with box axes
    theta = np.deg2rad(angle)
    R     = np.array([[ np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
    pts   = cnt.reshape(-1, 2).astype(np.float64)
    rot   = pts @ R.T   # shape (N, 2)

    # 3. Inner boundary = percentile from each side
    x_lo = np.percentile(rot[:, 0], INNER_PCT)
    x_hi = np.percentile(rot[:, 0], 100 - INNER_PCT)
    y_lo = np.percentile(rot[:, 1], INNER_PCT)
    y_hi = np.percentile(rot[:, 1], 100 - INNER_PCT)

    if x_hi - x_lo < 5 or y_hi - y_lo < 5:
        return None   # degenerate

    # 4. Build inner rectangle in rotated frame
    inner_rot = np.array([
        [x_lo, y_lo],
        [x_hi, y_lo],
        [x_hi, y_hi],
        [x_lo, y_hi],
    ], dtype=np.float64)

    # 5. Rotate back to image frame
    R_inv   = R.T
    corners = (inner_rot @ R_inv.T).astype(np.float32)

    # Sort clockwise
    ctr     = corners.mean(0)
    corners = corners[np.argsort(np.arctan2(corners[:,1]-ctr[1],
                                             corners[:,0]-ctr[0]))]
    return corners


# ════════════════════════════════════════════════════════════════════════
# STEP 5 — Floor depth around a specific blob
# ════════════════════════════════════════════════════════════════════════

def floor_depth(depth_m: np.ndarray,
                blob:    np.ndarray,
                z_top:   float,
                all_surfaces_mask: np.ndarray) -> float:
    """
    Returns the floor depth for this blob.

    If CAMERA_HEIGHT_MM is set → use that directly (exact, no estimation).
    Otherwise → sample pixels in a ring outside the blob.
    """
    if CAMERA_HEIGHT_MM > 0:
        return CAMERA_HEIGHT_MM / 1000.0   # always exact

    expanded = cv2.dilate(blob, np.ones((80, 80), np.uint8))
    ring     = (expanded > 0) & (blob == 0) & (all_surfaces_mask == 0)
    samples  = depth_m[ring & (depth_m > z_top + 0.025) & (depth_m < z_top + 0.400)]
    if len(samples) < 30:
        return 0.0
    return float(np.percentile(samples, 35))  # high percentile = floor depth (deeper pixels)


# ════════════════════════════════════════════════════════════════════════
# STEP 6 — Projection helpers
# ════════════════════════════════════════════════════════════════════════

def back_project(u, v, z):
    return ((u - CX)*z/FX, (v - CY)*z/FY)

def project(xw, yw, z):
    if z <= 0.01: return None
    return (int(xw*FX/z + CX), int(yw*FY/z + CY))


# ════════════════════════════════════════════════════════════════════════
# STEP 7 — Draw 3D cuboid
# ════════════════════════════════════════════════════════════════════════

def draw_cuboid(canvas, corners_px, z_top_draw, z_floor, color,
                z_top_measure=None):
    """
    z_top_draw    — depth used for visual projection (z_top + push offset)
    z_top_measure — depth used for dimension calculation (actual z_top, no push)
                    defaults to z_top_draw if not supplied
    Separating them fixes the ~2% inflation caused by Z_PUSH_MM.
    """
    if z_top_measure is None:
        z_top_measure = z_top_draw

    bright = color
    dim    = tuple(max(0, c - 80) for c in color)

    if z_floor <= z_top_draw + 0.010:
        top2d = [(int(round(u)), int(round(v))) for u,v in corners_px]
        for i in range(4):
            cv2.line(canvas, top2d[i], top2d[(i+1)%4], (0,0,0), 3, cv2.LINE_AA)
            cv2.line(canvas, top2d[i], top2d[(i+1)%4], bright,  2, cv2.LINE_AA)
        return canvas

    # Back-project corners to world, then re-project both faces.
    # This makes top and bottom face consistent with each other.
    top_world = [back_project(u, v, z_top_draw) for u,v in corners_px]

    # Bottom face from world corners at z_floor
    bot2d = [project(xw, yw, z_floor) or (int(round(corners_px[i][0])),
                                            int(round(corners_px[i][1])))
             for i, (xw,yw) in enumerate(top_world)]

    # Top face derived FROM the bottom face using perspective scale.
    # Both faces share the same world XY, so:
    #   top_px - center = (bot_px - center) * (z_floor / z_top)
    # This makes the top face exactly as large as it should be —
    # no reliance on noisy mask edges.
    scale = z_floor / z_top_draw
    top2d = [
        (int(round(CX + (bx - CX) * scale)),
         int(round(CY + (by - CY) * scale)))
        for (bx, by) in bot2d
    ]

    # Vertical edges
    for i in range(4):
        cv2.line(canvas, top2d[i], bot2d[i], (0,0,0), 3, cv2.LINE_AA)
        cv2.line(canvas, top2d[i], bot2d[i], dim,     2, cv2.LINE_AA)
    # Bottom face
    for i in range(4):
        cv2.line(canvas, bot2d[i], bot2d[(i+1)%4], (0,0,0), 2, cv2.LINE_AA)
        cv2.line(canvas, bot2d[i], bot2d[(i+1)%4], dim,     1, cv2.LINE_AA)
    # Top face
    for i in range(4):
        cv2.line(canvas, top2d[i], top2d[(i+1)%4], (0,0,0), 4, cv2.LINE_AA)
        cv2.line(canvas, top2d[i], top2d[(i+1)%4], bright,  2, cv2.LINE_AA)
    # Corner dots
    for pt in top2d:
        cv2.circle(canvas, pt, 6, (0,0,0),      -1)
        cv2.circle(canvas, pt, 4, (255,255,255), -1)

    # ── Dimension calculation uses z_top_measure (no push offset) ────
    # Back-project corners using true z_top → accurate L and W.
    # Use edge lengths between adjacent corners (not axis-aligned min/max)
    # so diagonal boxes are measured correctly too.
    wc = [back_project(u, v, z_top_measure) for u,v in corners_px]  # 4 world corners

    e0 = np.sqrt((wc[1][0]-wc[0][0])**2 + (wc[1][1]-wc[0][1])**2)  # edge 0→1
    e1 = np.sqrt((wc[2][0]-wc[1][0])**2 + (wc[2][1]-wc[1][1])**2)  # edge 1→2
    L  = max(e0, e1) * 1000   # mm
    Ww = min(e0, e1) * 1000   # mm
    H_box = (z_floor - z_top_measure) * 1000

    cx_px = int(np.mean([p[0] for p in top2d]))
    cy_px = int(np.mean([p[1] for p in top2d]))
    for k, txt in enumerate([f"H={H_box:.0f}mm", f"L={L:.0f}mm", f"W={Ww:.0f}mm"]):
        y = cy_px - 20 + k*18
        cv2.putText(canvas, txt, (cx_px+8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0), 3)
        cv2.putText(canvas, txt, (cx_px+8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color,   1)
    return canvas


def depth_colormap(depth_m):
    d = depth_m.copy(); d[d<0.10]=0; d[d>2.50]=0
    return cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255/2.0),
                             cv2.COLORMAP_JET)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    global TOLERANCE_MM

    print("Opening D435i...")
    pipeline    = rs.pipeline()
    cfg         = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile     = pipeline.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align       = rs.align(rs.stream.color)

    sp = rs.spatial_filter();  sp.set_option(rs.option.filter_magnitude, 2)
    tp = rs.temporal_filter(); tp.set_option(rs.option.filter_smooth_alpha, 0.5)
    hf = rs.hole_filling_filter(); hf.set_option(rs.option.holes_fill, 2)

    for _ in range(30): pipeline.wait_for_frames()

    # ── Load YOLO ────────────────────────────────────────────────────
    mp = MODEL_PATH if os.path.exists(MODEL_PATH) else "models/yolo11n-seg.pt"
    print(f"Loading YOLO: {mp}")
    model = YOLO(mp)
    model(np.zeros((H, W, 3), dtype=np.uint8), verbose=False)
    print("Ready.  Q=quit  S=save  +/-=tolerance")
    print(f"Camera height: {CAMERA_HEIGHT_MM}mm   Tolerance: ±{TOLERANCE_MM}mm")

    win  = "YOLO + Depth  —  Multi-Box 3D Cuboid"
    snap = 0
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W * 3 + 20, H)

    while True:
        try:
            fs = align.process(pipeline.wait_for_frames(1200))
        except RuntimeError:
            continue
        cf = fs.get_color_frame(); df = fs.get_depth_frame()
        if not cf or not df: continue

        df      = hf.process(tp.process(sp.process(df)))
        color   = np.asanyarray(cf.get_data()).copy()
        depth_m = np.asanyarray(df.as_depth_frame().get_data()).astype(np.float32) * depth_scale

        tol_m = TOLERANCE_MM / 1000.0

        # ── Step 1: YOLO — detection only, no geometry ────────────────
        # YOLO answers ONE question: "Is there a box here?"
        # It gives an approximate bounding box region.
        # Everything else (corners, size, height) comes from depth.
        yres = model(color, conf=0.20, iou=0.45, imgsz=640,
                     device="0", half=True,
                     retina_masks=False,   # don't need masks
                     verbose=False)[0]

        detections = []   # list of (x1,y1,x2,y2) bounding rectangles
        if yres.boxes is not None:
            for i in range(len(yres.boxes)):
                b = yres.boxes.xyxy[i].cpu().numpy().astype(int)
                detections.append((max(0,b[0]), max(0,b[1]),
                                   min(W,b[2]), min(H,b[3])))

        # ── Step 2: For each detected region — use depth to find box ──
        # Process:
        #   a) Crop depth to YOLO region  (just a rectangle — no mask)
        #   b) z_top = median depth in that region
        #   c) Surface mask = all depth pixels at z_top ± TOLERANCE_MM
        #      → This is the box top face in pure depth data
        #      → This is what panel 3 shows
        #   d) Corners from that depth surface mask
        #   e) H = CAMERA_HEIGHT_MM − z_top  (exact)
        box_panel      = np.zeros_like(color)
        all_blob_union = np.zeros((H, W), dtype=np.uint8)
        n_boxes        = 0

        for i, (x1, y1, x2, y2) in enumerate(detections):
            col = COLORS[i % len(COLORS)]

            # a) Crop depth to YOLO region
            depth_roi = np.zeros_like(depth_m)
            depth_roi[y1:y2, x1:x2] = depth_m[y1:y2, x1:x2]

            valid = (depth_roi > 0.10) & (depth_roi < 2.50)
            if valid.sum() < 30:
                continue

            # b) z_top — exclude floor pixels first, then median
            # Without this: large bbox with more floor than box pixels
            # → median = floor depth → mask inverts (bg green, box black)
            floor_m = CAMERA_HEIGHT_MM / 1000.0 if CAMERA_HEIGHT_MM > 0 else 2.50
            near    = depth_roi[(valid) & (depth_roi < floor_m - 0.025)]
            if len(near) < 20:
                near = depth_roi[valid]
            z_top = float(np.percentile(near, 50))

            # c) Surface mask at z_top ± TOLERANCE_MM
            #    Only pixels on the box top face — background removed
            blob_mask = surface_mask(depth_roi, z_top, tol_m)

            blobs = split_blobs(blob_mask)
            if not blobs:
                continue
            blob = blobs[0]

            # d) Corners from the depth surface mask
            corners = corners_from_blob(blob)
            if corners is None:
                continue

            # e) Height
            z_floor = floor_depth(depth_m, blob, z_top,
                                   np.zeros((H, W), np.uint8))

            # Draw
            z_draw = z_top + Z_PUSH_MM / 1000.0
            color     = draw_cuboid(color,     corners, z_draw, z_floor, col,
                                    z_top_measure=z_top)
            box_panel = draw_cuboid(box_panel, corners, z_draw, z_floor, col,
                                    z_top_measure=z_top)

            all_blob_union = np.maximum(all_blob_union, blob)
            n_boxes += 1

        # ── Panel 2: full depth heatmap (unmasked) ───────────────────
        full_depth = depth_colormap(depth_m)

        # ── Panel 3: masked object in depth colours (black background) ──
        masked_depth = np.zeros_like(color)
        if all_blob_union.sum() > 0:
            masked_depth[all_blob_union > 0] = full_depth[all_blob_union > 0]

        # ── Labels ───────────────────────────────────────────────────
        def lbl(panel, text):
            cv2.putText(panel, text, (8,22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0,0,0),     3)
            cv2.putText(panel, text, (8,22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (220,220,220), 1)

        lbl(color,        f"RGB  —  {n_boxes} box(es)")
        lbl(full_depth,   f"Depth (full)  tol=±{TOLERANCE_MM}mm  +/-=adjust")
        lbl(masked_depth, f"Masked object  ({n_boxes} detected)")

        # ── Compose: RGB | full depth | masked object ─────────────────
        out = np.zeros((H, W*3+20, 3), dtype=np.uint8)
        out[:, :W]             = color
        out[:, W+10:W*2+10]    = full_depth
        out[:, W*2+20:W*3+20]  = masked_depth
        out[:, W+4:W+10]       = 60
        out[:, W*2+14:W*2+20]  = 60

        cv2.imshow(win, out)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27): break
        elif k == ord('s'):
            snap += 1; os.makedirs("output", exist_ok=True)
            cv2.imwrite(f"output/snapshot_{snap}.jpg", out)
            print(f"Saved snapshot_{snap}.jpg")
        elif k in (ord('+'), ord('=')):
            TOLERANCE_MM = min(TOLERANCE_MM + 5, 200)
            print(f"Tolerance: ±{TOLERANCE_MM}mm")
        elif k == ord('-'):
            TOLERANCE_MM = max(TOLERANCE_MM - 5, 5)
            print(f"Tolerance: ±{TOLERANCE_MM}mm")

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
