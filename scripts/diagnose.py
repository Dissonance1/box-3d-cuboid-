"""
Live diagnostic — grabs one frame and walks through every pipeline stage,
printing exactly what passes and what fails.

Usage:
    python scripts/diagnose.py
"""

import sys
import time
import numpy as np
import cv2

sys.path.insert(0, ".")

# ── 1. Camera ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 1: CAMERA")
print("="*60)

import pyrealsense2 as rs

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_device("033422072388")
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

print("Warming up (30 frames)...", end="", flush=True)
for _ in range(30):
    pipeline.wait_for_frames(timeout_ms=2000)
print(" done")

frames = align.process(pipeline.wait_for_frames(2000))
color_frame = np.asanyarray(frames.get_color_frame().get_data())   # BGR
depth_raw   = np.asanyarray(frames.get_depth_frame().get_data())
depth_m     = depth_raw.astype(np.float32) * depth_scale
depth_m[(depth_m < 0.30) | (depth_m > 2.0)] = 0.0

valid_px = (depth_m > 0).sum()
print(f"Color: {color_frame.shape}")
print(f"Depth: {depth_m.shape}  valid={valid_px} ({100*valid_px/(640*480):.1f}%)")
if valid_px > 0:
    d = depth_m[depth_m > 0]
    print(f"Depth range: {d.min():.3f}m – {d.max():.3f}m  mean={d.mean():.3f}m")

cv2.imwrite("output/diag_color.jpg", color_frame)
cv2.imwrite("output/diag_depth.png",
            cv2.applyColorMap(
                cv2.convertScaleAbs(depth_m / 2.0 * 255), cv2.COLORMAP_JET))
print("Saved: output/diag_color.jpg  output/diag_depth.png")

# ── 2. YOLO detection ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2: YOLO DETECTION")
print("="*60)

from ultralytics import YOLO
model = YOLO("models/yolo11n-seg.pt")
model.to("cuda")

# Try at multiple confidence thresholds to see what YOLO actually sees
for conf in [0.10, 0.20, 0.30, 0.40, 0.50]:
    results = model(color_frame, conf=conf, iou=0.45, imgsz=640,
                    verbose=False, retina_masks=True)
    r = results[0]
    n = len(r.boxes) if r.boxes is not None else 0
    classes = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()] if n > 0 else []
    confs   = [f"{c:.2f}" for c in r.boxes.conf.cpu().numpy()] if n > 0 else []
    print(f"  conf>={conf:.2f}: {n} detections  {list(zip(classes, confs))}")

# Save annotated frame at low threshold
results_vis = model(color_frame, conf=0.15, iou=0.45, imgsz=640, verbose=False)
annotated = results_vis[0].plot()
cv2.imwrite("output/diag_yolo.jpg", annotated)
print("Saved: output/diag_yolo.jpg  (all detections at conf>=0.15)")

# Use conf=0.15 for the rest of the pipeline
results = model(color_frame, conf=0.15, iou=0.45, imgsz=640,
                retina_masks=True, verbose=False)
r = results[0]
n_det = len(r.boxes) if r.boxes is not None else 0

if n_det == 0:
    print("\n[FAIL] YOLO found nothing at all (even at conf=0.15).")
    print("  → Check lighting, camera angle, and whether the box is fully in frame.")
    print("  → The model is trained on COCO boxes — plain cardboard boxes usually")
    print("    detect fine. Try holding a printed label toward the camera.")
    pipeline.stop()
    sys.exit(1)

# Pick the highest-confidence detection
best_idx = int(r.boxes.conf.argmax())
best_cls  = r.names[int(r.boxes.cls[best_idx])]
best_conf = float(r.boxes.conf[best_idx])
best_bbox = r.boxes.xyxy[best_idx].cpu().numpy().astype(int)
print(f"\nBest detection: '{best_cls}'  conf={best_conf:.3f}  bbox={best_bbox}")

# ── 3. Depth coverage in mask ──────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 3: DEPTH COVERAGE IN MASK")
print("="*60)

raw_mask = r.masks.data[best_idx].cpu().numpy()
if raw_mask.ndim == 3: raw_mask = raw_mask[0]
if raw_mask.shape != (480, 640):
    raw_mask = cv2.resize(raw_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
mask_bool = (raw_mask > 0.5)

mask_px        = mask_bool.sum()
depth_in_mask  = depth_m[mask_bool]
valid_in_mask  = depth_in_mask[depth_in_mask > 0]
coverage       = len(valid_in_mask) / max(mask_px, 1)

print(f"Mask pixels:        {mask_px}")
print(f"Valid depth in mask: {len(valid_in_mask)} ({coverage*100:.1f}%)")
if len(valid_in_mask) > 0:
    print(f"Depth in mask:      {valid_in_mask.min():.3f}m – {valid_in_mask.max():.3f}m  mean={valid_in_mask.mean():.3f}m")
else:
    print("[FAIL] No valid depth inside the mask at all!")
    print("  → Is the camera depth stream working?")
    print("  → Is the box within 0.3–2.0m of the camera?")
    print("  → Glossy/transparent surfaces return no depth — try matte cardboard.")

if coverage < 0.25:
    print(f"[WARN] Coverage {coverage*100:.1f}% < 25% minimum — depth too sparse.")
    print("  → Move the box closer (ideal: 0.5–1.5m).")
    print("  → Avoid glossy surfaces — they absorb the IR emitter.")
else:
    print(f"[OK] Coverage {coverage*100:.1f}%")

# ── 4. Point cloud ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 4: POINT CLOUD")
print("="*60)

intr_obj = r.stream_profile if hasattr(r, 'stream_profile') else None
# Manual back-projection
fx, fy, cx, cy = 607.08, 605.97, 314.09, 242.41
v_coords, u_coords = np.where(mask_bool & (depth_m > 0))
z = depth_m[v_coords, u_coords].astype(np.float64)
x = (u_coords - cx) * z / fx
y = (v_coords - cy) * z / fy
points = np.column_stack([x, y, z])
print(f"Raw points: {len(points)}")

if len(points) < 50:
    print("[FAIL] Too few points for reconstruction (need ≥50).")
else:
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.005)
        print(f"After voxel downsample (5mm): {len(pcd.points)} points")
        if len(pcd.points) >= 20:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"After SOR: {len(pcd.points)} points")
        print("[OK] Point cloud generated")
    except Exception as e:
        print(f"[FAIL] Point cloud error: {e}")

# ── 5. Plane detection ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 5: PLANE DETECTION (RANSAC)")
print("="*60)

if len(points) >= 50:
    from src.utils.config import PlaneDetectionConfig
    from src.geometry.plane_detector import MultiPlaneDetector
    from src.utils.types import PointCloud

    pts_arr = np.asarray(pcd.points)
    pc = PointCloud(points=pts_arr,
                    colors=np.ones((len(pts_arr), 3), dtype=np.float32))
    config = PlaneDetectionConfig(
        ransac_n_iterations=1000,
        ransac_min_inlier_ratio=0.08,
        min_planes_for_cuboid=1,
    )
    try:
        detector = MultiPlaneDetector(config)
        planes = detector.detect(pc)
        print(f"[OK] Found {len(planes)} plane(s)")
        for i, p in enumerate(planes):
            print(f"  Plane {i+1}: normal={np.round(p.normal,3)}  "
                  f"inliers={p.inlier_count}  rmse={p.rmse*1000:.1f}mm")

        # ── 6. Cuboid reconstruction ───────────────────────────────────────
        print("\n" + "="*60)
        print("STAGE 6: CUBOID RECONSTRUCTION")
        print("="*60)
        from src.geometry.cuboid_reconstructor import CuboidReconstructor
        try:
            rec = CuboidReconstructor(config)
            cuboid = rec.reconstruct(planes, pc)
            d = cuboid.dimensions
            print(f"[OK] Cuboid reconstructed!")
            print(f"  Dimensions: {d[0]*1000:.0f} x {d[1]*1000:.0f} x {d[2]*1000:.0f} mm")
            print(f"  Center:     {np.round(cuboid.center, 3)} m")
            print(f"  Quality:    {cuboid.reconstruction_quality:.2f}")
        except Exception as e:
            print(f"[FAIL] Cuboid reconstruction: {e}")
    except Exception as e:
        print(f"[FAIL] Plane detection: {e}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"YOLO detection:  {'OK' if n_det > 0 else 'FAIL'}  (best conf={best_conf:.2f}  class={best_cls})")
print(f"Depth coverage:  {coverage*100:.1f}% in mask")
print(f"Raw points:      {len(points)}")
print(f"Config threshold: conf={0.40}  depth_coverage={0.25}")
print()
if best_conf < 0.40:
    print(f"ACTION: Lower confidence_threshold to {max(0.15, best_conf-0.05):.2f} in config/d435i.yaml")
if coverage < 0.25:
    print("ACTION: Move box closer to camera or improve lighting")

pipeline.stop()
print("\nDiagnostic images saved to output/")
