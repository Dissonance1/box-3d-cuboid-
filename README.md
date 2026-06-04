# Box 3D Cuboid Detection & Measurement

Real-time 3D bounding box detection, dimension measurement, and pose estimation for carton boxes using Intel RealSense D435i and YOLO11 segmentation.

---

## Pipeline

```
YOLO11-seg  →  detects box region (bbox only)
     ↓
Depth Camera  →  crops depth to bbox region
     ↓
Surface Mask  →  pixels at z_top ± tolerance (box top face)
     ↓
Depth Blob    →  connected components → box top boundary
     ↓
Corners       →  inner-percentile rectangle from blob contour
     ↓
3D Cuboid     →  top face (depth corners) + bottom face (z_floor projection)
     ↓
Dimensions    →  L × W × H in mm
```

**YOLO** only answers: *"Is there a box here?"*  
**Depth** does all geometry: corners, size, height, orientation.

---

## Hardware

| Component | Spec |
|---|---|
| Camera | Intel RealSense D435i |
| Resolution | 640 × 480 @ 30 fps |
| GPU | NVIDIA (CUDA, FP16) |
| Camera height | 990 mm above floor (configurable) |

---

## Quick Start

```bash
# Install dependencies
pip install pyrealsense2 ultralytics open3d opencv-python scipy

# Run live detection
python live_view.py
```

### Keys

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `D` | Toggle depth overlay |
| `+` / `-` | Increase / decrease depth tolerance |

---

## Display

Three panels side by side:

| Panel | Shows |
|---|---|
| Left | RGB with 3D cuboid overlay |
| Middle | Full depth heatmap |
| Right | Masked box (depth colours, black background) |

---

## Configuration

Edit `live_view.py` top section:

```python
CAMERA_HEIGHT_MM = 990   # camera-to-floor distance in mm
TOLERANCE_MM     = 30    # ±mm around box top depth
Z_PUSH_MM        = 50    # visual offset to seat cuboid on surface
MIN_BOX_PX       = 80    # minimum blob area (lower for small boxes)
```

---

## Training Your Own Model

### 1. Collect data

```bash
python scripts/collect_training_data.py
# Press SPACE to save frames, Q to quit
```

### 2. Annotate corners

```bash
python scripts/annotate_corners.py
# Click 4 corners per box: TL → TR → BR → BL
```

### 3. Train YOLO11-seg

```bash
python scripts/train_carton.py
# Trains yolo11n-seg on your dataset
# Best model saved to models/carton_seg_best.pt
```

### 4. Export to TensorRT (optional, for Jetson)

```bash
python scripts/export_tensorrt.py --model models/carton_seg_best.pt --fp16
```

---

## Project Structure

```
├── live_view.py              # Main detection script
├── main.py                   # Full production pipeline entry point
├── config/
│   ├── d435i.yaml            # D435i camera config
│   ├── default.yaml          # Default config
│   └── jetson.yaml           # Jetson edge config
├── src/
│   ├── camera/               # RealSense + dummy camera drivers
│   ├── detector/             # YOLO + depth detectors
│   ├── geometry/             # RANSAC plane + cuboid reconstruction
│   ├── pointcloud/           # Open3D point cloud pipeline
│   ├── tracking/             # ByteTrack + Kalman filter
│   ├── pipeline/             # End-to-end pipeline orchestrator
│   └── api/                  # FastAPI REST endpoints
├── scripts/
│   ├── collect_training_data.py
│   ├── annotate_corners.py
│   ├── train_carton.py
│   └── validate_accuracy.py
├── tests/                    # Unit + integration + performance tests
├── deploy/                   # systemd service + Docker configs
├── Dockerfile
└── docker-compose.yml
```

---

## REST API (full pipeline)

```bash
python main.py --config config/d435i.yaml
```

| Endpoint | Description |
|---|---|
| `GET /health` | System health |
| `GET /detections` | All detected boxes with pose + dimensions |
| `GET /pose/{id}` | Pose for specific track ID |
| `GET /metrics` | FPS, latency, track count |
| `GET /docs` | Swagger UI |

---

## Accuracy (D435i at 990mm overhead)

| Measurement | Target | Achieved |
|---|---|---|
| L × W dimensions | ±10 mm | ±15–20 mm |
| Height | ±5 mm | ±5–10 mm |
| Yaw angle | ±5° | ±3–8° |
| FPS | 15–30 | ~18–25 |

---

## Docker

```bash
docker compose up
```

---

## License

Private — Meridian Data Labs
