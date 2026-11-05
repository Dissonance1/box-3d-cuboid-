# Box 3D Cuboid Detection & Measurement

Real-time 3D bounding box detection, dimension measurement, and pose estimation for carton boxes using an RGB-D camera (Intel RealSense D435i or LIPSedge AE400) and YOLO11 segmentation.

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
| Camera | Intel RealSense D435i (USB) **or** LIPSedge AE400 (PoE / Ethernet) |
| Resolution | D435i 640 × 480 @ 30 fps · AE400 1280 × 720 @ 30 fps |
| GPU | NVIDIA (CUDA, FP16) — falls back to CPU if none |
| Camera height | 990 mm above floor (configurable) |

---

## How to Run

### 1. Install

```bash
pip install -r requirements.txt
```

- **D435i:** install the [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases).
- **AE400:** install the **LIPSedge Camera SDK** (AE400/AE450 package). The driver expects it at
  `C:\Program Files\LIPSedge Camera SDK 1.02\LIPSedge AE400_AE450 2.4.4.3_v1.5.5\OpenNI2\Redist`;
  if yours is elsewhere, set `camera.openni_redist` in `config/ae400.yaml`.
- **Model:** put the trained carton model at `models/carton_seg_best.pt` (not in git).
  Without it, the generic `yolo11n-seg.pt` is downloaded, which does **not** detect boxes well.

### 2. Connect the camera

**D435i** — plug into a USB 3 port.

**AE400 (PoE)** — wire it as:

```
[PoE injector / switch] ─ PoE port ──→ AE400
                        └─ LAN port ──→ laptop Ethernet
```

The AE400's default IP is `192.168.0.100`. The laptop's Ethernet port needs an address on the
same subnet. If your Wi-Fi is also on `192.168.0.x`, also add a route so traffic to the camera goes
over the cable rather than Wi-Fi. One-time setup, in **PowerShell as Administrator**:

```powershell
New-NetIPAddress -InterfaceAlias Ethernet -IPAddress 192.168.0.50 -PrefixLength 32
New-NetRoute -DestinationPrefix 192.168.0.100/32 -InterfaceAlias Ethernet -RouteMetric 1
```

Check the camera is reachable: open `http://192.168.0.100/` in a browser. You should see a page titled **AE400**.

To undo the network setup later:

```powershell
Remove-NetRoute -DestinationPrefix 192.168.0.100/32 -Confirm:$false
Remove-NetIPAddress -IPAddress 192.168.0.50 -Confirm:$false
```

### 3. Check the camera feed (optional)

```bash
python scripts/camera_preview.py config/ae400.yaml    # live RGB + depth, Q=quit
```

### 4. Run live detection

Run from the project root:

```bash
python live_view.py config/ae400.yaml   # LIPSedge AE400
python live_view.py                     # D435i (uses config/d435i.yaml)
```

Before measuring, set `CAMERA_HEIGHT_MM`, `BOX_MIN_MM` and `BOX_MAX_MM` at the top of
`live_view.py` to match how your camera is mounted (see [Configuration](#configuration)).

### Keys

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `0` | Show all three views side by side (default) |
| `1` / `2` / `3` | Show only RGB / depth / masked view |
| `S` | Save snapshot of the current view to `output/` |
| `+` / `-` | Increase / decrease depth tolerance |

---

## Display

Three panels side by side (press `1`/`2`/`3` to show one full-size, `0` for all):

| Panel | Shows |
|---|---|
| Left | RGB with 3D cuboid overlay |
| Middle | Full depth heatmap |
| Right | Masked box (depth colours, black background) |

---

## Configuration

The camera (type, IP, resolution, depth range) comes from the YAML file passed on the command line:
`config/ae400.yaml` or `config/d435i.yaml`. Lens intrinsics are read from the camera automatically.

Scene settings are at the top of `live_view.py`:

```python
CAMERA_HEIGHT_MM = 990   # camera-to-floor distance in mm
BOX_MIN_MM       = 800   # ignore anything closer than this
BOX_MAX_MM       = 985   # ignore the floor (just above CAMERA_HEIGHT_MM)
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
│   ├── ae400.yaml            # LIPSedge AE400 (PoE) camera config
│   ├── d435i.yaml            # D435i camera config
│   ├── default.yaml          # Default config
│   └── jetson.yaml           # Jetson edge config
├── src/
│   ├── camera/               # RealSense, AE400 + dummy camera drivers
│   ├── detector/             # YOLO + depth detectors
│   ├── geometry/             # RANSAC plane + cuboid reconstruction
│   ├── pointcloud/           # Open3D point cloud pipeline
│   ├── tracking/             # ByteTrack + Kalman filter
│   ├── pipeline/             # End-to-end pipeline orchestrator
│   └── api/                  # FastAPI REST endpoints
├── scripts/
│   ├── camera_preview.py     # Live RGB + depth preview for any camera
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
python main.py --config config/ae400.yaml   # or config/d435i.yaml
# --no-vis  disable the OpenCV window   --no-api  disable the REST API   --device cpu  run YOLO on CPU
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
