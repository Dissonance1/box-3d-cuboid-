# How to Annotate Box Corners for YOLO11-Pose

## What you are labeling

For each box in each image, you label **4 corner points** in this fixed order:

```
Corner 1 (TL)  ●───────────────● Corner 2 (TR)
               │                │
               │    (top face)  │
               │                │
Corner 4 (BL)  ●───────────────● Corner 3 (BR)
```

Always label corners **clockwise starting from top-left**.
Be consistent — if you mix orders the model learns noise.

---

## Option A: Roboflow (recommended, easiest)

1. Go to https://roboflow.com and create a free account
2. Create new project → Object Detection + Keypoints
3. Set keypoints to 4 (one per corner)
4. Upload your images from `datasets/carton_corners/images/train/`
5. Annotate:
   - Draw bounding box around the full carton
   - Place 4 keypoints at the corners in order: TL → TR → BR → BL
6. Export as **YOLO11 Pose** format
7. Download and extract to `datasets/carton_corners/`

---

## Option B: CVAT (free, self-hosted)

1. Install CVAT: https://docs.cvat.ai/docs/administration/basics/installation/
2. Create a task with "Pose Estimation" type
3. Define skeleton with 4 joints: TL, TR, BR, BL
4. Annotate all images
5. Export as YOLO format

---

## Option C: Label manually (for small datasets)

Use the provided `annotate_corners.py` script which lets you click
the 4 corners of each box directly.

---

## How many images do you need?

| Scenario | Images needed |
|---|---|
| One box type, controlled lighting | 100–200 |
| Multiple box sizes, normal lighting | 300–500 |
| Real warehouse (varied lighting, damage) | 500–1000+ |

**Data augmentation doubles your effective dataset** — Roboflow can
apply brightness, flip, rotation augmentations automatically.

---

## YOLO Keypoint Label Format

Each image gets a `.txt` file with the same name.
One line per box:

```
class_id  cx  cy  w  h  kp1x kp1y kp1v  kp2x kp2y kp2v  kp3x kp3y kp3v  kp4x kp4y kp4v
```

- All x/y values normalized [0, 1]
- `kpNv` = visibility: 2 = visible, 1 = occluded, 0 = not labeled
- `class_id` = 0 (just "box")

Example for a box filling half the image, tilted 30°:
```
0 0.50 0.50 0.60 0.45  0.22 0.28 2  0.78 0.22 2  0.78 0.72 2  0.22 0.72 2
```
