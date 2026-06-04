"""
Train YOLO11-Pose to detect 4 box corners as keypoints.

Requires annotated dataset in:
    datasets/carton_corners/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

Usage:
    python scripts/train_corners.py
"""

import os, shutil
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path("datasets/carton_corners")

# ── Create data.yaml ──────────────────────────────────────────────────
def make_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    content = f"""# Carton corner keypoint dataset
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names: [box]

# 4 keypoints: TL, TR, BR, BL
kpt_shape: [4, 3]   # 4 keypoints, each with (x, y, visibility)
"""
    yaml_path.write_text(content)
    return yaml_path

# ── Split train/val if only train exists ─────────────────────────────
def ensure_val_split(ratio=0.15):
    import random, shutil
    train_imgs = list((DATASET_DIR/"images/train").glob("*.jpg"))
    val_dir_i  = DATASET_DIR/"images/val"
    val_dir_l  = DATASET_DIR/"labels/val"
    if val_dir_i.exists() and len(list(val_dir_i.glob("*.jpg"))) > 0:
        return   # already split
    val_dir_i.mkdir(parents=True, exist_ok=True)
    val_dir_l.mkdir(parents=True, exist_ok=True)

    n_val = max(5, int(len(train_imgs) * ratio))
    val   = random.sample(train_imgs, n_val)
    for img in val:
        lbl = DATASET_DIR/"labels/train"/(img.stem + ".txt")
        shutil.move(str(img), str(val_dir_i/img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(val_dir_l/lbl.name))
    print(f"Val split: moved {n_val} images to val/")


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset not found at {DATASET_DIR}")
        print("Run: python scripts/collect_training_data.py")
        print("Then: python scripts/annotate_corners.py")
        return

    n_train = len(list((DATASET_DIR/"images/train").glob("*.jpg")))
    print(f"Training images: {n_train}")
    if n_train < 10:
        print("Need at least 10 images. Collect more data first.")
        return

    ensure_val_split()
    yaml_path = make_data_yaml()
    print(f"data.yaml: {yaml_path}")

    # ── Train ─────────────────────────────────────────────────────────
    model = YOLO("yolo11n-pose.pt")   # nano pose model — fast

    results = model.train(
        data=str(yaml_path),
        task="pose",
        epochs=100,
        imgsz=640,
        batch=16,           # reduce to 8 if GPU runs out of memory
        device="0",
        workers=2,
        project="runs/train",
        name="carton_corners",
        exist_ok=True,

        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        weight_decay=0.0005,

        # Augmentation for boxes at all angles
        degrees=30.0,       # random rotation ±30°
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,

        # Pose-specific
        kobj=1.0,           # keypoint objectness weight
        pose=12.0,          # keypoint regression weight

        amp=True,
        val=True,
        verbose=True,
    )

    best = Path("runs/train/carton_corners/weights/best.pt")
    if best.exists():
        dest = Path("models/carton_corners_best.pt")
        shutil.copy2(best, dest)
        print(f"\nModel saved to: {dest}")
        print("Use this model in live_view.py")
    else:
        print("Training finished. Check runs/train/carton_corners/weights/")


if __name__ == "__main__":
    main()
