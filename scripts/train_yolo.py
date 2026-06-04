"""
YOLO11-Seg training script for box/carton/crate/package segmentation.

Dataset structure (YOLO segmentation format):
  datasets/
    box-seg/
      images/
        train/   *.jpg or *.png
        val/
        test/
      labels/
        train/   *.txt  (one per image)
        val/
        test/
      data.yaml

Label format (YOLO segmentation, one instance per line):
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  Coordinates normalized [0,1]. Polygon vertices in clockwise order.
  Use at least 8 polygon points for accurate mask boundaries.
  Example: 0 0.5 0.3 0.6 0.3 0.6 0.5 0.5 0.5  (class=box, 4 polygon vertices)

Recommended dataset size:
  Minimum:   3,000 images per class (12,000 total)
  Good:     10,000 images per class
  Production: 30,000+ images with challenging cases

Augmentation pipeline (configured in YOLO training):
  - Random horizontal flip
  - Mosaic (4-image composite, disabled last 10 epochs)
  - MixUp (probability 0.1)
  - HSV color jitter (hue±15, saturation±70, value±40)
  - Scale ±50%
  - Rotation ±10°
  - Translation ±10%
  - Perspective distortion 0.0005
  - Copy-paste augmentation 0.1
  Note: DO NOT use rotation augmentation >45° for box detection
  since extreme rotations create ambiguous mask boundaries.

Transfer learning strategy:
  1. Start from COCO-pretrained yolo11n-seg.pt or yolo11m-seg.pt
  2. Freeze backbone for first 10 epochs (freeze=10)
  3. Fine-tune all layers for remaining epochs
  4. Use lower lr0 for fine-tuning (1e-3 vs 1e-2 default)

Usage:
  python scripts/train_yolo.py --model yolo11n-seg --epochs 100 --data datasets/box-seg/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


DATA_YAML_TEMPLATE = """
path: {dataset_root}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: box
  1: carton
  2: crate
  3: package
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11-Seg for box detection")
    parser.add_argument("--model", default="yolo11n-seg", choices=[
        "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"
    ])
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="box-seg")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N backbone layers")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Regularization
        dropout=0.0,
        weight_decay=0.0005,
        # Learning rate schedule
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Training control
        close_mosaic=10,   # disable mosaic last 10 epochs
        amp=True,          # automatic mixed precision
        val=True,
        save=True,
        # Freeze backbone
        freeze=args.freeze,
        # Task-specific
        task="segment",
        overlap_mask=True,  # handle overlapping box instances
        mask_ratio=4,       # mask downsample ratio
        retina_masks=True,  # full-resolution masks (better accuracy)
    )

    print(f"\nTraining complete. Best model: {results.save_dir}/weights/best.pt")
    print("Copy best.pt to models/ and update config.detector.model_path")


if __name__ == "__main__":
    main()
