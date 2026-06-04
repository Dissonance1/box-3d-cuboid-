"""
Train YOLO11n-seg on the Carton Box segmentation dataset.

Dataset: 4209 train / 737 val / 316 test  (1 class: box)
Model:   yolo11n-seg  (nano — fastest, good for real-time on RTX GPU)

Expected training time:
  RTX 3080/4070:  ~1.5–2.5 hours for 100 epochs
  RTX 3060:       ~2.5–4 hours

Output:
  runs/train/carton-seg/weights/best.pt   ← use this in production
  runs/train/carton-seg/weights/last.pt

After training, update config/d435i.yaml:
  detector:
    model_path: models/carton_seg_best.pt
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

DATA_YAML   = Path("datasets/carton_seg/data.yaml")
BASE_MODEL  = "yolo11n-seg.pt"          # pretrained COCO weights (auto-download)
OUTPUT_DIR  = Path("runs/train")
RUN_NAME    = "carton-seg"
FINAL_MODEL = Path("models/carton_seg_best.pt")


def main() -> None:
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run first:  python scripts/prepare_seg_dataset.py")
        return

    print("=" * 60)
    print("YOLO11n-Seg  |  Carton Box  |  1 class")
    print("=" * 60)

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML.resolve()),
        task="segment",
        epochs=100,
        imgsz=640,
        batch=32,                  # RTX 5070 Ti has 12 GB VRAM — batch 32 is safe and faster
        device="0",                # first GPU
        workers=2,          # reduced from 4 — prevents RAM exhaustion on 16 GB systems
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,

        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        momentum=0.937,

        # Transfer learning — freeze backbone for first 10 epochs
        freeze=10,

        # Augmentation — tuned for warehouse/logistics box images
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,               # boxes rarely tilt >5° on conveyor
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0003,
        flipud=0.0,                # boxes are always upright
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,
        close_mosaic=10,

        # Segmentation-specific
        overlap_mask=True,
        mask_ratio=4,
        retina_masks=True,         # full-resolution masks for depth accuracy

        # Training control
        amp=True,                  # automatic mixed precision (FP16)
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        FINAL_MODEL.parent.mkdir(exist_ok=True)
        shutil.copy2(best_weights, FINAL_MODEL)
        print(f"\nBest model saved to: {FINAL_MODEL}")
        print(f"\nUpdate config/d435i.yaml:")
        print(f"  detector:")
        print(f"    model_path: {FINAL_MODEL}")
    else:
        print("WARNING: best.pt not found — check training logs")

    print("\nDone. Evaluate with:")
    print(f"  python -c \"from ultralytics import YOLO; YOLO('{FINAL_MODEL}').val(data='{DATA_YAML}')\"")


if __name__ == "__main__":
    main()
