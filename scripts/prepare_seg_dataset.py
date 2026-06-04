"""
Convert YOLO detection labels (class cx cy w h) to YOLO segmentation
polygon labels (class x1 y1 x2 y2 x3 y3 x4 y4) — 4-corner rectangle.

This is the correct approach for training YOLO11-seg when you only have
bounding box annotations. Each box becomes a tight rectangular polygon.
The result is near-identical segmentation quality for rectangular objects
like cartons.

Usage:
    python scripts/prepare_seg_dataset.py
"""

import os
import glob
import shutil
from pathlib import Path

DATASET_SRC  = Path("Carton Box.yolov11")
DATASET_DST  = Path("datasets/carton_seg")


def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> str:
    """Convert normalized cx,cy,w,h to 4-corner polygon string."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy - h / 2
    x3 = cx + w / 2
    y3 = cy + h / 2
    x4 = cx - w / 2
    y4 = cy + h / 2
    # Clamp to [0,1]
    coords = [x1,y1, x2,y2, x3,y3, x4,y4]
    coords = [max(0.0, min(1.0, v)) for v in coords]
    return " ".join(f"{v:.6f}" for v in coords)


def convert_split(split: str) -> int:
    src_lbl = DATASET_SRC / split / "labels"
    src_img = DATASET_SRC / split / "images"

    split_out = "valid" if split == "valid" else split
    dst_lbl = DATASET_DST / split_out / "labels"
    dst_img = DATASET_DST / split_out / "images"
    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    label_files = list(src_lbl.glob("*.txt"))
    converted = 0

    for lbl_path in label_files:
        lines_out = []
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                poly = bbox_to_polygon(float(cx), float(cy), float(w), float(h))
                lines_out.append(f"{cls} {poly}")
            elif len(parts) > 5:
                # Already polygon format — copy as-is
                lines_out.append(line)
        (dst_lbl / lbl_path.name).write_text("\n".join(lines_out))
        converted += 1

    # Copy images (symlink would be faster but copy is portable)
    for img_path in src_img.glob("*"):
        dst = dst_img / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)

    return converted


def write_data_yaml() -> Path:
    yaml_path = DATASET_DST / "data.yaml"
    abs_base = DATASET_DST.resolve()
    content = f"""# Carton Box segmentation dataset
# Converted from detection bbox → 4-corner polygon by prepare_seg_dataset.py

path: {abs_base.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: box
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


if __name__ == "__main__":
    print("Converting detection labels -> segmentation polygons...")
    for split in ["train", "valid", "test"]:
        n = convert_split(split)
        print(f"  {split}: {n} files converted")

    yaml_path = write_data_yaml()
    print(f"\nDataset ready at: {DATASET_DST.resolve()}")
    print(f"data.yaml:        {yaml_path.resolve()}")
    print("\nNext: python scripts/train_carton.py")
