"""
Export YOLO11-Seg model to TensorRT engine for production deployment.

Must run on the TARGET device (Jetson or x86 GPU with TensorRT installed).
The engine is NOT portable between different GPU architectures.

Usage:
    python scripts/export_tensorrt.py \
        --model models/yolo11n-seg.pt \
        --output models/yolo11n-seg.engine \
        --size 640 \
        --batch 1 \
        --fp16

TensorRT FP16 latency targets (typical):
  Jetson Orin NX:   yolo11n-seg @ 640 → ~8ms/frame
  RTX 3080:         yolo11n-seg @ 640 → ~3ms/frame
  Jetson AGX Orin:  yolo11n-seg @ 640 → ~5ms/frame
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO11-Seg to TensorRT")
    parser.add_argument("--model", required=True, help="Input .pt model path")
    parser.add_argument("--output", default=None, help="Output .engine path")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace", type=int, default=4, help="Workspace GB")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    output_path = args.output or str(Path(args.model).with_suffix(".engine"))

    print(f"Exporting {args.model} → TensorRT engine")
    print(f"  Input size: {args.size}x{args.size}")
    print(f"  FP16: {args.fp16}")
    print(f"  Workspace: {args.workspace}GB")
    print("  This may take 10-30 minutes on first run...")

    path = model.export(
        format="engine",
        imgsz=args.size,
        half=args.fp16,
        device=0,
        batch=args.batch,
        workspace=args.workspace,
        verbose=True,
    )
    print(f"\nEngine exported to: {path}")
    print("Update config: detector.model_path:", path)
    print("Update config: detector.device: tensorrt")


if __name__ == "__main__":
    main()
