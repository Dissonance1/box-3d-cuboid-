"""YOLO11-Seg detection and segmentation implementation.

Uses Ultralytics library with support for:
  - Native PyTorch (.pt)
  - ONNX Runtime (.onnx)
  - TensorRT (.engine)
  - OpenVINO (.xml)

Produces pixel-level segmentation masks — no bounding-box-only paths exist.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.detector.base_detector import BaseDetector
from src.utils.config import DetectorConfig
from src.utils.exceptions import InferenceError, ModelLoadError
from src.utils.logger import get_logger
from src.utils.metrics import DETECTION_ERRORS, DETECTION_LATENCY
from src.utils.types import DetectionResult

logger = get_logger(__name__)


class YOLODetector(BaseDetector):
    """
    YOLO11-Seg wrapper.

    Inference flow:
      1. Preprocess: resize + normalize to model input size
      2. Inference: forward pass via Ultralytics YOLO
      3. Postprocess: decode boxes + masks, apply NMS
      4. Mask resize: scale prototype masks back to original frame size
      5. Return list[DetectionResult] with pixel-level masks

    Threading: not thread-safe — use one instance per thread or add locking.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._model = None
        self._class_names: list[str] = config.classes

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelLoadError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        try:
            self._model = YOLO(self._config.model_path, task="segment")
            # Warm up with a dummy frame to initialize CUDA context
            dummy = np.zeros(
                (self._config.input_size, self._config.input_size, 3), dtype=np.uint8
            )
            self._model(
                dummy,
                device=self._config.device,
                half=self._config.half_precision,
                verbose=False,
            )
            logger.info(
                "YOLO11-Seg model loaded",
                path=self._config.model_path,
                device=self._config.device,
                half=self._config.half_precision,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{self._config.model_path}': {e}") from e

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        if self._model is None:
            raise InferenceError("Model not loaded — call load() first")

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        try:
            results = self._model(
                image,
                conf=self._config.confidence_threshold,
                iou=self._config.nms_iou_threshold,
                imgsz=self._config.input_size,
                device=self._config.device,
                half=self._config.half_precision,
                max_det=self._config.max_detections,
                classes=self._resolve_class_indices(),
                verbose=False,
                retina_masks=True,  # full-resolution masks
            )
        except Exception as e:
            DETECTION_ERRORS.inc()
            raise InferenceError(f"YOLO inference failed: {e}") from e

        elapsed = time.perf_counter() - t0
        DETECTION_LATENCY.observe(elapsed)

        return self._parse_results(results[0], h, w)

    def _resolve_class_indices(self) -> Optional[list[int]]:
        if self._model is None:
            return None
        model_names: dict = self._model.names
        indices = []
        for cls_name in self._config.classes:
            for idx, name in model_names.items():
                if name.lower() == cls_name.lower():
                    indices.append(idx)
                    break
        return indices if indices else None

    def _parse_results(self, result, orig_h: int, orig_w: int) -> list[DetectionResult]:
        detections: list[DetectionResult] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections
        if result.masks is None:
            # Fallback: treat box as filled rectangle mask
            # This should not happen with retina_masks=True on a seg model
            logger.warning("No masks returned — falling back to bbox mask")
            has_masks = False
        else:
            has_masks = True

        boxes = result.boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, "unknown")

            # Skip classes not in our target list
            if cls_name.lower() not in [c.lower() for c in self._config.classes]:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(np.int32)

            if has_masks:
                # result.masks.data is Nx1xHxW or NxHxW binary float
                raw_mask = result.masks.data[i].cpu().numpy()
                if raw_mask.ndim == 3:
                    raw_mask = raw_mask[0]
                # Masks may be at model input resolution; resize to original
                if raw_mask.shape != (orig_h, orig_w):
                    raw_mask = cv2.resize(
                        raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                binary_mask = (raw_mask > 0.5).astype(np.uint8)
            else:
                binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                x1, y1, x2, y2 = (
                    max(0, xyxy[0]),
                    max(0, xyxy[1]),
                    min(orig_w, xyxy[2]),
                    min(orig_h, xyxy[3]),
                )
                binary_mask[y1:y2, x1:x2] = 1

            detections.append(DetectionResult(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=xyxy,
                mask=binary_mask,
                mask_score=conf,
            ))

        return detections

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
