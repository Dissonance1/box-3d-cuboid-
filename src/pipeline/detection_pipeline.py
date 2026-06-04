"""End-to-end detection pipeline orchestrator.

Executes each stage in order for every camera frame and returns
a FrameResult. All per-object failures are caught and logged without
dropping the entire frame — robust partial success is preferred over
all-or-nothing failure in a production system.

Pipeline per frame:
  Camera → Detector → (per detection):
    MaskProcessor → DepthProcessor → PointCloudGenerator →
    MultiPlaneDetector → CuboidReconstructor → PoseEstimator →
    DimensionEstimate → ConfidenceEstimator
  → TrackManager (all detections together)
  → FrameResult

Performance targets:
  Full pipeline at 30 FPS requires <33ms per frame.
  YOLO11n-seg on 640x360 with TensorRT typically takes 8-15ms.
  Point cloud + geometry per object takes 2-5ms.
  Budget: detection 15ms + 3 objects × 5ms geometry = 30ms total.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.camera.base_camera import BaseCamera
from src.depth.depth_processor import DepthProcessor
from src.detector.base_detector import BaseDetector
from src.detector.depth_detector import DepthDetector
from src.filtering.confidence_estimator import ConfidenceEstimator
from src.geometry.cuboid_reconstructor import CuboidReconstructor
from src.geometry.plane_detector import MultiPlaneDetector
from src.pointcloud.cloud_generator import PointCloudGenerator
from src.pose.pose_estimator import PoseEstimator
from src.segmentation.mask_processor import MaskProcessor
from src.tracking.track_manager import TrackManager
from src.utils.config import SystemConfig
from src.utils.exceptions import (
    BoxPoseSystemError,
    CuboidReconstructionError,
    InsufficientDepthError,
    InsufficientPointsError,
    PlaneDetectionError,
)
from src.utils.logger import get_logger
from src.utils.metrics import (
    DETECTIONS_TOTAL,
    FRAMES_DROPPED,
    FRAMES_PROCESSED,
    FRAME_LATENCY,
    tick_fps,
)
from src.utils.types import (
    BoxMeasurement,
    CuboidGeometry,
    DimensionEstimate,
    FrameResult,
)

logger = get_logger(__name__)

_FRAME_BUDGET_S = 0.060  # 60ms = allow 16+ FPS even under load


class DetectionPipeline:
    """
    Orchestrates the complete box detection and measurement pipeline.

    Usage:
        pipeline = DetectionPipeline(config, camera, detector)
        for frame_result in pipeline.run():
            ...  # publish to API, visualization, etc.
    """

    def __init__(
        self,
        config: SystemConfig,
        camera: BaseCamera,
        detector: BaseDetector,
    ) -> None:
        self._config = config
        self._camera = camera
        self._detector = detector

        # Depth-based fallback detector: used when YOLO finds no box-class objects.
        # This handles the common case where the COCO-pretrained model does not
        # recognise plain cardboard boxes (no "box" class in COCO).
        self._depth_detector = DepthDetector(config.detector)
        self._depth_detector.load()

        self._mask_processor = MaskProcessor(config.depth)
        self._depth_processor = DepthProcessor(config.depth)
        self._pc_generator = PointCloudGenerator(config.pointcloud)
        self._plane_detector = MultiPlaneDetector(config.plane_detection)
        self._cuboid_reconstructor = CuboidReconstructor(config.plane_detection)
        self._pose_estimator = PoseEstimator(config)
        self._confidence_estimator = ConfidenceEstimator()
        self._track_manager = TrackManager(config)

        self._running = False
        self._fps_window: list[float] = []

    def run(self):
        """Generator that yields FrameResult for each processed frame."""
        self._running = True
        logger.info("Pipeline started")

        for frame in self._camera.stream():
            if not self._running:
                break

            t_frame_start = time.perf_counter()
            result = self._process_frame(frame)
            elapsed = time.perf_counter() - t_frame_start

            FRAME_LATENCY.observe(elapsed)
            fps = tick_fps()

            if result is not None:
                result.fps = fps
                result.pipeline_time_ms = elapsed * 1000
                FRAMES_PROCESSED.inc()
                yield result
            else:
                FRAMES_DROPPED.inc()

        logger.info("Pipeline stopped")

    def _process_frame(self, frame) -> Optional[FrameResult]:
        """Process one camera frame. Returns None on fatal frame error."""
        try:
            t0 = time.perf_counter()

            # Stage 1a: YOLO detection
            detections = self._detector.detect(frame.color)
            detection_ms = (time.perf_counter() - t0) * 1000

            # Stage 1b: Depth-based fallback when YOLO finds nothing.
            # YOLO11n-seg is COCO-trained — plain cardboard boxes are not a
            # COCO class and will not be detected. The depth detector segments
            # box-shaped objects purely from the depth frame geometry.
            if not detections:
                intr = frame.intrinsics
                self._depth_detector.set_intrinsics(
                    intr.fx, intr.fy, intr.cx, intr.cy
                )
                detections = self._depth_detector.detect_from_depth(
                    frame.color,
                    frame.depth,
                    min_depth=self._config.depth.min_valid_depth_m,
                    max_depth=self._config.depth.max_valid_depth_m,
                )
                if detections:
                    logger.debug(
                        "Depth detector fallback active",
                        objects=len(detections),
                    )

            if not detections:
                return FrameResult(
                    frame_number=frame.frame_number,
                    timestamp_ns=frame.timestamp_ns,
                    measurements=[],
                    pipeline_time_ms=0.0,
                    fps=0.0,
                )

            DETECTIONS_TOTAL.inc(len(detections))

            # Stage 2: Per-detection processing
            raw_measurements: list[BoxMeasurement] = []
            for detection in detections:
                m = self._process_detection(frame, detection)
                if m is not None:
                    raw_measurements.append(m)

            if not raw_measurements:
                return FrameResult(
                    frame_number=frame.frame_number,
                    timestamp_ns=frame.timestamp_ns,
                    measurements=[],
                    pipeline_time_ms=0.0,
                    fps=0.0,
                )

            # Stage 3: Multi-object tracking + temporal stabilization
            tracked = self._track_manager.update(raw_measurements)

            return FrameResult(
                frame_number=frame.frame_number,
                timestamp_ns=frame.timestamp_ns,
                measurements=tracked,
                pipeline_time_ms=0.0,
                fps=0.0,
            )

        except BoxPoseSystemError as e:
            logger.warning("Pipeline frame error", error=str(e))
            return None
        except Exception as e:
            logger.error("Unexpected pipeline error", exc_info=True)
            return None

    def _process_detection(self, frame, detection) -> Optional[BoxMeasurement]:
        """
        Run geometry pipeline for a single detection.
        Returns BoxMeasurement or None on non-fatal failure.
        """
        t0 = time.perf_counter()
        try:
            # Stage 2a: Mask refinement
            refined_mask = self._mask_processor.process(detection)

            # Stage 2b: Depth extraction
            depth_roi = self._depth_processor.extract_roi(frame, detection, refined_mask)

            # Stage 2c: Point cloud generation + filtering
            point_cloud = self._pc_generator.generate(frame, depth_roi)

            # Stage 2d: Plane detection
            planes = self._plane_detector.detect(point_cloud)

            # Stage 2e: Cuboid reconstruction
            cuboid = self._cuboid_reconstructor.reconstruct(planes, point_cloud)

            # Stage 2e.1: Overhead height override
            # When camera height is configured, replace synthesized height with
            # the direct measurement: box_height = camera_height - top_face_depth
            cam_h = self._config.camera_height_above_floor_m
            if cam_h > 0.0:
                direct_height = self._depth_processor.estimate_box_height_overhead(
                    depth_roi, cam_h
                )
                if direct_height is not None:
                    dims = cuboid.dimensions.copy()
                    dims[2] = direct_height   # dims[2] = height (vertical)
                    cuboid.dimensions = dims

            # Stage 2f: Pose estimation
            pose = self._pose_estimator.estimate(cuboid)

            # Stage 2g: Dimension estimation
            dims = self._make_dimension_estimate(cuboid)

            # Stage 2h: Confidence
            confidence = self._confidence_estimator.compute(
                detection=detection,
                depth_roi=depth_roi,
                point_cloud=point_cloud,
                cuboid=cuboid,
                tracking_stability=0.5,  # updated by TrackManager
            )

            proc_ms = (time.perf_counter() - t0) * 1000

            return BoxMeasurement(
                track_id=-1,  # assigned by TrackManager
                timestamp_ns=frame.timestamp_ns,
                detection=detection,
                depth_roi=depth_roi,
                point_cloud=point_cloud,
                cuboid=cuboid,
                pose=pose,
                dimensions=dims,
                confidence=confidence,
                processing_time_ms=proc_ms,
            )

        except InsufficientDepthError as e:
            logger.debug("Insufficient depth", coverage=e.coverage, min=e.min_required)
            return None
        except InsufficientPointsError as e:
            logger.debug("Insufficient points", count=e.count, min=e.min_required)
            return None
        except PlaneDetectionError as e:
            logger.debug("Plane detection failed", error=str(e))
            return None
        except CuboidReconstructionError as e:
            logger.debug("Cuboid reconstruction failed", planes=e.planes_found, error=str(e))
            return None
        except Exception as e:
            logger.warning("Object processing error", exc_info=True)
            return None

    @staticmethod
    def _make_dimension_estimate(cuboid: CuboidGeometry) -> DimensionEstimate:
        """
        Build DimensionEstimate from cuboid geometry.

        Uncertainty estimation:
          - Observed faces (inlier_count > 0): low uncertainty based on RANSAC RMSE
          - Synthesized faces: higher uncertainty (no direct depth data)
        """
        observed_rmse = np.mean([
            p.rmse for p in cuboid.faces if p.inlier_count > 0
        ]) if any(p.inlier_count > 0 for p in cuboid.faces) else 0.020

        base_uncertainty_m = max(0.003, observed_rmse * 1.5)
        uncertainty = np.array([base_uncertainty_m] * 3)

        # Synthesized planes have 3× higher uncertainty
        for i, p in enumerate(cuboid.faces[:3]):
            if p.inlier_count == 0:
                uncertainty[i] = base_uncertainty_m * 3.0

        d = cuboid.dimensions
        dim_confidence = max(0.1, cuboid.reconstruction_quality)

        return DimensionEstimate(
            length_m=float(d[0]),
            width_m=float(d[1]),
            height_m=float(d[2]),
            uncertainty_m=uncertainty,
            confidence=dim_confidence,
        )

    def stop(self) -> None:
        self._running = False
        self._camera.stop()
