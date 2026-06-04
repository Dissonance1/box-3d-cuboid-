"""Prometheus metrics collection for production monitoring.

All metrics are registered once at import time. Each pipeline stage
updates its relevant gauge/histogram via the singleton MetricsCollector.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

_REGISTRY = CollectorRegistry(auto_describe=True)

# Pipeline throughput
FRAMES_PROCESSED = Counter(
    "box_pose_frames_processed_total",
    "Total frames processed by the pipeline",
    registry=_REGISTRY,
)
FRAMES_DROPPED = Counter(
    "box_pose_frames_dropped_total",
    "Frames dropped due to processing overload",
    registry=_REGISTRY,
)
DETECTIONS_TOTAL = Counter(
    "box_pose_detections_total",
    "Total box detections across all frames",
    registry=_REGISTRY,
)

# Latency histograms (seconds)
FRAME_LATENCY = Histogram(
    "box_pose_frame_latency_seconds",
    "End-to-end per-frame processing time",
    buckets=[0.010, 0.020, 0.033, 0.050, 0.067, 0.100, 0.200, 0.500],
    registry=_REGISTRY,
)
DETECTION_LATENCY = Histogram(
    "box_pose_detection_latency_seconds",
    "YOLO inference time",
    buckets=[0.005, 0.010, 0.020, 0.033, 0.050, 0.100],
    registry=_REGISTRY,
)
POINTCLOUD_LATENCY = Histogram(
    "box_pose_pointcloud_latency_seconds",
    "Point cloud generation + filtering time",
    buckets=[0.001, 0.005, 0.010, 0.020, 0.050],
    registry=_REGISTRY,
)
GEOMETRY_LATENCY = Histogram(
    "box_pose_geometry_latency_seconds",
    "RANSAC plane detection + cuboid reconstruction time",
    buckets=[0.001, 0.005, 0.010, 0.020, 0.050],
    registry=_REGISTRY,
)

# System state gauges
CURRENT_FPS = Gauge(
    "box_pose_fps",
    "Current pipeline frames-per-second",
    registry=_REGISTRY,
)
ACTIVE_TRACKS = Gauge(
    "box_pose_active_tracks",
    "Number of confirmed tracked objects",
    registry=_REGISTRY,
)
CONFIDENCE_MEAN = Gauge(
    "box_pose_confidence_mean",
    "Mean detection confidence across active tracks",
    registry=_REGISTRY,
)
CAMERA_CONNECTED = Gauge(
    "box_pose_camera_connected",
    "1 if camera is streaming, 0 if disconnected",
    registry=_REGISTRY,
)
DEPTH_COVERAGE_MEAN = Gauge(
    "box_pose_depth_coverage_mean",
    "Mean depth coverage ratio across detections",
    registry=_REGISTRY,
)

# Error counters
CAMERA_ERRORS = Counter(
    "box_pose_camera_errors_total",
    "Camera hardware/stream errors",
    registry=_REGISTRY,
)
DETECTION_ERRORS = Counter(
    "box_pose_detection_errors_total",
    "Model inference errors",
    registry=_REGISTRY,
)
GEOMETRY_ERRORS = Counter(
    "box_pose_geometry_errors_total",
    "Plane detection / cuboid reconstruction failures",
    registry=_REGISTRY,
)


class _FPSTracker:
    """Rolling window FPS calculator."""

    def __init__(self, window: int = 30) -> None:
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - 2.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            n = len(self._timestamps)
            fps = (n - 1) / (self._timestamps[-1] - self._timestamps[0]) if n > 1 else 0.0
        CURRENT_FPS.set(fps)
        return fps


_fps_tracker = _FPSTracker()


def tick_fps() -> float:
    return _fps_tracker.tick()


def start_prometheus_server(port: int = 9090) -> None:
    start_http_server(port, registry=_REGISTRY)
