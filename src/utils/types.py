"""Core data types for the box pose estimation system.

All inter-module data is passed via these typed dataclasses to ensure
clean interfaces and enable static analysis throughout the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()
    REMOVED = auto()


class DetectionClass(Enum):
    BOX = "box"
    CARTON = "carton"
    CRATE = "crate"
    PACKAGE = "package"


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 0.001  # millimeters -> meters
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass
class CameraFrame:
    color: np.ndarray           # HxWx3 uint8, BGR
    depth: np.ndarray           # HxW float32, meters
    timestamp_ns: int           # nanoseconds since epoch
    frame_number: int
    intrinsics: CameraIntrinsics
    depth_confidence: np.ndarray | None = None  # HxW float32 0-1

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1e9

    @property
    def height(self) -> int:
        return self.color.shape[0]

    @property
    def width(self) -> int:
        return self.color.shape[1]


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray        # [x1, y1, x2, y2] in pixels
    mask: np.ndarray             # HxW binary uint8
    mask_score: float = 0.0


@dataclass
class DepthROI:
    mask: np.ndarray             # HxW binary mask for the detection
    depth_map: np.ndarray        # HxW float32 meters, invalid=0
    valid_ratio: float           # fraction of mask pixels with valid depth
    mean_depth: float            # meters
    min_depth: float
    max_depth: float


@dataclass
class PointCloud:
    points: np.ndarray           # Nx3 float64
    colors: np.ndarray           # Nx3 float32 0-1
    normals: np.ndarray | None = None  # Nx3
    source_mask: np.ndarray | None = None  # original pixel indices

    @property
    def size(self) -> int:
        return len(self.points)


@dataclass
class PlaneModel:
    normal: np.ndarray           # unit vector (3,)
    offset: float                # d in ax+by+cz+d=0
    inlier_count: int
    inlier_ratio: float
    fitness: float               # fraction of cluster points on plane
    rmse: float                  # RMS distance of inliers
    centroid: np.ndarray         # (3,) centroid of inlier points

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from points to plane."""
        return points @ self.normal + self.offset

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project point onto plane."""
        d = np.dot(self.normal, point) + self.offset
        return point - d * self.normal


@dataclass
class CuboidGeometry:
    corners: np.ndarray          # 8x3 float64
    faces: list[PlaneModel]      # up to 6 planes
    center: np.ndarray           # (3,)
    rotation_matrix: np.ndarray  # 3x3
    dimensions: np.ndarray       # [L, W, H] meters
    reconstruction_quality: float  # 0-1

    @property
    def length(self) -> float:
        return float(self.dimensions[0])

    @property
    def width(self) -> float:
        return float(self.dimensions[1])

    @property
    def height(self) -> float:
        return float(self.dimensions[2])


@dataclass
class PoseEstimate:
    position: np.ndarray         # [x, y, z] meters in camera frame
    rotation_matrix: np.ndarray  # 3x3
    quaternion: np.ndarray       # [qw, qx, qy, qz]
    euler_deg: np.ndarray        # [roll, pitch, yaw] degrees
    position_world: np.ndarray | None = None  # in world frame
    rotation_world: np.ndarray | None = None


@dataclass
class DimensionEstimate:
    length_m: float
    width_m: float
    height_m: float
    uncertainty_m: np.ndarray    # [dL, dW, dH] one-sigma
    confidence: float

    def to_mm(self) -> tuple[float, float, float]:
        return self.length_m * 1000, self.width_m * 1000, self.height_m * 1000

    def to_cm(self) -> tuple[float, float, float]:
        return self.length_m * 100, self.width_m * 100, self.height_m * 100


@dataclass
class ConfidenceBreakdown:
    segmentation: float          # YOLO mask confidence
    depth_completeness: float    # valid depth pixel ratio
    point_density: float         # points per cubic cm
    plane_quality: float         # RANSAC plane fitness
    cuboid_quality: float        # reconstruction quality
    tracking_stability: float    # Kalman innovation ratio
    overall: float               # weighted composite score


@dataclass
class BoxMeasurement:
    """Primary output object from the full pipeline for one detected box."""
    track_id: int
    timestamp_ns: int
    detection: DetectionResult
    depth_roi: DepthROI
    point_cloud: PointCloud
    cuboid: CuboidGeometry
    pose: PoseEstimate
    dimensions: DimensionEstimate
    confidence: ConfidenceBreakdown
    processing_time_ms: float    # end-to-end latency for this object

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1e9

    def to_dict(self) -> dict[str, Any]:
        p = self.pose.euler_deg
        d = self.dimensions
        pos = self.pose.position
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp_s,
            "position": {"x": round(float(pos[0]), 4),
                         "y": round(float(pos[1]), 4),
                         "z": round(float(pos[2]), 4)},
            "pose": {"roll": round(float(p[0]), 2),
                     "pitch": round(float(p[1]), 2),
                     "yaw": round(float(p[2]), 2)},
            "dimensions": {"length": round(d.length_m, 4),
                           "width": round(d.width_m, 4),
                           "height": round(d.height_m, 4)},
            "confidence": round(self.confidence.overall, 3),
            "class": self.detection.class_name,
        }


@dataclass
class FrameResult:
    """All measurements from one camera frame."""
    frame_number: int
    timestamp_ns: int
    measurements: list[BoxMeasurement]
    pipeline_time_ms: float
    fps: float

    @property
    def count(self) -> int:
        return len(self.measurements)
