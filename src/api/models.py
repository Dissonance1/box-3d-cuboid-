"""Pydantic v2 response models for the REST API.

All floating-point values are rounded to meaningful precision for wire transfer.
Internal numpy arrays are converted to plain Python types during serialization.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Position3D(BaseModel):
    x: float = Field(description="X position in meters (camera frame: right)")
    y: float = Field(description="Y position in meters (camera frame: down)")
    z: float = Field(description="Z position in meters (camera frame: forward / depth)")


class PoseAngles(BaseModel):
    roll: float = Field(description="Roll angle in degrees")
    pitch: float = Field(description="Pitch angle in degrees")
    yaw: float = Field(description="Yaw angle in degrees")


class Quaternion(BaseModel):
    qw: float
    qx: float
    qy: float
    qz: float


class Dimensions3D(BaseModel):
    length: float = Field(description="Longest dimension in meters")
    width: float = Field(description="Second dimension in meters")
    height: float = Field(description="Height (vertical) in meters")
    length_mm: float = Field(description="Length in millimeters")
    width_mm: float = Field(description="Width in millimeters")
    height_mm: float = Field(description="Height in millimeters")
    uncertainty_mm: list[float] = Field(description="[dL, dW, dH] one-sigma uncertainty in mm")


class ConfidenceDetail(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    segmentation: float = Field(ge=0.0, le=1.0)
    depth_completeness: float = Field(ge=0.0, le=1.0)
    point_density: float = Field(ge=0.0, le=1.0)
    plane_quality: float = Field(ge=0.0, le=1.0)
    cuboid_quality: float = Field(ge=0.0, le=1.0)
    tracking_stability: float = Field(ge=0.0, le=1.0)


class BoxDetection(BaseModel):
    track_id: int
    timestamp: float = Field(description="Unix timestamp seconds")
    class_name: str
    position: Position3D
    pose: PoseAngles
    quaternion: Quaternion
    dimensions: Dimensions3D
    confidence: ConfidenceDetail
    processing_time_ms: float


class FrameDetections(BaseModel):
    frame_number: int
    timestamp: float
    fps: float
    count: int
    detections: list[BoxDetection]


class CameraInfo(BaseModel):
    connected: bool
    type: str
    width: int
    height: int
    fps: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float


class SystemMetrics(BaseModel):
    fps: float
    frame_count: int
    active_tracks: int
    pipeline_latency_ms: float
    detection_latency_ms: float
    geometry_latency_ms: float
    camera_connected: bool
    uptime_s: float


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded | unhealthy")
    camera: bool
    detector: bool
    pipeline_running: bool
    fps: float
    message: Optional[str] = None
