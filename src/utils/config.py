"""Configuration management using Pydantic v2 with YAML loading.

All system parameters are validated at startup with sensible defaults.
Override via YAML file, environment variables, or programmatic dict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class CameraConfig(BaseModel):
    type: Literal["realsense", "zed", "orbbec", "file", "dummy"] = "realsense"
    serial_number: Optional[str] = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth_mode: Literal["high_accuracy", "high_density", "medium_density"] = "high_accuracy"
    enable_emitter: bool = True
    laser_power: int = 150        # 0-360 mW
    depth_min_m: float = 0.3
    depth_max_m: float = 4.0
    align_depth_to_color: bool = True
    # Post-processing filters
    enable_decimation: bool = False
    decimation_magnitude: int = 2
    enable_spatial: bool = True
    spatial_magnitude: int = 2
    spatial_alpha: float = 0.5
    spatial_delta: int = 20
    enable_temporal: bool = True
    temporal_alpha: float = 0.4
    temporal_delta: int = 20
    enable_hole_filling: bool = True
    hole_filling_mode: int = 1    # 0=disabled, 1=2px, 2=4px, 3=8px, 4=16px, 5=unlimited
    frame_queue_size: int = 5
    warmup_frames: int = 30       # discard first N frames for auto-exposure


class DetectorConfig(BaseModel):
    model_path: str = "models/yolo11-seg.pt"
    device: Literal["cpu", "cuda", "tensorrt", "openvino"] = "cuda"
    confidence_threshold: float = 0.45
    nms_iou_threshold: float = 0.45
    input_size: int = 640
    half_precision: bool = True   # FP16 for GPU
    batch_size: int = 1
    classes: list[str] = Field(default_factory=lambda: ["box", "carton", "crate", "package"])
    max_detections: int = 32
    # ONNX Runtime settings
    onnx_providers: list[str] = Field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])


class DepthConfig(BaseModel):
    min_valid_depth_m: float = 0.2
    max_valid_depth_m: float = 5.0
    min_depth_coverage: float = 0.30  # min ratio of valid pixels in mask
    temporal_window: int = 5          # frames for temporal averaging
    gradient_threshold_m: float = 0.05  # edge noise rejection
    depth_dilation_px: int = 2        # erode mask before depth extraction to avoid edge noise


class PointCloudConfig(BaseModel):
    voxel_size_m: float = 0.005       # 5mm voxel downsampling
    sor_nb_neighbors: int = 30        # statistical outlier removal neighbors
    sor_std_ratio: float = 2.0
    ror_nb_points: int = 10           # radius outlier removal
    ror_radius_m: float = 0.015
    min_cluster_points: int = 100
    max_cluster_points: int = 50000
    dbscan_eps_m: float = 0.015
    dbscan_min_points: int = 10
    min_points_for_reconstruction: int = 200
    normal_estimation_radius_m: float = 0.02


class PlaneDetectionConfig(BaseModel):
    ransac_distance_threshold_m: float = 0.008  # 8mm inlier threshold
    ransac_n_iterations: int = 1000
    ransac_min_inlier_ratio: float = 0.15       # min fraction of points on plane
    max_planes: int = 6
    min_planes_for_cuboid: int = 2              # at minimum top + one side
    plane_angle_tolerance_deg: float = 15.0     # orthogonality tolerance
    plane_parallel_tolerance_deg: float = 10.0
    min_plane_area_m2: float = 0.005            # 50cm²


class TrackingConfig(BaseModel):
    high_score_threshold: float = 0.6
    low_score_threshold: float = 0.1
    match_iou_threshold: float = 0.3
    max_age: int = 30             # frames before track is removed
    min_hits: int = 3             # frames before track is confirmed
    # Kalman filter noise
    process_noise_pos: float = 0.01    # meters
    process_noise_dim: float = 0.005
    process_noise_angle: float = 2.0   # degrees
    measurement_noise_pos: float = 0.02
    measurement_noise_dim: float = 0.01
    measurement_noise_angle: float = 3.0


class TemporalFilterConfig(BaseModel):
    window_size: int = 5
    position_alpha: float = 0.6       # EMA alpha, higher = faster response
    dimension_alpha: float = 0.3      # dimensions change slowly
    angle_alpha: float = 0.4
    outlier_sigma: float = 3.0        # reject measurements > N sigma from mean


class VisualizationConfig(BaseModel):
    enabled: bool = True
    window_name: str = "Box Pose Estimation"
    show_masks: bool = True
    show_cuboid: bool = True
    show_axes: bool = True
    show_dimensions: bool = True
    show_confidence: bool = True
    show_fps: bool = True
    show_depth_colormap: bool = False
    axis_length_m: float = 0.1
    cuboid_line_thickness: int = 2
    font_scale: float = 0.5
    mask_alpha: float = 0.35
    save_frames: bool = False
    save_dir: str = "output/frames"


class APIConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_result_cache: int = 100  # keep last N frame results in memory


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    file: Optional[str] = None
    rotation_mb: int = 100
    retention_days: int = 7


class MetricsConfig(BaseModel):
    enabled: bool = True
    prometheus_port: int = 9090
    export_interval_s: float = 5.0


class SystemConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    pointcloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    plane_detection: PlaneDetectionConfig = Field(default_factory=PlaneDetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    temporal_filter: TemporalFilterConfig = Field(default_factory=TemporalFilterConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    # Camera-to-world extrinsic (rotation matrix + translation)
    # Identity by default (camera IS the world frame reference)
    camera_to_world_rotation: list[list[float]] = Field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    camera_to_world_translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_height_above_floor_m: float = 0.0   # [MUST SET for overhead] camera lens height above floor

    @field_validator("camera_to_world_rotation")
    @classmethod
    def validate_rotation(cls, v: list[list[float]]) -> list[list[float]]:
        arr = __import__("numpy").array(v)
        if arr.shape != (3, 3):
            raise ValueError("camera_to_world_rotation must be 3x3")
        det = __import__("numpy").linalg.det(arr)
        if not (0.99 < abs(det) < 1.01):
            raise ValueError(f"Rotation matrix determinant {det:.4f} not near ±1")
        return v


def load_config(config_path: str | Path | None = None) -> SystemConfig:
    """Load configuration from YAML file with environment variable overrides."""
    raw: dict = {}

    if config_path is None:
        # Search standard locations
        for candidate in [
            "config/default.yaml",
            "/etc/box-pose/config.yaml",
            Path.home() / ".config/box-pose/config.yaml",
        ]:
            p = Path(candidate)
            if p.exists():
                config_path = p
                break

    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p) as f:
            raw = yaml.safe_load(f) or {}

    # Allow env-var overrides for deployment flexibility
    env_overrides = _collect_env_overrides()
    _deep_merge(raw, env_overrides)

    return SystemConfig(**raw)


def _collect_env_overrides() -> dict:
    """Collect BOX_POSE__* environment variables into nested dict."""
    result: dict = {}
    prefix = "BOX_POSE__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split("__")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = _coerce_env_value(value)
    return result


def _coerce_env_value(v: str):
    if v.lower() in ("true", "yes", "1"):
        return True
    if v.lower() in ("false", "no", "0"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
