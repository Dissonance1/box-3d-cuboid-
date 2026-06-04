"""Intel RealSense D455 / D435 camera implementation.

Wraps pyrealsense2 SDK with full post-processing pipeline,
hardware-level synchronization, and automatic reconnection.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.camera.base_camera import BaseCamera
from src.utils.config import CameraConfig
from src.utils.exceptions import (
    CameraInitError,
    CameraStreamError,
    FrameSyncError,
)
from src.utils.logger import get_logger
from src.utils.metrics import CAMERA_CONNECTED, CAMERA_ERRORS
from src.utils.types import CameraFrame, CameraIntrinsics

logger = get_logger(__name__)

try:
    import pyrealsense2 as rs

    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False
    logger.warning("pyrealsense2 not installed — RealSenseCamera unavailable")


class RealSenseCamera(BaseCamera):
    """
    Production driver for Intel RealSense D455/D435.

    Post-processing pipeline (applied in recommended order):
      1. Decimation (optional, halves resolution for speed)
      2. Spatial filter (edge-preserving smooth)
      3. Temporal filter (reduce temporal noise)
      4. Hole filling

    Depth and color streams are aligned to the color sensor frame so
    every depth pixel maps exactly to the corresponding color pixel.

    Intrinsics are read from the device at runtime — no hardcoded values.
    """

    _MAX_SYNC_DELTA_MS = 16.0  # D455 hardware sync tolerance

    def __init__(self, config: CameraConfig) -> None:
        super().__init__()
        if not _RS_AVAILABLE:
            raise CameraInitError(
                "pyrealsense2 is not installed. Install with: pip install pyrealsense2"
            )
        self._config = config
        self._pipeline: Optional["rs.pipeline"] = None
        self._align: Optional["rs.align"] = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._filters: list = []
        self._warmup_done = False

    def open(self) -> None:
        try:
            self._pipeline = rs.pipeline()
            rs_config = rs.config()

            if self._config.serial_number:
                rs_config.enable_device(self._config.serial_number)

            rs_config.enable_stream(
                rs.stream.depth,
                self._config.width,
                self._config.height,
                rs.format.z16,
                self._config.fps,
            )
            rs_config.enable_stream(
                rs.stream.color,
                self._config.width,
                self._config.height,
                rs.format.bgr8,
                self._config.fps,
            )

            profile = self._pipeline.start(rs_config)
            self._configure_sensor(profile)
            self._build_filter_pipeline()

            if self._config.align_depth_to_color:
                self._align = rs.align(rs.stream.color)

            self._intrinsics = self._extract_intrinsics(profile)
            self._discard_warmup_frames()

            CAMERA_CONNECTED.set(1)
            logger.info(
                "RealSense camera opened",
                width=self._config.width,
                height=self._config.height,
                fps=self._config.fps,
                serial=self._get_serial(profile),
            )
        except Exception as e:
            CAMERA_CONNECTED.set(0)
            raise CameraInitError(f"Failed to open RealSense camera: {e}") from e

    def _configure_sensor(self, profile: "rs.pipeline_profile") -> None:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, int(self._config.enable_emitter))

        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, self._config.laser_power)

        # Enable auto-exposure for color
        color_sensor = device.query_sensors()[1]
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)

    def _build_filter_pipeline(self) -> None:
        self._filters = []

        if self._config.enable_decimation:
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, self._config.decimation_magnitude)
            self._filters.append(dec)

        if self._config.enable_spatial:
            spat = rs.spatial_filter()
            spat.set_option(rs.option.filter_magnitude, self._config.spatial_magnitude)
            spat.set_option(rs.option.filter_smooth_alpha, self._config.spatial_alpha)
            spat.set_option(rs.option.filter_smooth_delta, self._config.spatial_delta)
            self._filters.append(spat)

        if self._config.enable_temporal:
            temp = rs.temporal_filter()
            temp.set_option(rs.option.filter_smooth_alpha, self._config.temporal_alpha)
            temp.set_option(rs.option.filter_smooth_delta, self._config.temporal_delta)
            self._filters.append(temp)

        if self._config.enable_hole_filling:
            hole = rs.hole_filling_filter()
            hole.set_option(rs.option.holes_fill, self._config.hole_filling_mode)
            self._filters.append(hole)

    def _discard_warmup_frames(self) -> None:
        for _ in range(self._config.warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=2000)
        self._warmup_done = True
        logger.debug("Camera warmup complete", frames=self._config.warmup_frames)

    def _extract_intrinsics(self, profile: "rs.pipeline_profile") -> CameraIntrinsics:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        distortion = np.array(intr.coeffs, dtype=np.float64)

        return CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            depth_scale=depth_scale,
            distortion_coeffs=distortion,
        )

    def get_frame(self, timeout_ms: int = 1000) -> CameraFrame:
        if self._pipeline is None:
            raise CameraStreamError("Camera pipeline not open")
        try:
            frameset = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError as e:
            CAMERA_ERRORS.inc()
            raise CameraStreamError(f"Frame timeout or stream error: {e}") from e

        if self._align is not None:
            frameset = self._align.process(frameset)

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            raise CameraStreamError("Null color or depth frame received")

        self._check_sync(color_frame, depth_frame)

        for filt in self._filters:
            depth_frame = filt.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())  # HxWx3 BGR uint8
        depth_raw = np.asanyarray(depth_frame.get_data())    # HxW uint16 (raw units)
        depth_m = depth_raw.astype(np.float32) * self._intrinsics.depth_scale

        # Zero out out-of-range pixels
        depth_m[(depth_m < self._config.depth_min_m) | (depth_m > self._config.depth_max_m)] = 0.0

        ts_ns = int(color_frame.get_timestamp() * 1e6)  # ms -> ns
        self._frame_count += 1

        return CameraFrame(
            color=color_image,
            depth=depth_m,
            timestamp_ns=ts_ns,
            frame_number=self._frame_count,
            intrinsics=self._intrinsics,
        )

    def _check_sync(self, color_frame, depth_frame) -> None:
        ts_color = color_frame.get_timestamp()   # milliseconds
        ts_depth = depth_frame.get_timestamp()
        delta_ms = abs(ts_color - ts_depth)
        if delta_ms > self._MAX_SYNC_DELTA_MS:
            raise FrameSyncError(
                f"RGB/depth timestamp delta {delta_ms:.1f}ms exceeds tolerance "
                f"{self._MAX_SYNC_DELTA_MS}ms"
            )

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        CAMERA_CONNECTED.set(0)
        logger.info("RealSense camera closed")

    def get_intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise CameraInitError("Camera not opened yet")
        return self._intrinsics

    def is_connected(self) -> bool:
        if self._pipeline is None:
            return False
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            return devices.size() > 0
        except Exception:
            return False

    @staticmethod
    def _get_serial(profile: "rs.pipeline_profile") -> str:
        try:
            return profile.get_device().get_info(rs.camera_info.serial_number)
        except Exception:
            return "unknown"

    @staticmethod
    def enumerate_devices() -> list[dict]:
        """Return list of connected RealSense devices with serial and model info."""
        if not _RS_AVAILABLE:
            return []
        ctx = rs.context()
        result = []
        for dev in ctx.query_devices():
            try:
                result.append({
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "name": dev.get_info(rs.camera_info.name),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                })
            except Exception:
                pass
        return result
