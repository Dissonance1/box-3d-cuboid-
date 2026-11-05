"""LIPSedge AE400 / AE450 network (PoE) camera implementation.

Uses the LIPS OpenNI2 SDK. The LIPS driver finds the camera by IP address,
read from the AE4_DEVICE_IP environment variable at OpenNI init.
"""

from __future__ import annotations

import math
import os
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
    from openni import openni2

    _OPENNI_AVAILABLE = True
except ImportError:
    _OPENNI_AVAILABLE = False
    logger.warning("openni not installed — AE400Camera unavailable")

_DEFAULT_REDIST = (
    r"C:\Program Files\LIPSedge Camera SDK 1.02"
    r"\LIPSedge AE400_AE450 2.4.4.3_v1.5.5\OpenNI2\Redist"
)


class AE400Camera(BaseCamera):
    """
    Driver for LIPSedge AE400/AE450 over Ethernet.

    Depth is registered to the color sensor by the camera, so every depth
    pixel maps to the same color pixel. Intrinsics are derived from the
    stream field of view (OpenNI does not expose fx/fy directly).
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__()
        if not _OPENNI_AVAILABLE:
            raise CameraInitError("openni is not installed. Install with: pip install openni")
        self._config = config
        self._device = None
        self._depth = None
        self._color = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        # AE400 color and depth free-run ~half a frame apart (measured ~18ms at 30fps),
        # so allow one frame period; this still rejects dropped/stale frames
        self._max_sync_delta_ms = 1000.0 / config.fps

    def open(self) -> None:
        try:
            if self._config.ip_address:
                os.environ["AE4_DEVICE_IP"] = self._config.ip_address
            openni2.initialize(self._config.openni_redist or _DEFAULT_REDIST)
            self._device = openni2.Device.open_any()

            self._depth = self._start_stream(openni2.SENSOR_DEPTH, openni2.PIXEL_FORMAT_DEPTH_1_MM)
            self._color = self._start_stream(openni2.SENSOR_COLOR, openni2.PIXEL_FORMAT_RGB888)

            if self._config.align_depth_to_color:
                self._device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            try:
                self._device.set_depth_color_sync_enabled(True)
            except Exception:
                # LIPS driver rejects this while streaming; get_frame's timestamp check still guards sync
                logger.warning("AE400 hardware depth/color sync unavailable")

            self._intrinsics = self._extract_intrinsics()
            for _ in range(self._config.warmup_frames):
                self._read_pair()

            CAMERA_CONNECTED.set(1)
            logger.info(
                "AE400 camera opened",
                ip=self._config.ip_address or os.environ.get("AE4_DEVICE_IP"),
                width=self._config.width,
                height=self._config.height,
                fps=self._config.fps,
                serial=self._device.get_device_info().uri.decode(),
            )
        except Exception as e:
            CAMERA_CONNECTED.set(0)
            self.close()
            raise CameraInitError(f"Failed to open AE400 camera: {e}") from e

    def _start_stream(self, sensor, pixel_format):
        stream = self._device.create_stream(sensor)
        modes = stream.get_sensor_info().videoModes
        for m in modes:
            if (m.resolutionX, m.resolutionY, m.fps, m.pixelFormat) == (
                self._config.width, self._config.height, self._config.fps, pixel_format
            ):
                stream.set_video_mode(m)
                break
        else:
            available = sorted({(m.resolutionX, m.resolutionY, m.fps) for m in modes
                                if m.pixelFormat == pixel_format})
            raise CameraInitError(
                f"{self._config.width}x{self._config.height}@{self._config.fps} not supported. "
                f"Available: {available}"
            )
        stream.start()
        return stream

    def _extract_intrinsics(self) -> CameraIntrinsics:
        # Registered depth shares the color sensor's optics
        stream = self._color if self._config.align_depth_to_color else self._depth
        w, h = self._config.width, self._config.height
        fx = (w / 2.0) / math.tan(stream.get_horizontal_fov() / 2.0)
        fy = (h / 2.0) / math.tan(stream.get_vertical_fov() / 2.0)
        return CameraIntrinsics(
            fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
            width=w, height=h, depth_scale=0.001,
        )

    def _read_pair(self):
        color, depth = self._color.read_frame(), self._depth.read_frame()
        # Streams aren't hardware-synced: re-read whichever is a frame behind (max 2 catch-ups)
        lag_us = 750.0 * self._max_sync_delta_ms
        for _ in range(2):
            if color.timestamp < depth.timestamp - lag_us:
                color = self._color.read_frame()
            elif depth.timestamp < color.timestamp - lag_us:
                depth = self._depth.read_frame()
            else:
                break
        return color, depth

    def get_frame(self, timeout_ms: int = 1000) -> CameraFrame:
        if self._device is None:
            raise CameraStreamError("Camera not open")
        try:
            idx = openni2.wait_for_any_stream([self._depth], timeout_ms / 1000.0)
            if idx is None:
                raise CameraStreamError(f"No depth frame within {timeout_ms}ms")
            color_frame, depth_frame = self._read_pair()
        except CameraStreamError:
            CAMERA_ERRORS.inc()
            raise
        except Exception as e:
            CAMERA_ERRORS.inc()
            raise CameraStreamError(f"Frame read error: {e}") from e

        delta_ms = abs(color_frame.timestamp - depth_frame.timestamp) / 1000.0  # us -> ms
        if delta_ms > self._max_sync_delta_ms:
            raise FrameSyncError(
                f"RGB/depth timestamp delta {delta_ms:.1f}ms exceeds tolerance "
                f"{self._max_sync_delta_ms:.1f}ms"
            )

        h, w = color_frame.height, color_frame.width
        rgb = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(h, w, 3)
        color_image = rgb[:, :, ::-1].copy()  # RGB -> BGR
        depth_raw = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(
            depth_frame.height, depth_frame.width
        )
        depth_m = depth_raw.astype(np.float32) * self._intrinsics.depth_scale
        depth_m[(depth_m < self._config.depth_min_m) | (depth_m > self._config.depth_max_m)] = 0.0

        self._frame_count += 1
        return CameraFrame(
            color=color_image,
            depth=depth_m,
            timestamp_ns=int(color_frame.timestamp) * 1000,  # us -> ns
            frame_number=self._frame_count,
            intrinsics=self._intrinsics,
        )

    def close(self) -> None:
        for s in (self._depth, self._color):
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass
        self._depth = self._color = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
            openni2.unload()
        CAMERA_CONNECTED.set(0)
        logger.info("AE400 camera closed")

    def get_intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise CameraInitError("Camera not opened yet")
        return self._intrinsics

    def is_connected(self) -> bool:
        return self._device is not None and self._depth is not None
