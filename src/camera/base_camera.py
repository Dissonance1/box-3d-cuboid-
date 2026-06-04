"""Abstract camera interface for all RGB-D sensors.

New cameras (ZED 2i, Orbbec, etc.) are added by subclassing BaseCamera
without touching any downstream pipeline code.
"""

from __future__ import annotations

import abc
import threading
from typing import Iterator

from src.utils.types import CameraFrame, CameraIntrinsics


class BaseCamera(abc.ABC):
    """Thread-safe base class for all RGB-D cameras."""

    def __init__(self) -> None:
        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0

    @abc.abstractmethod
    def open(self) -> None:
        """Open the camera and start streaming. Raises CameraInitError on failure."""

    @abc.abstractmethod
    def close(self) -> None:
        """Stop streaming and release hardware resources."""

    @abc.abstractmethod
    def get_frame(self, timeout_ms: int = 1000) -> CameraFrame:
        """
        Retrieve the next synchronized RGB-Depth frame.

        Blocks until a frame is available or timeout_ms elapses.
        Raises CameraStreamError if the stream is broken.
        Raises FrameSyncError if RGB/depth timestamps diverge beyond 16ms.
        """

    @abc.abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """Return camera intrinsic parameters (calibrated)."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if hardware is present and streaming."""

    def stream(self, timeout_ms: int = 1000) -> Iterator[CameraFrame]:
        """Yield frames continuously until stop() is called."""
        self._running = True
        while self._running:
            try:
                yield self.get_frame(timeout_ms)
            except Exception:
                if not self._running:
                    return
                raise

    def stop(self) -> None:
        """Signal stream() to stop after current frame."""
        self._running = False

    def __enter__(self) -> "BaseCamera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count
