"""Factory for creating camera instances by type string.

Adding a new camera requires only:
  1. Implement BaseCamera in a new module
  2. Register the class here with a key string
"""

from __future__ import annotations

from src.camera.base_camera import BaseCamera
from src.utils.config import CameraConfig
from src.utils.exceptions import ConfigurationError

_REGISTRY: dict[str, type[BaseCamera]] = {}


def _register(name: str, cls: type[BaseCamera]) -> None:
    _REGISTRY[name] = cls


def _lazy_register_all() -> None:
    """Lazy import so missing optional SDKs don't break unrelated camera types."""
    if "realsense" not in _REGISTRY:
        try:
            from src.camera.realsense_camera import RealSenseCamera
            _register("realsense", RealSenseCamera)
        except Exception:
            pass

    if "dummy" not in _REGISTRY:
        try:
            from src.camera.dummy_camera import DummyCamera
            _register("dummy", DummyCamera)
        except Exception:
            pass

    if "file" not in _REGISTRY:
        try:
            from src.camera.file_camera import FileCamera
            _register("file", FileCamera)
        except Exception:
            pass


def create_camera(config: CameraConfig) -> BaseCamera:
    """Instantiate a camera matching config.type."""
    _lazy_register_all()
    camera_type = config.type.lower()
    if camera_type not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ConfigurationError(
            f"Camera type '{camera_type}' not available. "
            f"Available types: {available}. "
            f"Check that the required SDK is installed."
        )
    return _REGISTRY[camera_type](config)


def list_available_cameras() -> list[str]:
    _lazy_register_all()
    return list(_REGISTRY.keys())
