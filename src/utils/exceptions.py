"""Custom exception hierarchy for the box pose estimation system."""

from __future__ import annotations


class BoxPoseSystemError(Exception):
    """Base exception for all system errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class CameraError(BoxPoseSystemError):
    """Camera hardware or SDK errors."""


class CameraInitError(CameraError):
    """Camera failed to initialize."""


class CameraStreamError(CameraError):
    """Camera stream interrupted or unavailable."""


class CalibrationError(CameraError):
    """Camera calibration invalid or missing."""


class FrameSyncError(CameraError):
    """RGB and depth frames out of sync beyond tolerance."""


class DetectionError(BoxPoseSystemError):
    """Object detection errors."""


class ModelLoadError(DetectionError):
    """Detection model failed to load."""


class InferenceError(DetectionError):
    """Model inference failure."""


class DepthProcessingError(BoxPoseSystemError):
    """Depth data processing errors."""


class InsufficientDepthError(DepthProcessingError):
    """Not enough valid depth pixels in region."""

    def __init__(self, coverage: float, min_required: float) -> None:
        super().__init__(
            f"Depth coverage {coverage:.1%} below minimum {min_required:.1%}"
        )
        self.coverage = coverage
        self.min_required = min_required


class PointCloudError(BoxPoseSystemError):
    """Point cloud generation or filtering errors."""


class InsufficientPointsError(PointCloudError):
    """Point cloud has too few points for reconstruction."""

    def __init__(self, count: int, min_required: int) -> None:
        super().__init__(
            f"Point count {count} below minimum {min_required}"
        )
        self.count = count
        self.min_required = min_required


class GeometryError(BoxPoseSystemError):
    """Geometric reconstruction errors."""


class PlaneDetectionError(GeometryError):
    """RANSAC plane detection failed."""


class CuboidReconstructionError(GeometryError):
    """Could not reconstruct a valid cuboid from detected planes."""

    def __init__(self, message: str, planes_found: int = 0) -> None:
        super().__init__(message)
        self.planes_found = planes_found


class PoseEstimationError(BoxPoseSystemError):
    """Pose estimation errors."""


class TrackingError(BoxPoseSystemError):
    """Object tracking errors."""


class ConfigurationError(BoxPoseSystemError):
    """Configuration validation or loading errors."""


class APIError(BoxPoseSystemError):
    """REST API errors."""


class PipelineError(BoxPoseSystemError):
    """End-to-end pipeline errors."""
