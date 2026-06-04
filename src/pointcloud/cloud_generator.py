"""Point cloud generation from depth ROI.

Converts valid depth pixels within a segmentation mask to a 3D point cloud
in camera frame coordinates, then applies the full Open3D filtering pipeline.
"""

from __future__ import annotations

import numpy as np

from src.pointcloud.cloud_filter import PointCloudFilter
from src.pointcloud.cloud_validator import PointCloudValidator
from src.utils.config import PointCloudConfig
from src.utils.exceptions import InsufficientPointsError
from src.utils.logger import get_logger
from src.utils.metrics import POINTCLOUD_LATENCY
from src.utils.types import CameraFrame, CameraIntrinsics, DepthROI, PointCloud

import time

logger = get_logger(__name__)

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    logger.warning("open3d not installed — point cloud operations unavailable")


class PointCloudGenerator:
    """
    Generates and filters a 3D point cloud from a masked depth ROI.

    Coordinate system (camera frame):
      X: right
      Y: down
      Z: forward (depth axis)

    Pipeline:
      1. Back-project valid depth pixels to 3D using pinhole model
      2. Voxel downsample
      3. Statistical outlier removal (SOR)
      4. Radius outlier removal (ROR)
      5. DBSCAN cluster extraction — keep dominant cluster
      6. Normal estimation
    """

    def __init__(self, config: PointCloudConfig) -> None:
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required. Install with: pip install open3d")
        self._config = config
        self._filter = PointCloudFilter(config)
        self._validator = PointCloudValidator(config)

    def generate(
        self,
        frame: CameraFrame,
        depth_roi: DepthROI,
    ) -> PointCloud:
        t0 = time.perf_counter()

        points, colors = self._backproject(frame.intrinsics, depth_roi, frame.color)

        if len(points) < self._config.min_points_for_reconstruction:
            raise InsufficientPointsError(len(points), self._config.min_points_for_reconstruction)

        pc = PointCloud(points=points, colors=colors)
        pc = self._filter.apply(pc)

        if pc.size < self._config.min_points_for_reconstruction:
            raise InsufficientPointsError(pc.size, self._config.min_points_for_reconstruction)

        pc = self._estimate_normals(pc)
        self._validator.validate(pc)

        POINTCLOUD_LATENCY.observe(time.perf_counter() - t0)
        logger.debug("Point cloud generated", points=pc.size)
        return pc

    def _backproject(
        self,
        intrinsics: CameraIntrinsics,
        depth_roi: DepthROI,
        color: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Back-project valid depth pixels to 3D.

        For pixel (u, v) with depth Z:
          X = (u - cx) * Z / fx
          Y = (v - cy) * Z / fy
          Z = depth_value
        """
        depth_map = depth_roi.depth_map
        valid = depth_map > 0

        v_coords, u_coords = np.where(valid)
        z = depth_map[v_coords, u_coords].astype(np.float64)

        x = (u_coords - intrinsics.cx) * z / intrinsics.fx
        y = (v_coords - intrinsics.cy) * z / intrinsics.fy

        points = np.column_stack([x, y, z])

        # Extract corresponding RGB colors (normalized 0-1)
        bgr = color[v_coords, u_coords].astype(np.float32) / 255.0
        colors = bgr[:, ::-1]  # BGR -> RGB

        return points, colors

    def _estimate_normals(self, pc: PointCloud) -> PointCloud:
        o3d_cloud = self._to_o3d(pc)
        o3d_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(
                radius=self._config.normal_estimation_radius_m
            )
        )
        # Orient normals towards the camera (origin)
        o3d_cloud.orient_normals_towards_camera_location(camera_location=np.zeros(3))
        pc.normals = np.asarray(o3d_cloud.normals)
        return pc

    @staticmethod
    def _to_o3d(pc: PointCloud) -> "o3d.geometry.PointCloud":
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None:
            o3d_cloud.colors = o3d.utility.Vector3dVector(pc.colors)
        return o3d_cloud

    @staticmethod
    def _from_o3d(o3d_cloud: "o3d.geometry.PointCloud", original_pc: PointCloud) -> PointCloud:
        points = np.asarray(o3d_cloud.points)
        colors = np.asarray(o3d_cloud.colors) if o3d_cloud.has_colors() else original_pc.colors
        normals = np.asarray(o3d_cloud.normals) if o3d_cloud.has_normals() else None
        return PointCloud(points=points, colors=colors, normals=normals)
