"""Point cloud filtering pipeline using Open3D.

Each stage is applied in order with recommended parameters for warehouse-scale
boxes at 0.3–4.0m range with a RealSense D455 depth sensor.

Parameter recommendations (can be tuned via PointCloudConfig):
  voxel_size_m:    0.004–0.008  (smaller = more detail, slower)
  sor_nb_neighbors: 20–50       (larger = stricter, safe value: 30)
  sor_std_ratio:   1.5–3.0      (smaller = stricter outlier rejection)
  ror_nb_points:   5–15
  ror_radius_m:    0.010–0.020  (scale with voxel size)
"""

from __future__ import annotations

import numpy as np

from src.utils.config import PointCloudConfig
from src.utils.types import PointCloud

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore


class PointCloudFilter:

    def __init__(self, config: PointCloudConfig) -> None:
        self._config = config

    def apply(self, pc: PointCloud) -> PointCloud:
        """Apply full filter pipeline. Returns new filtered PointCloud."""
        cloud = self._to_o3d(pc)

        # Stage 1: Voxel downsampling — uniform spatial density
        cloud = cloud.voxel_down_sample(self._config.voxel_size_m)

        # Stage 2: Statistical outlier removal — removes global outliers
        if len(cloud.points) >= self._config.sor_nb_neighbors:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=self._config.sor_nb_neighbors,
                std_ratio=self._config.sor_std_ratio,
            )

        # Stage 3: Radius outlier removal — removes isolated local points
        if len(cloud.points) >= self._config.ror_nb_points:
            cloud, _ = cloud.remove_radius_outlier(
                nb_points=self._config.ror_nb_points,
                radius=self._config.ror_radius_m,
            )

        # Stage 4: DBSCAN cluster extraction — keep dominant cluster
        cloud = self._extract_dominant_cluster(cloud)

        return self._from_o3d(cloud, pc)

    def _extract_dominant_cluster(
        self, cloud: "o3d.geometry.PointCloud"
    ) -> "o3d.geometry.PointCloud":
        """
        Use DBSCAN to cluster points. Return the largest cluster that falls
        within the expected size range. This handles scenes with floor, walls,
        and background clutter leaking through the segmentation mask.
        """
        if len(cloud.points) < self._config.dbscan_min_points:
            return cloud

        labels = np.array(cloud.cluster_dbscan(
            eps=self._config.dbscan_eps_m,
            min_points=self._config.dbscan_min_points,
            print_progress=False,
        ))

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            return cloud

        # Find largest cluster within valid point count range
        valid = [
            (lbl, cnt)
            for lbl, cnt in zip(unique_labels, counts)
            if self._config.min_cluster_points <= cnt <= self._config.max_cluster_points
        ]
        if not valid:
            # Fallback: just take the largest regardless of bounds
            valid = list(zip(unique_labels, counts))

        best_label = max(valid, key=lambda x: x[1])[0]
        indices = np.where(labels == best_label)[0]
        return cloud.select_by_index(indices.tolist())

    @staticmethod
    def _to_o3d(pc: PointCloud) -> "o3d.geometry.PointCloud":
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None and len(pc.colors) == len(pc.points):
            cloud.colors = o3d.utility.Vector3dVector(pc.colors)
        return cloud

    @staticmethod
    def _from_o3d(cloud: "o3d.geometry.PointCloud", original: PointCloud) -> PointCloud:
        pts = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        normals = np.asarray(cloud.normals) if cloud.has_normals() else None
        return PointCloud(points=pts, colors=colors, normals=normals)
