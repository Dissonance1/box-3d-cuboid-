"""Shared pytest fixtures for unit, integration, and performance tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.config import (
    PlaneDetectionConfig,
    PointCloudConfig,
    SystemConfig,
    TrackingConfig,
)
from src.utils.types import (
    CameraIntrinsics,
    DetectionResult,
    PlaneModel,
    PointCloud,
)


@pytest.fixture
def camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=909.0, fy=909.0, cx=640.0, cy=360.0,
        width=1280, height=720, depth_scale=0.001,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture
def synthetic_box_points(rng) -> np.ndarray:
    """
    Synthesize point cloud for a 400x300x200mm box at 1.5m depth.
    Box centered at (0.05, 0.1, 1.5) with slight yaw rotation (15 deg).
    """
    L, W, H = 0.400, 0.300, 0.200
    cx, cy, cz = 0.05, 0.10, 1.50
    yaw_deg = 15.0
    yaw = np.deg2rad(yaw_deg)

    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0,           1, 0           ],
        [-np.sin(yaw),0, np.cos(yaw) ],
    ])

    all_pts = []
    density = 500  # points per face

    # Top face (y = cy - H/2)
    u = rng.uniform(-L/2, L/2, density)
    v = np.zeros(density)
    w = rng.uniform(-W/2, W/2, density)
    top = np.column_stack([u, np.full(density, -H/2), w])
    all_pts.append(top)

    # Front face (z = cz + W/2 in local → varies)
    u2 = rng.uniform(-L/2, L/2, density)
    v2 = rng.uniform(-H/2, H/2, density)
    front = np.column_stack([u2, v2, np.full(density, W/2)])
    all_pts.append(front)

    # Right face
    v3 = rng.uniform(-H/2, H/2, density)
    w3 = rng.uniform(-W/2, W/2, density)
    right = np.column_stack([np.full(density, L/2), v3, w3])
    all_pts.append(right)

    pts_local = np.vstack(all_pts)
    pts_world = (R @ pts_local.T).T + np.array([cx, cy, cz])

    # Add realistic RealSense depth noise (1-2mm at 1.5m)
    noise = rng.normal(0, 0.0015, pts_world.shape)
    pts_world += noise

    return pts_world.astype(np.float64)


@pytest.fixture
def synthetic_point_cloud(synthetic_box_points) -> PointCloud:
    n = len(synthetic_box_points)
    colors = np.ones((n, 3), dtype=np.float32) * 0.5
    return PointCloud(points=synthetic_box_points, colors=colors)


@pytest.fixture
def plane_config() -> PlaneDetectionConfig:
    return PlaneDetectionConfig()


@pytest.fixture
def system_config() -> SystemConfig:
    return SystemConfig()


@pytest.fixture
def simple_plane() -> PlaneModel:
    return PlaneModel(
        normal=np.array([0.0, 1.0, 0.0]),
        offset=-1.0,
        inlier_count=500,
        inlier_ratio=0.5,
        fitness=0.9,
        rmse=0.003,
        centroid=np.array([0.0, 1.0, 0.0]),
    )
