"""Integration tests for the REST API.

Tests all endpoints with mock pipeline state injected via the shared
api state dict — no real camera or model required.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app, update_api_state, set_camera_info
from src.utils.config import SystemConfig
from src.utils.types import (
    BoxMeasurement,
    ConfidenceBreakdown,
    CuboidGeometry,
    DepthROI,
    DetectionResult,
    DimensionEstimate,
    FrameResult,
    PlaneModel,
    PointCloud,
    PoseEstimate,
)


def _make_mock_measurement(track_id: int = 1) -> BoxMeasurement:
    pos = np.array([0.1, 0.05, 1.5])
    R = np.eye(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    euler = np.array([0.0, 0.0, 15.0])

    dummy_plane = PlaneModel(
        normal=np.array([0.0, 1.0, 0.0]),
        offset=-1.5,
        inlier_count=300,
        inlier_ratio=0.4,
        fitness=0.85,
        rmse=0.003,
        centroid=pos.copy(),
    )
    corners = np.zeros((8, 3))

    return BoxMeasurement(
        track_id=track_id,
        timestamp_ns=int(time.time() * 1e9),
        detection=DetectionResult(
            class_id=0, class_name="box", confidence=0.92,
            bbox_xyxy=np.array([100, 100, 400, 400]),
            mask=np.zeros((720, 1280), dtype=np.uint8),
        ),
        depth_roi=DepthROI(
            mask=np.zeros((720, 1280), dtype=np.uint8),
            depth_map=np.zeros((720, 1280), dtype=np.float32),
            valid_ratio=0.75, mean_depth=1.5, min_depth=1.3, max_depth=1.7,
        ),
        point_cloud=PointCloud(
            points=np.random.rand(500, 3),
            colors=np.random.rand(500, 3),
        ),
        cuboid=CuboidGeometry(
            corners=corners,
            faces=[dummy_plane],
            center=pos.copy(),
            rotation_matrix=R.copy(),
            dimensions=np.array([0.401, 0.302, 0.198]),
            reconstruction_quality=0.88,
        ),
        pose=PoseEstimate(
            position=pos.copy(),
            rotation_matrix=R.copy(),
            quaternion=q.copy(),
            euler_deg=euler.copy(),
        ),
        dimensions=DimensionEstimate(
            length_m=0.401, width_m=0.302, height_m=0.198,
            uncertainty_m=np.array([0.003, 0.003, 0.004]),
            confidence=0.88,
        ),
        confidence=ConfidenceBreakdown(
            segmentation=0.92, depth_completeness=0.75, point_density=0.80,
            plane_quality=0.85, cuboid_quality=0.88, tracking_stability=0.90,
            overall=0.85,
        ),
        processing_time_ms=18.5,
    )


@pytest.fixture(scope="module")
def client():
    app = create_app(SystemConfig())
    set_camera_info("realsense", 1280, 720, 30, 909.0, 909.0, 640.0, 360.0, 0.001)
    result = FrameResult(
        frame_number=100,
        timestamp_ns=int(time.time() * 1e9),
        measurements=[_make_mock_measurement(1), _make_mock_measurement(2)],
        pipeline_time_ms=22.0,
        fps=28.5,
    )
    update_api_state(result, camera_connected=True, pipeline_ms=22.0)
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded", "unhealthy")
        assert "camera" in data
        assert "fps" in data

    def test_health_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        required_keys = {"status", "camera", "detector", "pipeline_running", "fps"}
        assert required_keys.issubset(data.keys())


class TestDetectionsEndpoint:

    def test_returns_frame(self, client):
        resp = client.get("/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert "frame_number" in data
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_detection_schema(self, client):
        resp = client.get("/detections")
        data = resp.json()
        if data["detections"]:
            d = data["detections"][0]
            assert "track_id" in d
            assert "position" in d
            assert "pose" in d
            assert "dimensions" in d
            assert "confidence" in d
            # Check nested schema
            assert "x" in d["position"]
            assert "roll" in d["pose"]
            assert "length" in d["dimensions"]
            assert "overall" in d["confidence"]

    def test_detection_count(self, client):
        resp = client.get("/detections")
        data = resp.json()
        assert data["count"] == len(data["detections"])

    def test_position_values_reasonable(self, client):
        resp = client.get("/detections")
        data = resp.json()
        for d in data["detections"]:
            assert -5.0 < d["position"]["x"] < 5.0
            assert 0.1 < d["position"]["z"] < 10.0


class TestPoseEndpoint:

    def test_pose_for_valid_track(self, client):
        resp = client.get("/pose/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["track_id"] == 1

    def test_pose_for_missing_track(self, client):
        resp = client.get("/pose/9999")
        assert resp.status_code == 404


class TestMetricsEndpoint:

    def test_metrics_schema(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "fps" in data
        assert "active_tracks" in data
        assert "camera_connected" in data


class TestTrackingEndpoint:

    def test_tracking_list(self, client):
        resp = client.get("/tracking")
        assert resp.status_code == 200
        data = resp.json()
        assert "tracks" in data


class TestSystemStatus:

    def test_system_status(self, client):
        resp = client.get("/system-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "fps" in data
