"""FastAPI application factory and lifecycle management.

The pipeline runs in a background thread. The API layer reads from a
thread-safe result cache — it never directly calls pipeline code.
This decouples API latency from frame processing latency.

Endpoints are documented via OpenAPI at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    BoxDetection,
    CameraInfo,
    ConfidenceDetail,
    Dimensions3D,
    FrameDetections,
    HealthResponse,
    PoseAngles,
    Position3D,
    Quaternion,
    SystemMetrics,
)
from src.utils.logger import get_logger
from src.utils.types import BoxMeasurement, FrameResult

logger = get_logger(__name__)

# Shared state — populated by pipeline thread, read by API handlers
_state: dict[str, Any] = {
    "latest_result": None,
    "camera_info": None,
    "pipeline_running": False,
    "start_time": time.time(),
    "frame_count": 0,
    "last_fps": 0.0,
    "last_pipeline_ms": 0.0,
    "last_detection_ms": 0.0,
    "last_geometry_ms": 0.0,
    "camera_connected": False,
}


def update_api_state(
    result: FrameResult,
    camera_connected: bool,
    pipeline_ms: float = 0.0,
    detection_ms: float = 0.0,
    geometry_ms: float = 0.0,
) -> None:
    """Called by the pipeline thread to update the API cache."""
    _state["latest_result"] = result
    _state["camera_connected"] = camera_connected
    _state["pipeline_running"] = True
    _state["frame_count"] = result.frame_number
    _state["last_fps"] = result.fps
    _state["last_pipeline_ms"] = pipeline_ms
    _state["last_detection_ms"] = detection_ms
    _state["last_geometry_ms"] = geometry_ms


def set_camera_info(camera_type: str, width: int, height: int, fps: int,
                    fx: float, fy: float, cx: float, cy: float, depth_scale: float) -> None:
    _state["camera_info"] = {
        "type": camera_type, "width": width, "height": height, "fps": fps,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": depth_scale,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("REST API starting")
    yield
    logger.info("REST API shutting down")


def create_app(config=None) -> FastAPI:
    app = FastAPI(
        title="Box Pose Estimation API",
        description=(
            "Real-time 3D box pose, orientation, and dimension estimation "
            "using Intel RealSense D455 and YOLO11-Seg."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    origins = config.api.cors_origins if config else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Endpoints ──────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        fps = _state.get("last_fps", 0.0)
        connected = _state.get("camera_connected", False)
        running = _state.get("pipeline_running", False)

        if not connected:
            status = "unhealthy"
        elif fps < 5.0:
            status = "degraded"
        else:
            status = "ok"

        return HealthResponse(
            status=status,
            camera=connected,
            detector=running,
            pipeline_running=running,
            fps=round(fps, 1),
        )

    @app.get("/detections", response_model=FrameDetections, tags=["Detections"])
    async def get_detections():
        """Latest frame with all detected box measurements."""
        result: FrameResult | None = _state.get("latest_result")
        if result is None:
            raise HTTPException(503, "No frames available yet")
        return _frame_result_to_model(result)

    @app.get("/pose/{track_id}", response_model=BoxDetection, tags=["Detections"])
    async def get_pose(track_id: int):
        """Pose for a specific track ID from the latest frame."""
        result: FrameResult | None = _state.get("latest_result")
        if result is None:
            raise HTTPException(503, "No frames available yet")
        for m in result.measurements:
            if m.track_id == track_id:
                return _measurement_to_model(m)
        raise HTTPException(404, f"Track {track_id} not in latest frame")

    @app.get("/tracking", tags=["Tracking"])
    async def get_tracking():
        """List of active track IDs with last known position."""
        result: FrameResult | None = _state.get("latest_result")
        if result is None:
            return {"tracks": []}
        return {
            "frame": result.frame_number,
            "tracks": [
                {
                    "track_id": m.track_id,
                    "class": m.detection.class_name,
                    "position": m.pose.position.tolist(),
                    "confidence": m.confidence.overall,
                }
                for m in result.measurements
            ],
        }

    @app.get("/camera", response_model=CameraInfo, tags=["System"])
    async def get_camera():
        info = _state.get("camera_info")
        if info is None:
            raise HTTPException(503, "Camera info not available")
        return CameraInfo(connected=_state["camera_connected"], **info)

    @app.get("/metrics", response_model=SystemMetrics, tags=["System"])
    async def get_metrics():
        uptime = time.time() - _state.get("start_time", time.time())
        result = _state.get("latest_result")
        active = len(result.measurements) if result else 0
        return SystemMetrics(
            fps=round(_state.get("last_fps", 0.0), 1),
            frame_count=_state.get("frame_count", 0),
            active_tracks=active,
            pipeline_latency_ms=round(_state.get("last_pipeline_ms", 0.0), 2),
            detection_latency_ms=round(_state.get("last_detection_ms", 0.0), 2),
            geometry_latency_ms=round(_state.get("last_geometry_ms", 0.0), 2),
            camera_connected=_state.get("camera_connected", False),
            uptime_s=round(uptime, 1),
        )

    @app.get("/system-status", tags=["System"])
    async def get_system_status():
        """Combined health + metrics for dashboard consumers."""
        result = _state.get("latest_result")
        return {
            "status": "ok" if _state.get("pipeline_running") else "offline",
            "fps": round(_state.get("last_fps", 0.0), 1),
            "active_tracks": len(result.measurements) if result else 0,
            "camera_connected": _state.get("camera_connected", False),
            "frame_count": _state.get("frame_count", 0),
            "uptime_s": round(time.time() - _state.get("start_time", time.time()), 1),
        }

    return app


def _measurement_to_model(m: BoxMeasurement) -> BoxDetection:
    p = m.pose
    q = p.quaternion
    d = m.dimensions
    c = m.confidence
    unc = (d.uncertainty_m * 1000).tolist()
    return BoxDetection(
        track_id=m.track_id,
        timestamp=m.timestamp_s,
        class_name=m.detection.class_name,
        position=Position3D(x=round(float(p.position[0]), 4),
                            y=round(float(p.position[1]), 4),
                            z=round(float(p.position[2]), 4)),
        pose=PoseAngles(roll=round(float(p.euler_deg[0]), 2),
                        pitch=round(float(p.euler_deg[1]), 2),
                        yaw=round(float(p.euler_deg[2]), 2)),
        quaternion=Quaternion(qw=round(float(q[0]), 5), qx=round(float(q[1]), 5),
                              qy=round(float(q[2]), 5), qz=round(float(q[3]), 5)),
        dimensions=Dimensions3D(
            length=round(d.length_m, 4), width=round(d.width_m, 4), height=round(d.height_m, 4),
            length_mm=round(d.length_m * 1000, 1), width_mm=round(d.width_m * 1000, 1),
            height_mm=round(d.height_m * 1000, 1),
            uncertainty_mm=[round(v, 2) for v in unc],
        ),
        confidence=ConfidenceDetail(
            overall=c.overall, segmentation=c.segmentation,
            depth_completeness=c.depth_completeness, point_density=c.point_density,
            plane_quality=c.plane_quality, cuboid_quality=c.cuboid_quality,
            tracking_stability=c.tracking_stability,
        ),
        processing_time_ms=round(m.processing_time_ms, 2),
    )


def _frame_result_to_model(result: FrameResult) -> FrameDetections:
    return FrameDetections(
        frame_number=result.frame_number,
        timestamp=result.timestamp_ns / 1e9,
        fps=round(result.fps, 1),
        count=result.count,
        detections=[_measurement_to_model(m) for m in result.measurements],
    )
