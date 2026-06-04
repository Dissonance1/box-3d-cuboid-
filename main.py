"""Application entry point.

Starts the full system:
  1. Load configuration
  2. Configure logging and metrics
  3. Open camera
  4. Load detector
  5. Start pipeline thread
  6. Start REST API (uvicorn)
  7. Start visualization loop (optional, main thread)

Shutdown is coordinated via threading.Event — Ctrl+C, SIGTERM, or
/health returning "unhealthy" for too long triggers clean shutdown.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box Pose & Dimension Estimation System")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    p.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization")
    p.add_argument("--no-api", action="store_true", help="Disable REST API")
    p.add_argument("--device", default=None, help="Override detector device (cuda/cpu/tensorrt)")
    p.add_argument("--log-level", default=None, help="Override log level")
    p.add_argument("--profile", action="store_true", help="Enable cProfile for one pipeline cycle")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    from src.utils.config import load_config
    config = load_config(args.config)

    if args.device:
        config.detector.device = args.device
    if args.log_level:
        config.logging.level = args.log_level.upper()
    if args.no_vis:
        config.visualization.enabled = False

    # ── Configure logging ──────────────────────────────────────────────────
    from src.utils.logger import configure_logging
    configure_logging(
        level=config.logging.level,
        fmt=config.logging.format,
        log_file=config.logging.file,
        rotation_mb=config.logging.rotation_mb,
        retention_days=config.logging.retention_days,
    )
    from src.utils.logger import get_logger
    logger = get_logger("main")
    logger.info("Box Pose Estimation System starting", config=args.config)

    # ── Prometheus metrics ─────────────────────────────────────────────────
    if config.metrics.enabled:
        from src.utils.metrics import start_prometheus_server
        start_prometheus_server(config.metrics.prometheus_port)
        logger.info("Prometheus metrics", port=config.metrics.prometheus_port)

    # ── Create camera ──────────────────────────────────────────────────────
    from src.camera.camera_factory import create_camera
    from src.camera.calibration import CalibrationValidator

    camera = create_camera(config.camera)
    camera.open()

    intrinsics = camera.get_intrinsics()
    CalibrationValidator().validate(intrinsics)

    from src.api.app import set_camera_info
    set_camera_info(
        config.camera.type,
        intrinsics.width, intrinsics.height, config.camera.fps,
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
        intrinsics.depth_scale,
    )

    # ── Load detector ──────────────────────────────────────────────────────
    from src.detector.yolo_detector import YOLODetector
    detector = YOLODetector(config.detector)
    detector.load()

    # ── Create pipeline ────────────────────────────────────────────────────
    from src.pipeline.detection_pipeline import DetectionPipeline
    pipeline = DetectionPipeline(config, camera, detector)

    # ── Shutdown event ─────────────────────────────────────────────────────
    shutdown = threading.Event()

    def _signal_handler(sig, _):
        logger.info("Shutdown signal received", signal=sig)
        shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Visualization ──────────────────────────────────────────────────────
    vis = None
    if config.visualization.enabled:
        from src.visualization.visualizer import Visualizer
        vis = Visualizer(config.visualization)

    # ── REST API (background thread) ───────────────────────────────────────
    api_thread = None
    if config.api.enabled and not args.no_api:
        from src.api.app import create_app
        app = create_app(config)
        uvicorn_config = uvicorn.Config(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(uvicorn_config)

        def _run_api():
            server.run()

        api_thread = threading.Thread(target=_run_api, daemon=True, name="api-server")
        api_thread.start()
        logger.info("REST API started", host=config.api.host, port=config.api.port)

    # ── Pipeline (background thread) ───────────────────────────────────────
    latest_result = threading.local()

    from src.api.app import update_api_state

    def _run_pipeline():
        try:
            for result in pipeline.run():
                update_api_state(
                    result,
                    camera_connected=camera.is_connected(),
                    pipeline_ms=result.pipeline_time_ms,
                )
                latest_result.value = result
                if shutdown.is_set():
                    break
        except Exception:
            logger.error("Pipeline thread crashed", exc_info=True)
        finally:
            shutdown.set()

    pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True, name="pipeline")
    pipeline_thread.start()

    # ── Shared frame buffer — pipeline thread writes, vis thread reads ────
    import threading as _threading
    _latest_frame = {"color": None}
    _frame_lock = _threading.Lock()
    _orig_process = pipeline._process_frame

    def _process_frame_with_capture(frame):
        with _frame_lock:
            _latest_frame["color"] = frame.color.copy()
        return _orig_process(frame)

    pipeline._process_frame = _process_frame_with_capture

    # ── Main loop (visualization) ──────────────────────────────────────────
    import cv2
    try:
        while not shutdown.is_set():
            if vis is not None:
                result = latest_result.value if hasattr(latest_result, "value") else None
                with _frame_lock:
                    color = _latest_frame["color"]
                if color is not None and result is not None:
                    rendered = vis.render(color, result, intrinsics)
                    cv2.imshow(config.visualization.window_name, rendered)
                elif color is not None:
                    cv2.imshow(config.visualization.window_name, color)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    shutdown.set()
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        shutdown.set()
    finally:
        logger.info("Shutting down")
        pipeline.stop()
        pipeline_thread.join(timeout=5.0)
        detector.unload()
        camera.close()
        if vis is not None:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
