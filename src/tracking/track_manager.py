"""Track lifecycle manager.

Bridges ByteTracker with per-track Kalman state and temporal smoothing.
The manager holds the ground truth on active track states and provides
smoothed measurements to downstream consumers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.filtering.temporal_filter import TemporalState
from src.tracking.bytetrack import ByteTracker
from src.utils.config import SystemConfig
from src.utils.logger import get_logger
from src.utils.metrics import ACTIVE_TRACKS
from src.utils.types import BoxMeasurement, TrackState

logger = get_logger(__name__)


class TrackRecord:
    """All state associated with one active tracked object."""

    def __init__(self, track_id: int, config: SystemConfig) -> None:
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.temporal = TemporalState(config.temporal_filter)
        self.last_measurement: Optional[BoxMeasurement] = None
        self.frame_count = 0
        self.stability_score = 0.0

    def update(self, measurement: BoxMeasurement, stability: float) -> BoxMeasurement:
        """Apply temporal smoothing and return updated measurement."""
        pos = measurement.pose.position
        dims = np.array([
            measurement.dimensions.length_m,
            measurement.dimensions.width_m,
            measurement.dimensions.height_m,
        ])
        angles = measurement.pose.euler_deg  # [roll, pitch, yaw]

        pos_sm, dims_sm, ang_sm = self.temporal.update(pos, dims, angles)

        # Patch smoothed values back into measurement
        measurement.pose.position[:] = pos_sm
        measurement.pose.euler_deg[:] = ang_sm
        measurement.dimensions.length_m = float(dims_sm[0])
        measurement.dimensions.width_m = float(dims_sm[1])
        measurement.dimensions.height_m = float(dims_sm[2])

        self.stability_score = stability
        self.last_measurement = measurement
        self.frame_count += 1
        return measurement


class TrackManager:
    """Manages the full lifecycle of detected box tracks."""

    def __init__(self, config: SystemConfig) -> None:
        self._config = config
        self._tracker = ByteTracker(config.tracking)
        self._records: dict[int, TrackRecord] = {}

    def update(
        self,
        measurements: list[BoxMeasurement],
    ) -> list[BoxMeasurement]:
        """
        Update tracker with current frame measurements.
        Returns list of stabilized BoxMeasurements for confirmed tracks.
        """
        if not measurements:
            self._tracker.update([], [])
            self._prune_dead_tracks()
            return []

        # Convert measurements to tracker format: [cx,cy,cz,L,W,H,yaw,pitch,roll]
        tracker_inputs = []
        confidences = []
        for m in measurements:
            pos = m.pose.position
            dims = np.array([m.dimensions.length_m, m.dimensions.width_m, m.dimensions.height_m])
            angles = m.pose.euler_deg
            state_vec = np.concatenate([pos, dims, angles])
            tracker_inputs.append(state_vec)
            confidences.append(m.confidence.overall)

        # Run ByteTracker
        track_results = self._tracker.update(tracker_inputs, confidences)

        # Build mapping: measurement → track_id via spatial proximity
        output: list[BoxMeasurement] = []
        used_track_ids = set()

        for track_id, smoothed_state, stability in track_results:
            # Find the closest original measurement to this track
            best_meas = self._match_measurement_to_track(measurements, smoothed_state)
            if best_meas is None:
                continue

            # Ensure we have a record for this track
            if track_id not in self._records:
                self._records[track_id] = TrackRecord(track_id, self._config)

            record = self._records[track_id]
            best_meas.track_id = track_id
            best_meas.confidence.tracking_stability = stability
            best_meas.confidence.overall = self._recompute_overall(best_meas.confidence)

            smoothed = record.update(best_meas, stability)
            output.append(smoothed)
            used_track_ids.add(track_id)

        self._prune_dead_tracks(used_track_ids)
        ACTIVE_TRACKS.set(len(output))

        logger.debug(
            "Track update",
            confirmed=len(output),
            total_records=len(self._records),
        )
        return output

    @staticmethod
    def _match_measurement_to_track(
        measurements: list[BoxMeasurement],
        track_state: np.ndarray,
    ) -> Optional[BoxMeasurement]:
        """Find measurement closest to track position."""
        if not measurements:
            return None
        track_pos = track_state[:3]
        dists = [
            np.linalg.norm(m.pose.position - track_pos)
            for m in measurements
        ]
        min_idx = int(np.argmin(dists))
        if dists[min_idx] > 0.50:  # 50cm max assignment radius
            return None
        return measurements[min_idx]

    def _prune_dead_tracks(self, active_ids: set[int] | None = None) -> None:
        if active_ids is not None:
            dead = [tid for tid in self._records if tid not in active_ids]
            for tid in dead:
                if self._records[tid].frame_count > 0:
                    logger.debug("Track exited", track_id=tid,
                                 frames=self._records[tid].frame_count)
                del self._records[tid]

    @staticmethod
    def _recompute_overall(conf) -> float:
        from src.filtering.confidence_estimator import _WEIGHTS
        return max(0.0, min(1.0,
            _WEIGHTS["segmentation"] * conf.segmentation
            + _WEIGHTS["depth"] * conf.depth_completeness
            + _WEIGHTS["density"] * conf.point_density
            + _WEIGHTS["plane"] * conf.plane_quality
            + _WEIGHTS["cuboid"] * conf.cuboid_quality
            + _WEIGHTS["tracking"] * conf.tracking_stability
        ))

    def get_active_tracks(self) -> list[int]:
        return list(self._records.keys())

    def reset(self) -> None:
        self._tracker.reset()
        self._records.clear()
