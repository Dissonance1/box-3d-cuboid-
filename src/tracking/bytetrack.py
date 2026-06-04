"""ByteTrack multi-object tracking implementation.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.

Key innovation: use BOTH high-confidence AND low-confidence detections.
- Round 1: Match high-conf detections to existing confirmed tracks
- Round 2: Match low-conf detections to unmatched tracks (re-activation)
- Round 3: Initialize new tentative tracks from unmatched high-conf detections

Association metric: 3D IoU between predicted track boxes and detected boxes.
Fallback to 2D projected IoU when 3D data is noisy.

State machine:
  TENTATIVE → CONFIRMED after min_hits consecutive matches
  CONFIRMED → LOST after one miss
  LOST → CONFIRMED on re-match within max_age frames
  LOST → REMOVED after max_age frames without match
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman_box_filter import KalmanBoxFilter
from src.utils.config import TrackingConfig
from src.utils.logger import get_logger
from src.utils.types import BoxMeasurement, TrackState

logger = get_logger(__name__)


class Track:
    """Single tracked object instance."""

    _next_id = 1

    def __init__(
        self,
        measurement: np.ndarray,
        config: TrackingConfig,
    ) -> None:
        self.track_id = Track._next_id
        Track._next_id += 1
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.config = config

        self.kf = KalmanBoxFilter(
            process_noise_pos=config.process_noise_pos,
            process_noise_dim=config.process_noise_dim,
            process_noise_angle=config.process_noise_angle,
            measurement_noise_pos=config.measurement_noise_pos,
            measurement_noise_dim=config.measurement_noise_dim,
            measurement_noise_angle=config.measurement_noise_angle,
        )
        self.kf.initialize(measurement)
        self._last_measurement = measurement.copy()

    def predict(self) -> np.ndarray:
        """Advance Kalman filter one step. Returns predicted state."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, measurement: np.ndarray) -> None:
        self.kf.update(measurement)
        self._last_measurement = measurement.copy()
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.config.min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    def get_predicted_state(self) -> np.ndarray:
        return self.kf.get_state()

    def get_stability_score(self) -> float:
        """0-1 score: 1 = stable/well-tracked, 0 = noisy/new."""
        hit_score = min(self.hits / 10.0, 1.0)
        freshness = max(0.0, 1.0 - self.time_since_update / 5.0)
        nis_score = max(0.0, 1.0 - (self.kf.get_innovation_ratio() - 1.0) / 20.0)
        return float(0.4 * hit_score + 0.3 * freshness + 0.3 * nis_score)


class ByteTracker:
    """
    ByteTrack tracker for 3D box objects.

    Input: list of (measurement_9d, confidence) per frame
    Output: list of (track_id, measurement_9d) for confirmed tracks

    measurement_9d = [cx, cy, cz, L, W, H, yaw, pitch, roll]
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: list[Track] = []

    def update(
        self,
        measurements: list[np.ndarray],
        confidences: list[float],
    ) -> list[tuple[int, np.ndarray, float]]:
        """
        Process one frame of detections.

        Returns list of (track_id, smoothed_state_9d, stability_score)
        for all confirmed tracks that were matched this frame.
        """
        # Step 1: Predict all track positions
        for track in self._tracks:
            track.predict()

        # Step 2: Split detections by confidence
        high_dets = [
            (m, c) for m, c in zip(measurements, confidences)
            if c >= self._config.high_score_threshold
        ]
        low_dets = [
            (m, c) for m, c in zip(measurements, confidences)
            if self._config.low_score_threshold <= c < self._config.high_score_threshold
        ]

        active_tracks = [t for t in self._tracks if not t.is_removed]
        confirmed = [t for t in active_tracks if t.is_confirmed]
        lost = [t for t in active_tracks if t.is_lost]
        tentative = [t for t in active_tracks if t.state == TrackState.TENTATIVE]

        # Step 3: Round 1 — match high-conf dets with confirmed tracks
        unmatched_conf_tracks, unmatched_high_dets = self._associate(
            confirmed, high_dets
        )

        # Step 4: Round 2 — match low-conf dets with unmatched confirmed tracks
        still_unmatched_conf_tracks, _ = self._associate(
            [confirmed[i] for i in unmatched_conf_tracks],
            low_dets,
        )

        # Step 5: Round 3 — match remaining high-conf dets with tentative tracks
        unmatched_tentative, unmatched_new_dets = self._associate(
            tentative, [high_dets[i] for i in unmatched_high_dets]
        )

        # Step 6: Match with lost tracks using high-conf unmatched
        remaining_lost = [t for t in lost]
        _, truly_new_dets = self._associate(
            remaining_lost, [high_dets[i] for i in [unmatched_high_dets[i] for i in unmatched_new_dets]]
        )

        # Step 7: Mark unmatched confirmed/tentative tracks
        for i in still_unmatched_conf_tracks:
            confirmed[i].mark_missed()
        for i in unmatched_tentative:
            tentative[i].mark_missed()

        # Step 8: Mark lost tracks that exceeded max_age
        for t in lost:
            if t.time_since_update > self._config.max_age:
                t.mark_removed()

        # Step 9: Initialize new tentative tracks for truly new detections
        for det_idx in truly_new_dets:
            m, c = high_dets[unmatched_high_dets[unmatched_new_dets[det_idx]]]
            new_track = Track(m, self._config)
            self._tracks.append(new_track)

        # Step 10: Remove dead tracks
        self._tracks = [t for t in self._tracks if not t.is_removed]

        # Collect results: only confirmed tracks matched this frame
        results = []
        for t in self._tracks:
            if t.is_confirmed and t.time_since_update == 0:
                results.append((t.track_id, t.get_predicted_state(), t.get_stability_score()))

        logger.debug(
            "ByteTracker update",
            detections=len(measurements),
            confirmed=len([t for t in self._tracks if t.is_confirmed]),
            lost=len([t for t in self._tracks if t.is_lost]),
        )
        return results

    def _associate(
        self,
        tracks: list[Track],
        detections: list[tuple[np.ndarray, float]],
    ) -> tuple[list[int], list[int]]:
        """
        Associate tracks with detections using IoU cost matrix.
        Applies Hungarian algorithm for optimal assignment.

        Returns:
          unmatched_track_indices, unmatched_detection_indices
        """
        if not tracks or not detections:
            return list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = self._build_cost_matrix(tracks, detections)
        track_idx, det_idx = linear_sum_assignment(cost_matrix)

        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        matched_pairs = []

        for ti, di in zip(track_idx, det_idx):
            if cost_matrix[ti, di] > (1.0 - self._config.match_iou_threshold):
                continue  # IoU too low — not a valid match
            matched_pairs.append((ti, di))
            unmatched_tracks.remove(ti)
            unmatched_dets.remove(di)

        for ti, di in matched_pairs:
            m, _ = detections[di]
            tracks[ti].update(m)

        return unmatched_tracks, unmatched_dets

    @staticmethod
    def _build_cost_matrix(
        tracks: list[Track],
        detections: list[tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Build cost matrix as (1 - 3D IoU) for all track-detection pairs.

        State format: [cx, cy, cz, L, W, H, yaw, pitch, roll]
        IoU computed in 3D axis-aligned box approximation (ignores rotation)
        for computational efficiency. Rotation-aware IoU can be enabled for
        high-yaw scenarios.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost = np.ones((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            pred = track.get_predicted_state()  # [cx,cy,cz,L,W,H,...]
            for j, (meas, _) in enumerate(detections):
                iou = _iou_3d_axis_aligned(pred, meas)
                cost[i, j] = 1.0 - iou
        return cost

    def get_active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_confirmed)

    def reset(self) -> None:
        self._tracks.clear()
        Track._next_id = 1


def _iou_3d_axis_aligned(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    3D IoU for two axis-aligned boxes.
    state = [cx, cy, cz, L, W, H, ...]
    """
    cx1, cy1, cz1, L1, W1, H1 = state1[:6]
    cx2, cy2, cz2, L2, W2, H2 = state2[:6]

    # Half extents
    hL1, hW1, hH1 = L1 / 2, W1 / 2, H1 / 2
    hL2, hW2, hH2 = L2 / 2, W2 / 2, H2 / 2

    # Intersection
    ix = max(0, min(cx1 + hL1, cx2 + hL2) - max(cx1 - hL1, cx2 - hL2))
    iy = max(0, min(cy1 + hW1, cy2 + hW2) - max(cy1 - hW1, cy2 - hW2))
    iz = max(0, min(cz1 + hH1, cz2 + hH2) - max(cz1 - hH1, cz2 - hH2))
    intersection = ix * iy * iz

    vol1 = L1 * W1 * H1
    vol2 = L2 * W2 * H2
    union = vol1 + vol2 - intersection + 1e-8

    return float(intersection / union)
