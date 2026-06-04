"""Temporal measurement stabilization.

Applies per-track exponential moving average (EMA) with outlier rejection
to smooth position, orientation, and dimension estimates over time.

Design tradeoffs:
  - High alpha → fast response to real motion, more noise passthrough
  - Low alpha → smooth output, lag behind fast motion
  - Separate alphas for position vs dimensions (boxes change shape rarely)
  - Outlier rejection prevents filter poisoning from bad measurements

Each track has its own TemporalState — the TrackManager holds one per track.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.utils.config import TemporalFilterConfig


class TemporalState:
    """Per-track temporal smoothing state."""

    def __init__(self, config: TemporalFilterConfig) -> None:
        self._config = config
        self._position_ema: np.ndarray | None = None
        self._dimensions_ema: np.ndarray | None = None
        self._angles_ema: np.ndarray | None = None

        # History for outlier detection
        self._position_history: deque[np.ndarray] = deque(maxlen=config.window_size)
        self._dimension_history: deque[np.ndarray] = deque(maxlen=config.window_size)
        self._angle_history: deque[np.ndarray] = deque(maxlen=config.window_size)

    def update(
        self,
        position: np.ndarray,
        dimensions: np.ndarray,
        angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update EMA with new measurement. Returns smoothed values.

        position: [cx, cy, cz] meters
        dimensions: [L, W, H] meters
        angles: [yaw, pitch, roll] degrees
        """
        alpha_p = self._config.position_alpha
        alpha_d = self._config.dimension_alpha
        alpha_a = self._config.angle_alpha

        pos_in = self._reject_outlier_position(position)
        dim_in = self._reject_outlier_dimension(dimensions)
        ang_in = self._reject_outlier_angle(angles)

        if self._position_ema is None:
            self._position_ema = pos_in.copy()
            self._dimensions_ema = dim_in.copy()
            self._angles_ema = ang_in.copy()
        else:
            self._position_ema = alpha_p * pos_in + (1 - alpha_p) * self._position_ema
            self._dimensions_ema = alpha_d * dim_in + (1 - alpha_d) * self._dimensions_ema
            # Angle EMA with circular wrapping
            angle_diff = ang_in - self._angles_ema
            angle_diff = ((angle_diff + 180) % 360) - 180
            self._angles_ema = self._angles_ema + alpha_a * angle_diff
            self._angles_ema = ((self._angles_ema + 180) % 360) - 180

        self._position_history.append(pos_in)
        self._dimension_history.append(dim_in)
        self._angle_history.append(ang_in)

        return (
            self._position_ema.copy(),
            self._dimensions_ema.copy(),
            self._angles_ema.copy(),
        )

    def _reject_outlier_position(self, pos: np.ndarray) -> np.ndarray:
        if len(self._position_history) < 3 or self._position_ema is None:
            return pos
        residual = np.linalg.norm(pos - self._position_ema)
        sigma = self._config.position_alpha * 0.05  # expected noise ~ 5cm
        if residual > self._config.outlier_sigma * max(sigma, 0.010):
            return self._position_ema.copy()
        return pos

    def _reject_outlier_dimension(self, dim: np.ndarray) -> np.ndarray:
        if len(self._dimension_history) < 3 or self._dimensions_ema is None:
            return dim
        residual = np.linalg.norm(dim - self._dimensions_ema)
        if residual > self._config.outlier_sigma * 0.030:
            return self._dimensions_ema.copy()
        # Clamp dimensions to physically plausible range
        dim = np.clip(dim, 0.010, 3.0)
        return dim

    def _reject_outlier_angle(self, ang: np.ndarray) -> np.ndarray:
        if self._angles_ema is None:
            return ang
        diff = np.abs(((ang - self._angles_ema + 180) % 360) - 180)
        if diff.max() > self._config.outlier_sigma * 10.0:
            return self._angles_ema.copy()
        return ang

    def reset(self) -> None:
        self._position_ema = None
        self._dimensions_ema = None
        self._angles_ema = None
        self._position_history.clear()
        self._dimension_history.clear()
        self._angle_history.clear()
