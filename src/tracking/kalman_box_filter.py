"""Kalman filter for 3D box state tracking.

State vector (9D):
  [cx, cy, cz, L, W, H, yaw, pitch, roll]
  Position in meters, dimensions in meters, angles in degrees.

Velocity augmentation for prediction (18D full state):
  [cx, cy, cz, L, W, H, yaw, pitch, roll,
   vcx, vcy, vcz, vL, vW, vH, vyaw, vpitch, vroll]

Observation (9D):
  [cx, cy, cz, L, W, H, yaw, pitch, roll]

The constant-velocity model assumes smooth motion between frames.
Noise covariances are tuned for warehouse box handling:
  - Position changes up to ~5cm/frame at 30fps → ~150mm/s
  - Dimensions are near-constant → low process noise
  - Yaw can change quickly during manipulation → moderate noise
"""

from __future__ import annotations

import numpy as np

_STATE_DIM = 18   # [pos(3), dim(3), ang(3), vel_pos(3), vel_dim(3), vel_ang(3)]
_OBS_DIM = 9      # [pos(3), dim(3), ang(3)]


class KalmanBoxFilter:
    """
    Standard linear Kalman filter for a 3D box with constant-velocity model.

    Usage:
        kf = KalmanBoxFilter(process_noise, measurement_noise)
        kf.initialize(measurement_9d)
        prediction = kf.predict()
        kf.update(measurement_9d)
    """

    def __init__(
        self,
        process_noise_pos: float = 0.01,
        process_noise_dim: float = 0.005,
        process_noise_angle: float = 2.0,
        measurement_noise_pos: float = 0.02,
        measurement_noise_dim: float = 0.01,
        measurement_noise_angle: float = 3.0,
    ) -> None:
        n, m = _STATE_DIM, _OBS_DIM

        # State transition: x_k = F * x_{k-1}
        # Position/dimension/angle updated by adding velocity
        self.F = np.eye(n)
        self.F[:m, m:] = np.eye(m)  # x += v*dt (dt=1 frame)

        # Observation matrix: z = H * x (observe first 9 components)
        self.H = np.zeros((m, n))
        self.H[:m, :m] = np.eye(m)

        # Process noise covariance Q
        q_pos = process_noise_pos ** 2
        q_dim = process_noise_dim ** 2
        q_ang = process_noise_angle ** 2
        q_vpos = (process_noise_pos * 2) ** 2
        q_vdim = (process_noise_dim * 2) ** 2
        q_vang = (process_noise_angle * 2) ** 2

        Q_diag = np.array([
            q_pos, q_pos, q_pos,    # position
            q_dim, q_dim, q_dim,    # dimensions
            q_ang, q_ang, q_ang,    # angles
            q_vpos, q_vpos, q_vpos, # velocity position
            q_vdim, q_vdim, q_vdim, # velocity dimension
            q_vang, q_vang, q_vang, # velocity angle
        ])
        self.Q = np.diag(Q_diag)

        # Measurement noise covariance R
        R_diag = np.array([
            measurement_noise_pos ** 2,
            measurement_noise_pos ** 2,
            measurement_noise_pos ** 2,
            measurement_noise_dim ** 2,
            measurement_noise_dim ** 2,
            measurement_noise_dim ** 2,
            measurement_noise_angle ** 2,
            measurement_noise_angle ** 2,
            measurement_noise_angle ** 2,
        ])
        self.R = np.diag(R_diag)

        # State and covariance (initialized on first measurement)
        self.x = np.zeros(n)
        self.P = np.eye(n) * 100.0  # high initial uncertainty

        self._initialized = False

    def initialize(self, measurement: np.ndarray) -> None:
        """Initialize state from first measurement."""
        self.x[:_OBS_DIM] = measurement
        self.x[_OBS_DIM:] = 0.0  # zero initial velocity
        # Set initial uncertainty: high for velocity, moderate for observation
        self.P = np.eye(_STATE_DIM)
        self.P[:_OBS_DIM, :_OBS_DIM] = self.R * 2
        self.P[_OBS_DIM:, _OBS_DIM:] = np.eye(_OBS_DIM) * 100.0
        self._initialized = True

    def predict(self) -> np.ndarray:
        """Predict next state. Returns predicted observation vector (9D)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:_OBS_DIM].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update with new measurement. Returns updated state observation.

        Handles yaw wrapping: yaw angles near ±180° are unwrapped to prevent
        the filter from trying to average across the discontinuity.
        """
        z = measurement.copy()
        # Unwrap angle differences (indices 6,7,8 = yaw,pitch,roll)
        for angle_idx in [6, 7, 8]:
            diff = z[angle_idx] - self.x[angle_idx]
            if diff > 180:
                z[angle_idx] -= 360
            elif diff < -180:
                z[angle_idx] += 360

        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(_STATE_DIM) - K @ self.H) @ self.P

        # Normalize angles to [-180, 180]
        for angle_idx in [6, 7, 8]:
            self.x[angle_idx] = ((self.x[angle_idx] + 180) % 360) - 180

        return self.x[:_OBS_DIM].copy()

    def get_state(self) -> np.ndarray:
        """Return current state observation (9D)."""
        return self.x[:_OBS_DIM].copy()

    def get_innovation_ratio(self) -> float:
        """
        Measure how well the filter matches recent measurements.
        Low ratio → stable prediction; high ratio → noisy or new object.
        """
        S = self.H @ self.P @ self.H.T + self.R
        # Normalized innovation squared (NIS) — should be ~9 for 9D obs
        return float(np.trace(S) / (np.trace(self.R) + 1e-10))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def to_measurement(self) -> np.ndarray:
        return self.x[:_OBS_DIM].copy()
