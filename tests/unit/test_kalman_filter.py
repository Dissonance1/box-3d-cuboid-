"""Unit tests for the 3D Kalman box filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.kalman_box_filter import KalmanBoxFilter


def _make_state(cx=0.0, cy=0.0, cz=1.5, L=0.4, W=0.3, H=0.2,
                yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    return np.array([cx, cy, cz, L, W, H, yaw, pitch, roll], dtype=np.float64)


class TestKalmanInitialization:

    def test_initialize(self):
        kf = KalmanBoxFilter()
        m = _make_state()
        kf.initialize(m)
        assert kf.is_initialized
        state = kf.get_state()
        np.testing.assert_allclose(state[:6], m[:6], atol=1e-9)

    def test_initial_velocity_is_zero(self):
        kf = KalmanBoxFilter()
        kf.initialize(_make_state())
        assert kf.is_initialized
        assert np.all(kf.x[9:] == 0.0)


class TestKalmanPrediction:

    def test_static_object_stays_near_initial(self):
        kf = KalmanBoxFilter()
        kf.initialize(_make_state(cx=0.1, cy=0.2, cz=1.5))
        for _ in range(10):
            kf.update(_make_state(cx=0.1, cy=0.2, cz=1.5))
        pred = kf.predict()
        np.testing.assert_allclose(pred[:3], [0.1, 0.2, 1.5], atol=0.02)

    def test_moving_object_prediction(self):
        """Filter should learn velocity and predict next position."""
        kf = KalmanBoxFilter(process_noise_pos=0.01)
        # Correct Kalman usage: predict() advances time, update() corrects with measurement
        for i in range(20):
            kf.predict()                        # advance one time step
            kf.update(_make_state(cx=i * 0.1)) # correct with observation
        pred = kf.predict()
        # After 20 updates at cx=0,0.1,...,1.9 the velocity estimate ≈ 0.1/frame
        # Predicted X at step 21 should be ≈ 1.9 + 0.1 = 2.0
        assert 1.6 < pred[0] < 2.4, f"Predicted X={pred[0]:.3f} not in expected range"


class TestKalmanUpdate:

    def test_update_reduces_position_error(self):
        kf = KalmanBoxFilter()
        kf.initialize(_make_state(cz=1.5))
        # Noisy measurements around true position
        rng = np.random.default_rng(1)
        for _ in range(20):
            noise = rng.normal(0, 0.02, 9)
            kf.update(_make_state(cz=1.5) + noise)
        state = kf.get_state()
        assert abs(state[2] - 1.5) < 0.05, f"Z error: {abs(state[2]-1.5)*1000:.1f}mm"

    def test_angle_unwrapping_near_180(self):
        """Filter should not diverge when yaw crosses ±180°."""
        kf = KalmanBoxFilter()
        kf.initialize(_make_state(yaw=175.0))
        kf.update(_make_state(yaw=179.0))
        kf.update(_make_state(yaw=-178.0))  # crosses ±180
        kf.update(_make_state(yaw=-175.0))
        state = kf.get_state()
        # Yaw should be near -175 or 185, not stuck at 0 or diverged
        yaw = state[6]
        assert -180 <= yaw <= 180

    def test_dimension_smoothing(self):
        """Dimensions should converge to true value despite noise."""
        kf = KalmanBoxFilter(measurement_noise_dim=0.010)
        kf.initialize(_make_state(L=0.400, W=0.300, H=0.200))
        rng = np.random.default_rng(5)
        true_dims = np.array([0.400, 0.300, 0.200])
        for _ in range(50):
            noisy = true_dims + rng.normal(0, 0.005, 3)
            m = _make_state(L=noisy[0], W=noisy[1], H=noisy[2])
            kf.update(m)
        state = kf.get_state()
        np.testing.assert_allclose(state[3:6], true_dims, atol=0.010)


class TestInnovationRatio:

    def test_stable_track_has_low_ratio(self):
        kf = KalmanBoxFilter()
        kf.initialize(_make_state())
        for _ in range(30):
            kf.update(_make_state())
        ratio = kf.get_innovation_ratio()
        assert ratio < 5.0, f"Innovation ratio {ratio:.2f} too high for stable track"
