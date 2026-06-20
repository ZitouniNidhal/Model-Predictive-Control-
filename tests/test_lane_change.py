"""Tests for the build_lane_change_reference utility function."""

import math
import numpy as np
import pytest

from mpc_python.cvxpy_mpc.utils import build_lane_change_reference


# ---------------------------------------------------------------------------
# Shape and basic structure
# ---------------------------------------------------------------------------

def test_lane_change_output_shape():
    ref = build_lane_change_reference(n_points=100, speed=2.0, dt=0.1)
    assert ref.shape == (101, 4), f"Expected (101, 4), got {ref.shape}"


def test_lane_change_columns():
    ref = build_lane_change_reference(n_points=50, speed=1.5, dt=0.1)
    assert ref.shape[1] == 4  # [x, y, yaw, v]


def test_lane_change_all_finite():
    ref = build_lane_change_reference(n_points=80, speed=2.0, dt=0.1)
    assert np.all(np.isfinite(ref)), "Reference contains NaN or Inf values."


# ---------------------------------------------------------------------------
# Geometric correctness
# ---------------------------------------------------------------------------

def test_lane_change_starts_near_origin():
    """The first point should be near x=0, y=0 (start of approach)."""
    ref = build_lane_change_reference(n_points=100, speed=2.0, dt=0.1)
    assert abs(ref[0, 0]) < 2.0, f"x start too far from 0: {ref[0, 0]}"
    assert abs(ref[0, 1]) < 0.5, f"y start too far from 0: {ref[0, 1]}"


def test_lane_change_ends_at_lane_width():
    """The last point should be at approximately y=lane_width."""
    lane_width = 3.5
    ref = build_lane_change_reference(
        n_points=200, speed=2.0, dt=0.1, lane_width=lane_width,
        straight_length=20.0, transition_length=15.0,
    )
    assert abs(ref[-1, 1] - lane_width) < 0.5, (
        f"Final y={ref[-1, 1]:.3f} not close to lane_width={lane_width}"
    )


def test_lane_change_monotone_x():
    """X coordinates should be monotonically non-decreasing."""
    ref = build_lane_change_reference(n_points=100, speed=2.0, dt=0.1)
    assert np.all(np.diff(ref[:, 0]) >= -1e-6), "X coordinates are not monotone."


def test_lane_change_speed_within_bounds():
    """All speeds should be between 30 % of target and 100 %."""
    target_speed = 3.0
    ref = build_lane_change_reference(n_points=150, speed=target_speed, dt=0.1)
    speeds = ref[:, 3]
    assert np.all(speeds >= 0.3 * target_speed - 1e-6), (
        f"Min speed {speeds.min():.3f} below 30% of {target_speed}."
    )
    assert np.all(speeds <= target_speed + 1e-6), (
        f"Max speed {speeds.max():.3f} exceeds target {target_speed}."
    )


def test_lane_change_yaw_bounded():
    """Heading angles should be within [-π, π]."""
    ref = build_lane_change_reference(n_points=100, speed=2.0, dt=0.1)
    assert np.all(ref[:, 2] >= -math.pi - 1e-6)
    assert np.all(ref[:, 2] <= math.pi + 1e-6)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_lane_change_invalid_speed():
    with pytest.raises(ValueError, match="speed"):
        build_lane_change_reference(n_points=50, speed=-1.0, dt=0.1)


def test_lane_change_invalid_lane_width():
    with pytest.raises(ValueError, match="lane_width"):
        build_lane_change_reference(n_points=50, speed=2.0, dt=0.1, lane_width=0.0)


def test_lane_change_invalid_dt():
    with pytest.raises(ValueError, match="dt"):
        build_lane_change_reference(n_points=50, speed=2.0, dt=-0.05)


def test_lane_change_custom_parameters():
    ref = build_lane_change_reference(
        n_points=80,
        speed=1.0,
        dt=0.05,
        lane_width=2.0,
        straight_length=10.0,
        transition_length=8.0,
    )
    assert ref.shape == (81, 4)
    assert np.all(np.isfinite(ref))
