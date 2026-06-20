"""Tests for the mpc_python.cvxpy_mpc.metrics module."""

import numpy as np
import pytest

from mpc_python.cvxpy_mpc.metrics import (
    tracking_error_stats,
    control_effort,
    constraint_satisfaction,
    summarize,
)
from mpc_python.cvxpy_mpc.utils import compute_tracking_metrics


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_history():
    """A minimal but realistic history dict for testing metrics functions."""
    n = 20
    t = np.linspace(0, 2 * np.pi, n)
    states = np.column_stack([
        5.0 * np.cos(t),
        5.0 * np.sin(t),
        t + np.pi / 2,
        np.ones(n) * 1.5,
    ])
    controls = np.column_stack([
        0.1 * np.sin(t[:-1]),
        0.05 * np.cos(t[:-1]),
    ])
    errors = np.abs(np.sin(t[:-1]))  # synthetic tracking errors in [0, 1]
    return {
        "x": list(states),
        "u": list(controls),
        "xref": list(states),
        "error": list(errors),
    }


# ---------------------------------------------------------------------------
# tracking_error_stats
# ---------------------------------------------------------------------------

def test_tracking_error_stats_values(sample_history):
    stats = tracking_error_stats(sample_history)
    assert stats["n_steps"] == len(sample_history["error"])
    assert stats["mean"] >= 0.0
    assert stats["max"] >= stats["mean"]
    assert stats["rms"] >= stats["mean"]


def test_tracking_error_stats_empty():
    stats = tracking_error_stats({"error": []})
    assert stats["n_steps"] == 0
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["max"])
    assert np.isnan(stats["rms"])


def test_tracking_error_stats_constant():
    history = {"error": [2.0, 2.0, 2.0]}
    stats = tracking_error_stats(history)
    assert np.isclose(stats["mean"], 2.0)
    assert np.isclose(stats["max"], 2.0)
    assert np.isclose(stats["rms"], 2.0)


# ---------------------------------------------------------------------------
# control_effort
# ---------------------------------------------------------------------------

def test_control_effort_positive(sample_history):
    effort = control_effort(sample_history["u"])
    assert effort["total_accel"] >= 0.0
    assert effort["total_steer"] >= 0.0


def test_control_effort_zero_variation():
    u = np.ones((10, 2)) * 0.5
    effort = control_effort(u)
    assert np.isclose(effort["total_accel"], 0.0)
    assert np.isclose(effort["total_steer"], 0.0)


def test_control_effort_single_step():
    u = [[0.1, 0.05]]  # only one step — no differences
    effort = control_effort(u)
    assert np.isnan(effort["total_accel"])


# ---------------------------------------------------------------------------
# constraint_satisfaction
# ---------------------------------------------------------------------------

def test_constraint_satisfaction_all_valid(sample_history):
    constraints = {"a_min": -3.0, "a_max": 2.0, "delta_min": -0.5, "delta_max": 0.5}
    sat = constraint_satisfaction(sample_history["u"], constraints)
    # All controls in sample_history are within [-0.5, 0.5]
    assert sat["a_min"] == 1.0
    assert sat["a_max"] == 1.0
    assert sat["delta_min"] == 1.0
    assert sat["delta_max"] == 1.0
    assert sat["overall"] == 1.0


def test_constraint_satisfaction_partial_violation():
    # Controls that violate a_max at every other step
    u = np.zeros((10, 2))
    u[::2, 0] = 5.0  # a = 5.0 > a_max = 2.0
    constraints = {"a_min": -3.0, "a_max": 2.0, "delta_min": -0.5, "delta_max": 0.5}
    sat = constraint_satisfaction(u, constraints)
    assert sat["a_max"] < 1.0
    assert sat["overall"] < 1.0


def test_constraint_satisfaction_empty():
    sat = constraint_satisfaction([], {"a_min": -3.0, "a_max": 2.0})
    assert np.isnan(sat["overall"])


# ---------------------------------------------------------------------------
# compute_tracking_metrics (utils helper)
# ---------------------------------------------------------------------------

def test_compute_tracking_metrics_keys(sample_history):
    metrics = compute_tracking_metrics(sample_history)
    expected_keys = {"mean_error", "max_error", "rms_error", "mean_speed",
                     "total_accel_effort", "total_steer_effort", "n_steps"}
    assert expected_keys.issubset(metrics.keys())


def test_compute_tracking_metrics_mean_speed(sample_history):
    metrics = compute_tracking_metrics(sample_history)
    assert np.isclose(metrics["mean_speed"], 1.5)


# ---------------------------------------------------------------------------
# summarize — just smoke test (output goes to stdout)
# ---------------------------------------------------------------------------

def test_summarize_runs_without_error(sample_history, capsys):
    constraints = {"a_min": -3.0, "a_max": 2.0, "delta_min": -0.5, "delta_max": 0.5}
    summarize(sample_history, constraints)
    captured = capsys.readouterr()
    assert "MPC PERFORMANCE SUMMARY" in captured.out
    assert "Mean track. error" in captured.out
    assert "Accel effort" in captured.out
