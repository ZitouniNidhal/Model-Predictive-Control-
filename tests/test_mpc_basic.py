import numpy as np
import pytest

# Skip tests when CVXPY is not installed in this environment.
pytest.importorskip("cvxpy")

from mpc_python.cvxpy_mpc import (
    IterativeMPC,
    build_circular_reference,
    build_waypoint_reference,
    build_figure8_reference,
    load_yaml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    cfg = {
        "dt": 0.1,
        "horizon": 5,
        "max_iters": 2,
        "wheelbase": 0.16,
        "weights": {
            "q_x": 1.0, "q_y": 1.0, "q_yaw": 1.0, "q_v": 0.1,
            "r_a": 1.0, "r_delta": 1.0, "r_da": 1.0, "r_ddelta": 1.0,
        },
        "constraints": {
            "v_min": 0.0, "v_max": 2.0,
            "a_min": -1.0, "a_max": 1.0,
            "delta_min": -0.5, "delta_max": 0.5, "ddelta_max": 0.2,
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Original smoke tests (unchanged)
# ---------------------------------------------------------------------------

def test_iterative_mpc_smoke():
    cfg = _make_config()
    mpc = IterativeMPC(cfg)
    reference = build_circular_reference(20, radius=5.0, speed=1.0, dt=cfg["dt"])
    x0 = np.array([0.0, 0.0, 0.0, 0.1])
    _, u = mpc.solve(x0, reference[: cfg["horizon"] + 1], u_init=np.zeros((cfg["horizon"], 2)))

    assert u is not None
    assert u.shape == (cfg["horizon"], 2)
    assert np.all(np.isfinite(u))


def test_build_waypoint_reference():
    waypoints = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]
    reference = build_waypoint_reference(waypoints, speed=1.0, dt=0.1)

    assert reference.shape[1] == 4
    assert np.isclose(reference[0, 0], 0.0)
    assert np.isclose(reference[-1, 1], 2.0)


def test_build_waypoint_reference_curvature():
    waypoints = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
    reference = build_waypoint_reference(waypoints, speed=2.0, dt=0.1)

    assert reference.shape[1] == 4
    speeds = reference[:, 3]
    assert np.any(speeds < 2.0)
    assert np.all(speeds >= 0.6)


def test_load_yaml(tmp_path):
    sample = {
        "dt": 0.1,
        "horizon": 10,
        "weights": {"q_x": 1.0},
    }
    yaml_path = tmp_path / "sample.yaml"
    yaml_path.write_text("dt: 0.1\nhorizon: 10\nweights:\n  q_x: 1.0\n", encoding="utf-8")

    loaded = load_yaml(str(yaml_path))
    assert loaded["dt"] == sample["dt"]
    assert loaded["horizon"] == sample["horizon"]
    assert loaded["weights"]["q_x"] == 1.0


def test_build_figure8_reference():
    reference = build_figure8_reference(n_points=150, speed=4.0, dt=0.1)
    assert reference.shape[0] == 151
    assert reference.shape[1] == 4
    # Ensure speed profiling occurred in corners
    speeds = reference[:, 3]
    assert np.any(speeds < 4.0)


# ---------------------------------------------------------------------------
# New tests: robustness
# ---------------------------------------------------------------------------

def test_invalid_config_raises_negative_dt():
    """Negative dt must raise ValueError."""
    with pytest.raises(ValueError, match="dt must be positive"):
        IterativeMPC({"dt": -0.1, "horizon": 5})


def test_invalid_config_raises_zero_horizon():
    """Zero horizon must raise ValueError."""
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        IterativeMPC({"dt": 0.1, "horizon": 0})


def test_invalid_config_raises_negative_wheelbase():
    """Non-positive wheelbase must raise ValueError."""
    with pytest.raises(ValueError, match="wheelbase must be positive"):
        IterativeMPC({"dt": 0.1, "horizon": 5, "wheelbase": -1.0})


def test_terminal_velocity_within_bounds():
    """Optimal terminal velocity must respect [v_min, v_max]."""
    cfg = _make_config()
    v_min = cfg["constraints"]["v_min"]
    v_max = cfg["constraints"]["v_max"]

    mpc = IterativeMPC(cfg)
    reference = build_circular_reference(20, radius=5.0, speed=1.0, dt=cfg["dt"])
    x0 = np.array([0.0, 0.0, 0.0, 1.0])
    x_traj, u = mpc.solve(x0, reference[: cfg["horizon"] + 1])

    assert x_traj is not None, "Solver should succeed"
    terminal_v = x_traj[-1, 3]
    assert v_min - 1e-3 <= terminal_v <= v_max + 1e-3, (
        f"Terminal velocity {terminal_v:.4f} outside [{v_min}, {v_max}]"
    )


def test_solve_returns_info_dict():
    """return_info=True should return a third element with iteration metadata."""
    cfg = _make_config()
    mpc = IterativeMPC(cfg)
    reference = build_circular_reference(20, radius=5.0, speed=1.0, dt=cfg["dt"])
    x0 = np.array([0.0, 0.0, 0.0, 0.5])

    result = mpc.solve(x0, reference[: cfg["horizon"] + 1], return_info=True)
    assert len(result) == 3, "return_info=True should yield a 3-tuple"

    _, u, info = result
    assert u is not None
    assert isinstance(info, dict)
    assert "iterations" in info
    assert "obj_val" in info
    assert "status" in info
    assert info["iterations"] >= 1


def test_sqp_early_exit_feasible():
    """With a warm-start very close to optimal, SQP should exit before max_iters."""
    cfg = _make_config(max_iters=10, sqp_tol=1e-2)
    mpc = IterativeMPC(cfg)
    reference = build_circular_reference(20, radius=5.0, speed=1.0, dt=cfg["dt"])
    x0 = np.array([5.0, 0.0, 1.57, 1.0])  # already on the circle

    _, u, info = mpc.solve(x0, reference[: cfg["horizon"] + 1], return_info=True)
    assert u is not None
    assert info["iterations"] < cfg["max_iters"], (
        f"Expected early exit but used all {cfg['max_iters']} iterations."
    )


def test_solve_with_static_obstacle():
    """Solver should succeed (possibly using slack) when an obstacle is present."""
    cfg = _make_config()
    mpc = IterativeMPC(cfg)
    reference = build_circular_reference(20, radius=5.0, speed=1.0, dt=cfg["dt"])
    x0 = np.array([5.0, 0.0, 1.57, 0.5])

    obstacle = [{"center": [6.0, 0.0], "radius": 0.5}]
    _, u = mpc.solve(x0, reference[: cfg["horizon"] + 1], obstacles=obstacle)
    # With obstacle_slack=True (default) the solver should still find a solution
    assert u is not None, "Solver should succeed with slack obstacle constraints."
    assert u.shape == (cfg["horizon"], 2)
