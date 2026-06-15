import numpy as np
import pytest

# Skip tests when CVXPY is not installed in this environment.
pytest.importorskip("cvxpy")

from mpc_python.cvxpy_mpc import IterativeMPC, build_circular_reference, build_waypoint_reference, build_figure8_reference, load_yaml


def test_iterative_mpc_smoke():
    cfg = {
        "dt": 0.1,
        "horizon": 5,
        "max_iters": 2,
        "wheelbase": 0.16,
        "weights": {
            "q_x": 1.0,
            "q_y": 1.0,
            "q_yaw": 1.0,
            "q_v": 0.1,
            "r_a": 1.0,
            "r_delta": 1.0,
            "r_da": 1.0,
            "r_ddelta": 1.0,
        },
        "constraints": {
            "v_min": 0.0,
            "v_max": 2.0,
            "a_min": -1.0,
            "a_max": 1.0,
            "delta_min": -0.5,
            "delta_max": 0.5,
            "ddelta_max": 0.2,
        },
    }

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
    reference = build_figure8_reference(n_points=150, speed=1.5, dt=0.1)
    assert reference.shape[0] == 151
    assert reference.shape[1] == 4
    # Ensure speed profiling occurred in corners
    speeds = reference[:, 3]
    assert np.any(speeds < 1.5)
