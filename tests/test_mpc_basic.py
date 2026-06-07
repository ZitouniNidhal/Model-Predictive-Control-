import numpy as np

from mpc_python.cvxpy_mpc import IterativeMPC, build_circular_reference


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

    assert u.shape == (cfg["horizon"], 2)
    assert np.all(np.isfinite(u))
