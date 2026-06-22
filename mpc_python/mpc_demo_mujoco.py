import argparse
from pathlib import Path
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:  # pragma: no cover
    mujoco = None

from mpc_python.cvxpy_mpc import (
    IterativeMPC,
    build_circular_reference,
    build_waypoint_reference,
    build_figure8_reference,
    build_lane_change_reference,
    load_yaml,
)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo MPC demo entry point.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
    parser.add_argument(
        "--reference-mode",
        choices=["circle", "waypoints", "figure8", "lane_change"],
        default=None,
        help="Reference path type to use in MuJoCo demo.",
    )
    args = parser.parse_args()

    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is not installed. Install `mujoco` and run the no-sim demo first: python mpc_python/mpc_demo_nosim.py"
        )

    model_path = Path(__file__).resolve().parent / "models" / "mushr" / "mushr.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    mpc_config = load_yaml(args.config)
    sim_config = load_yaml(args.simulation)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch(model, data)

    horizon = int(mpc_config["horizon"])
    dt = float(mpc_config["dt"])
    reference = generate_reference(sim_config, 1000, dt, args.reference_mode)

    mpc = IterativeMPC(mpc_config)
    u_prev = np.zeros((horizon, 2), dtype=float)
    step = 0

    while viewer.is_running and step < int(sim_config["simulation"].get("max_steps", 200)):
        x_pos = data.qpos[0]
        y_pos = data.qpos[1]
        yaw = data.qpos[2]
        speed = float(data.qvel[0])
        x_state = [x_pos, y_pos, yaw, speed]

        ref_index = min(step, len(reference) - horizon - 1)
        ref_segment = reference[ref_index : ref_index + horizon + 1]
        x_pred, u_opt = mpc.solve(x_state, ref_segment, u_init=u_prev)
        if u_opt is None or len(u_opt) == 0:
            print("MPC failed to produce a valid control sequence. Ending MuJoCo demo.")
            break

        u_cmd = np.asarray(u_opt[0], dtype=float)
        data.ctrl[0] = float(u_cmd[0])
        data.ctrl[1] = float(u_cmd[1])

        mujoco.mj_step(model, data)
        viewer.render()
        time.sleep(dt)
        u_prev = np.vstack([u_opt[1:], u_opt[-1:]]) if len(u_opt) > 1 else np.array(u_opt, dtype=float)
        step += 1

    viewer.close()


def generate_reference(sim_config, n_points, dt, override_mode=None):
    reference_config = sim_config.get("reference", {})
    reference_type = override_mode or reference_config.get("type", "circle")
    speed = float(reference_config.get("speed", sim_config.get("track", {}).get("speed", 1.5)))

    if reference_type == "waypoints":
        waypoints = reference_config.get("waypoints", [])
        if not waypoints:
            raise ValueError("Waypoints reference mode requires `reference.waypoints` in simulation.yaml")
        reference = build_waypoint_reference(waypoints, speed, dt)
    elif reference_type == "figure8":
        reference = build_figure8_reference(n_points=n_points, speed=speed, dt=dt)
    elif reference_type == "lane_change":
        lc = reference_config.get("lane_change", {})
        reference = build_lane_change_reference(
            n_points=n_points,
            speed=speed,
            dt=dt,
            lane_width=float(lc.get("lane_width", 3.5)),
            straight_length=float(lc.get("straight_length", 20.0)),
            transition_length=float(lc.get("transition_length", 15.0)),
        )
    else:
        track = sim_config["track"]
        reference = build_circular_reference(
            n_points=n_points,
            radius=float(track["radius"]),
            speed=speed,
            dt=dt,
            center=tuple(track["center"]),
            start_angle=float(track["start_angle"]),
        )

    if reference.shape[0] < n_points + 1:
        extra_rows = np.tile(reference[-1:], (n_points + 1 - reference.shape[0], 1))
        reference = np.vstack([reference, extra_rows])
    return reference


if __name__ == "__main__":
    main()
