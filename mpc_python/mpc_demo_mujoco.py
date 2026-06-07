import argparse
from pathlib import Path
import time

try:
    import mujoco
    import mujoco.viewer
except ImportError:  # pragma: no cover
    mujoco = None

from mpc_python.cvxpy_mpc import IterativeMPC, load_yaml, build_circular_reference


def main():
    parser = argparse.ArgumentParser(description="MuJoCo MPC demo entry point.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
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
    track = sim_config["track"]
    reference = build_circular_reference(
        n_points=1000,
        radius=float(track["radius"]),
        speed=float(track["speed"]),
        dt=dt,
        center=tuple(track["center"]),
        start_angle=float(track["start_angle"]),
    )

    mpc = IterativeMPC(mpc_config)
    u_prev = [[0.0, 0.0]] * horizon
    step = 0

    while viewer.is_running and step < int(sim_config["simulation"].get("max_steps", 200)):
        x_pos = data.qpos[0]
        y_pos = data.qpos[1]
        yaw = data.qpos[2]
        speed = data.qvel[0]
        x_state = [x_pos, y_pos, yaw, float(speed)]

        ref_index = min(step, len(reference) - horizon - 1)
        ref_segment = reference[ref_index : ref_index + horizon + 1]
        x_pred, u_opt = mpc.solve(x_state, ref_segment, u_init=u_prev)
        u_cmd = u_opt[0]

        data.ctrl[0] = float(u_cmd[0])
        data.ctrl[1] = float(u_cmd[1])

        mujoco.mj_step(model, data)
        viewer.render()
        time.sleep(dt)
        u_prev = list(u_opt)
        step += 1

    viewer.close()


if __name__ == "__main__":
    main()
