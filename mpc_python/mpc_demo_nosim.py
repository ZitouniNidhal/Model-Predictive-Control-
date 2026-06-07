import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mpc_python.cvxpy_mpc import IterativeMPC, build_circular_reference, load_yaml, bicycle_model


def main():
    parser = argparse.ArgumentParser(description="Headless MPC no-simulation demo.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
    args = parser.parse_args()

    mpc_config = load_yaml(args.config)
    sim_config = load_yaml(args.simulation)

    track = sim_config["track"]
    horizon = int(mpc_config["horizon"])
    dt = float(mpc_config["dt"])
    max_steps = int(sim_config["simulation"].get("max_steps", 200))

    reference = build_circular_reference(
        n_points=max_steps + horizon,
        radius=float(track["radius"]),
        speed=float(track["speed"]),
        dt=dt,
        center=tuple(track["center"]),
        start_angle=float(track["start_angle"]),
    )

    mpc = IterativeMPC(mpc_config)
    x = np.array([reference[0, 0], reference[0, 1] - 0.5, reference[0, 2], 0.2])
    u_prev = np.zeros((horizon, 2), dtype=float)

    history = {
        "x": [x.copy()],
        "u": [],
        "xref": [reference[0].copy()],
    }

    for step in range(max_steps):
        ref_segment = reference[step : step + horizon + 1]
        if ref_segment.shape[0] < horizon + 1:
            break

        x_pred, u_opt = mpc.solve(x, ref_segment, u_prev=u_prev)
        u_cmd = u_opt[0]
        x = bicycle_model(x, u_cmd, dt, float(mpc_config.get("wheelbase", 0.16)))
        u_prev = np.vstack([u_opt[1:], u_opt[-1:]])

        history["x"].append(x.copy())
        history["u"].append(u_cmd.copy())
        history["xref"].append(ref_segment[1].copy())

        if (step + 1) % int(sim_config["simulation"].get("log_interval", 10)) == 0:
            print(f"Step {step + 1:3d} | x={x[0]:.2f},{x[1]:.2f} yaw={x[2]:.2f} v={x[3]:.2f}")

    plot_results(history, sim_config)


def plot_results(history, sim_config):
    x = np.array(history["x"])
    xref = np.array(history["xref"])
    obstacles = sim_config.get("obstacles", [])

    plt.figure(figsize=(10, 8))
    plt.plot(x[:, 0], x[:, 1], label="MPC trajectory", linewidth=2)
    plt.plot(xref[:, 0], xref[:, 1], "--", label="Reference path", linewidth=1)

    for obs in obstacles:
        if obs.get("type") == "static":
            center = obs["position"]
            radius = obs["radius"]
            circle = plt.Circle(center, radius, color="r", alpha=0.25)
            plt.gca().add_patch(circle)
            plt.text(center[0], center[1], "obs", color="r", ha="center", va="center")

    plt.axis("equal")
    plt.grid(True)
    plt.title("Iterative MPC Tracking - No Simulation Demo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
