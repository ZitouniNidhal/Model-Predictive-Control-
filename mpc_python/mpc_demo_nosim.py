import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mpc_python.cvxpy_mpc import (
    IterativeMPC,
    build_circular_reference,
    load_yaml,
    bicycle_model,
    wrap_angle,
)


def main():
    parser = argparse.ArgumentParser(description="Headless MPC no-simulation demo.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
    parser.add_argument("--obstacle-avoidance", action="store_true", help="Enable linearized obstacle constraints.")
    parser.add_argument("--start-offset", type=float, default=-0.5, help="Lateral offset from the initial reference point.")
    parser.add_argument("--start-speed", type=float, default=0.2, help="Initial vehicle speed.")
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
    x = np.array(
        [
            reference[0, 0] + args.start_offset,
            reference[0, 1] - args.start_offset,
            reference[0, 2],
            args.start_speed,
        ]
    )
    u_prev = np.zeros((horizon, 2), dtype=float)

    history = {
        "x": [x.copy()],
        "u": [],
        "xref": [reference[0].copy()],
        "error": [],
    }

    for step in range(max_steps):
        ref_segment = reference[step : step + horizon + 1]
        if ref_segment.shape[0] < horizon + 1:
            break

        obstacles = None
        if args.obstacle_avoidance:
            obstacles = build_horizon_obstacle_sequence(sim_config, step, horizon, dt)

        x_pred, u_opt = mpc.solve(x, ref_segment, u_init=u_prev, obstacles=obstacles)
        if u_opt is None or len(u_opt) == 0:
            break

        u_cmd = u_opt[0]
        x = bicycle_model(x, u_cmd, dt, float(mpc_config.get("wheelbase", 0.16)))
        u_prev = np.vstack([u_opt[1:], u_opt[-1:]])

        history["x"].append(x.copy())
        history["u"].append(u_cmd.copy())
        history["xref"].append(ref_segment[1].copy())
        history["error"].append(np.linalg.norm(x[:2] - ref_segment[1, :2]))

        if (step + 1) % int(sim_config["simulation"].get("log_interval", 10)) == 0:
            print(
                f"Step {step + 1:3d} | x={x[0]:.2f},{x[1]:.2f} yaw={x[2]:.2f} v={x[3]:.2f} error={history['error'][-1]:.3f}"
            )

    plot_results(history, sim_config, args.obstacle_avoidance)


def build_horizon_obstacle_sequence(sim_config, step, horizon, dt):
    obstacles = sim_config.get("obstacles", [])
    static_obs = [o for o in obstacles if o.get("type") == "static"]
    moving_obs = [o for o in obstacles if o.get("type") == "moving"]

    horizon_sequence = []
    for k in range(horizon + 1):
        t = (step + k) * dt
        step_obstacles = []

        for obs in static_obs:
            step_obstacles.append(
                {"center": obs["position"], "radius": float(obs["radius"])}
            )

        for obs in moving_obs:
            start = np.asarray(obs["start"], dtype=float)
            velocity = np.asarray(obs["velocity"], dtype=float)
            center = (start + velocity * t).tolist()
            step_obstacles.append({"center": center, "radius": float(obs["radius"])})

        horizon_sequence.append(step_obstacles)

    return horizon_sequence


def plot_results(history, sim_config, obstacle_avoidance):
    x = np.array(history["x"])
    xref = np.array(history["xref"])
    u = np.array(history["u"])
    error = np.array(history["error"])
    obstacles = sim_config.get("obstacles", [])

    fig = plt.figure(figsize=(12, 14))
    ax0 = fig.add_subplot(311)
    ax0.plot(x[:, 0], x[:, 1], label="MPC trajectory", linewidth=2)
    ax0.plot(xref[:, 0], xref[:, 1], "--", label="Reference path", linewidth=1)

    for obs in obstacles:
        if obs.get("type") == "static":
            center = obs["position"]
            radius = obs["radius"]
            circle = plt.Circle(center, radius, color="r", alpha=0.25)
            ax0.add_patch(circle)
            ax0.text(center[0], center[1], "static", color="r", ha="center", va="center")
        elif obs.get("type") == "moving":
            start = obs["start"]
            radius = obs["radius"]
            circle = plt.Circle(start, radius, color="orange", alpha=0.25)
            ax0.add_patch(circle)
            ax0.text(start[0], start[1], "moving start", color="orange", ha="center", va="center")

    ax0.set_aspect("equal")
    ax0.grid(True)
    ax0.set_title("Iterative MPC Trajectory")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend()

    ax1 = fig.add_subplot(312)
    ax1.plot(u[:, 0], label="acceleration (a)")
    ax1.plot(u[:, 1], label="steering (delta)")
    ax1.grid(True)
    ax1.set_title("Control Inputs")
    ax1.set_xlabel("step")
    ax1.legend()

    ax2 = fig.add_subplot(313)
    ax2.plot(error, label="tracking error")
    ax2.grid(True)
    ax2.set_title("Tracking Error")
    ax2.set_xlabel("step")
    ax2.legend()

    plt.tight_layout()
    if obstacle_avoidance:
        fig.suptitle("Iterative MPC with Obstacle Avoidance", y=1.02)
    else:
        fig.suptitle("Iterative MPC Tracking", y=1.02)
    plt.show()


if __name__ == "__main__":
    main()
