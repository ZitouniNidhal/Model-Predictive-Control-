import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from mpc_python.cvxpy_mpc import (
    IterativeMPC,
    build_circular_reference,
    build_waypoint_reference,
    load_yaml,
    bicycle_model,
)


def main():
    parser = argparse.ArgumentParser(description="Headless MPC no-simulation demo.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
    parser.add_argument("--reference-mode", choices=["circle", "waypoints"], default=None, help="Override reference generation mode.")
    parser.add_argument("--obstacle-avoidance", action="store_true", help="Enable linearized obstacle constraints.")
    parser.add_argument("--start-offset", type=float, default=-0.5, help="Lateral offset from the initial reference point.")
    parser.add_argument("--start-speed", type=float, default=0.2, help="Initial vehicle speed.")
    parser.add_argument("--save-log", type=str, default=None, help="Optional CSV file path to save state, control, and error history.")
    args = parser.parse_args()

    mpc_config = load_yaml(args.config)
    sim_config = load_yaml(args.simulation)

    horizon = int(mpc_config["horizon"])
    dt = float(mpc_config["dt"])
    max_steps = int(sim_config["simulation"].get("max_steps", 200))

    reference = generate_reference(sim_config, max_steps + horizon, dt, args.reference_mode)

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

    if args.save_log:
        save_history_csv(history, args.save_log)

    plot_results(history, sim_config, args.obstacle_avoidance)


def generate_reference(sim_config, n_points, dt, override_mode=None):
    reference_config = sim_config.get("reference", {})
    reference_type = override_mode or reference_config.get("type", "circle")
    speed = float(reference_config.get("speed", sim_config.get("track", {}).get("speed", 1.5)))

    if reference_type == "waypoints":
        waypoints = reference_config.get("waypoints", [])
        if not waypoints:
            raise ValueError("Waypoints reference mode requires `reference.waypoints` in simulation.yaml")
        reference = build_waypoint_reference(waypoints, speed, dt)
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


def save_history_csv(history, filename):
    headers = [
        "step",
        "x",
        "y",
        "yaw",
        "velocity",
        "acceleration",
        "steering",
        "xref_x",
        "xref_y",
        "xref_yaw",
        "xref_velocity",
        "tracking_error",
    ]

    rows = []
    for idx, state in enumerate(history["x"][:-1]):
        control = history["u"][idx] if idx < len(history["u"]) else [np.nan, np.nan]
        ref = history["xref"][idx]
        error = history["error"][idx] if idx < len(history["error"]) else np.nan
        rows.append([
            idx,
            state[0],
            state[1],
            state[2],
            state[3],
            control[0],
            control[1],
            ref[0],
            ref[1],
            ref[2],
            ref[3],
            error,
        ])

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved MPC history to {filename}")


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
