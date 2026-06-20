import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Headless MPC no-simulation demo.")
    parser.add_argument("--config", default="config/mpc.yaml", help="Path to MPC configuration")
    parser.add_argument("--simulation", default="config/simulation.yaml", help="Path to simulation configuration")
    parser.add_argument(
        "--reference-mode",
        choices=["circle", "waypoints", "figure8", "lane_change"],
        default=None,
        help="Override reference generation mode (circle, waypoints, figure8, lane_change).",
    )
    parser.add_argument("--obstacle-avoidance", action="store_true", help="Enable linearized obstacle constraints.")
    parser.add_argument("--start-offset", type=float, default=-0.5, help="Lateral offset from the initial reference point.")
    parser.add_argument("--start-speed", type=float, default=0.2, help="Initial vehicle speed.")
    parser.add_argument("--save-log", type=str, default=None, help="Optional CSV file path to save state, control, and error history.")
    parser.add_argument("--animate", action="store_true", help="Show real-time animation of the MPC controller.")
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print a formatted performance summary table after simulation.",
    )
    args = parser.parse_args()

    # Lazy imports to avoid requiring CVXPY at module-import time
    from mpc_python.cvxpy_mpc.cvxpy_mpc import IterativeMPC
    from mpc_python.cvxpy_mpc.utils import load_yaml, bicycle_model
    from mpc_python.cvxpy_mpc.metrics import summarize

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

    # Initialize animation if requested
    if args.animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(reference[:, 0], reference[:, 1], "k--", label="Reference Path", alpha=0.5)

        # Plot obstacles
        obstacles_list = sim_config.get("obstacles", [])
        moving_circles = []
        moving_obs_data = []
        for obs in obstacles_list:
            if obs.get("type") == "static":
                circle = plt.Circle(obs["position"], obs["radius"], color="r", alpha=0.3)
                ax.add_patch(circle)
                ax.text(obs["position"][0], obs["position"][1], "static", color="r", ha="center", va="center", fontsize=8)
            elif obs.get("type") == "moving":
                circle = plt.Circle(obs["start"], obs["radius"], color="orange", alpha=0.3)
                ax.add_patch(circle)
                moving_circles.append(circle)
                moving_obs_data.append(obs)

        traj_line, = ax.plot([], [], "b-", linewidth=2, label="Vehicle Trajectory")
        horizon_line, = ax.plot([], [], "g-o", linewidth=1.5, label="MPC Horizon")
        car_marker, = ax.plot([], [], "ro", markersize=8, label="Vehicle")

        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()
        ax.set_title("MPC Real-Time Animation")

    for step in range(max_steps):
        ref_segment = reference[step : step + horizon + 1]
        if ref_segment.shape[0] < horizon + 1:
            break

        obstacles = None
        if args.obstacle_avoidance:
            obstacles = build_horizon_obstacle_sequence(sim_config, step, horizon, dt)

        x_pred, u_opt = mpc.solve(x, ref_segment, u_init=u_prev, obstacles=obstacles)
        if u_opt is None or len(u_opt) == 0:
            # Fallback action: apply max deceleration, straight steering
            a_min = float(mpc_config.get("constraints", {}).get("a_min", -3.0))
            u_cmd = np.array([a_min, 0.0], dtype=float)
            x_pred = None
            print(f"Warning: MPC failed to find a solution at step {step}. Applying safety fallback deceleration command: {u_cmd}")
            u_prev = np.zeros((horizon, 2), dtype=float)
        else:
            u_cmd = np.asarray(u_opt[0], dtype=float)
            u_prev = np.vstack([u_opt[1:], u_opt[-1:]]) if len(u_opt) > 1 else np.array(u_opt, dtype=float)

        x = bicycle_model(x, u_cmd, dt, float(mpc_config.get("wheelbase", 0.16)))

        history["x"].append(x.copy())
        history["u"].append(u_cmd.copy())
        history["xref"].append(ref_segment[1].copy())
        history["error"].append(np.linalg.norm(x[:2] - ref_segment[1, :2]))

        if args.animate:
            # Update trajectory lines
            x_history = np.array(history["x"])
            traj_line.set_data(x_history[:, 0], x_history[:, 1])
            if x_pred is not None:
                horizon_line.set_data(x_pred[:, 0], x_pred[:, 1])
            car_marker.set_data([x[0]], [x[1]])

            # Update moving obstacles centers
            for circle, obs in zip(moving_circles, moving_obs_data):
                start = np.asarray(obs["start"], dtype=float)
                velocity = np.asarray(obs["velocity"], dtype=float)
                current_center = start + velocity * (step * dt)
                circle.set_center(current_center)

            # Auto-zoom/pan following the vehicle
            ax.set_xlim(x[0] - 8, x[0] + 8)
            ax.set_ylim(x[1] - 8, x[1] + 8)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        if (step + 1) % int(sim_config["simulation"].get("log_interval", 10)) == 0:
            print(
                f"Step {step + 1:3d} | x={x[0]:.2f},{x[1]:.2f} yaw={x[2]:.2f} v={x[3]:.2f} error={history['error'][-1]:.3f}"
            )

    if args.animate:
        plt.ioff()
        plt.close(fig)

    if args.save_log:
        save_history_csv(history, args.save_log)

    if args.metrics:
        constraints_cfg = mpc_config.get("constraints", {})
        summarize(history, constraints_cfg)

    plot_results(history, sim_config, args.obstacle_avoidance)


def generate_reference(sim_config, n_points, dt, override_mode=None):
    # Lazy import of reference builders to keep module import lightweight
    from mpc_python.cvxpy_mpc.utils import (
        build_waypoint_reference,
        build_circular_reference,
        build_figure8_reference,
        build_lane_change_reference,
    )

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
    n_rows = max(len(history["x"]), len(history["xref"]))
    for idx in range(n_rows):
        state = history["x"][idx] if idx < len(history["x"]) else [np.nan, np.nan, np.nan, np.nan]
        control = history["u"][idx] if idx < len(history["u"]) else [np.nan, np.nan]
        ref = history["xref"][idx] if idx < len(history["xref"]) else [np.nan, np.nan, np.nan, np.nan]
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

    fig = plt.figure(figsize=(12, 18))

    # --- Panel 1: Trajectory ---
    ax0 = fig.add_subplot(411)
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
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.legend()

    # --- Panel 2: Control Inputs ---
    ax1 = fig.add_subplot(412)
    steps = np.arange(len(u))
    ax1.plot(steps, u[:, 0], label="acceleration a (m/s²)")
    ax1.plot(steps, u[:, 1], label="steering δ (rad)")
    ax1.grid(True)
    ax1.set_title("Control Inputs")
    ax1.set_xlabel("step")
    ax1.set_ylabel("value")
    ax1.legend()

    # --- Panel 3: Tracking Error ---
    ax2 = fig.add_subplot(413)
    ax2.plot(error, label="tracking error (m)", color="crimson")
    ax2.grid(True)
    ax2.set_title("Tracking Error")
    ax2.set_xlabel("step")
    ax2.set_ylabel("error (m)")
    ax2.legend()

    # --- Panel 4: Velocity Profile ---
    ax3 = fig.add_subplot(414)
    if x.ndim == 2 and x.shape[1] >= 4:
        ax3.plot(x[:, 3], label="actual speed (m/s)", color="steelblue")
    if xref.ndim == 2 and xref.shape[1] >= 4:
        ax3.plot(xref[:, 3], "--", label="reference speed (m/s)", color="gray", alpha=0.7)
    ax3.grid(True)
    ax3.set_title("Velocity Profile")
    ax3.set_xlabel("step")
    ax3.set_ylabel("speed (m/s)")
    ax3.legend()

    plt.tight_layout()
    title = "Iterative MPC with Obstacle Avoidance" if obstacle_avoidance else "Iterative MPC Tracking"
    fig.suptitle(title, y=1.01)
    plt.show()


if __name__ == "__main__":
    main()
