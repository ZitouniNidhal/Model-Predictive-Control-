"""Utility functions for the MPC bicycle-model controller.

This module provides:
- YAML loading
- Angle wrapping
- Kinematic bicycle model simulation
- Jacobian linearization of dynamics and obstacle constraints
- Reference trajectory builders (circular, waypoint, figure-8, lane-change)
- Tracking performance metrics helper
"""

import logging
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    """Load and return a YAML configuration file from *path*.

    Args:
        path: Absolute or relative path to the ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed contents as a Python ``dict``.

    Raises:
        FileNotFoundError: If *path* does not point to an existing file.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    logger.debug("Loading YAML configuration from '%s'", path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    logger.info("Loaded configuration: %s", path)
    return config


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def wrap_angle(angle: float) -> float:
    """Wrap *angle* (in radians) to the interval ``[-π, π]``.

    Args:
        angle: Input angle in radians (any finite value).

    Returns:
        Equivalent angle in ``[-π, π]``.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def normalize_vector(vector: Union[Sequence[float], np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Return a unit-length 2-D vector, with a safe fallback for near-zero magnitude.

    Args:
        vector: A 2-element array-like ``[vx, vy]``.
        eps: Magnitude below which the vector is considered zero and the
             fallback ``[1, 0]`` is returned instead.

    Returns:
        Normalized 2-D ``np.ndarray`` of shape ``(2,)``.
    """
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"normalize_vector expects a 2-element array, got shape {vector.shape}.")
    norm = np.linalg.norm(vector)
    if norm < eps:
        logger.debug("Near-zero vector magnitude (%.2e); returning unit fallback [1, 0].", norm)
        return np.array([1.0, 0.0], dtype=float)
    return vector / norm


# ---------------------------------------------------------------------------
# Bicycle kinematic model
# ---------------------------------------------------------------------------

def bicycle_model(
    state: Union[Sequence[float], np.ndarray],
    control: Union[Sequence[float], np.ndarray],
    dt: float,
    wheelbase: float,
) -> np.ndarray:
    """Integrate the kinematic bicycle model by one time step *dt*.

    State vector: ``[x, y, yaw, v]``
    Control vector: ``[a, delta]``  (acceleration, steering angle in rad)

    Args:
        state: Current state ``[x, y, yaw, v]``.
        control: Control input ``[a, delta]``.
        dt: Integration step size (seconds). Must be > 0.
        wheelbase: Distance between axles (metres). Must be > 0.

    Returns:
        Next state ``[x_next, y_next, yaw_next, v_next]`` as ``np.ndarray``.

    Raises:
        ValueError: If *dt* or *wheelbase* are non-positive.
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")
    if wheelbase <= 0.0:
        raise ValueError(f"Wheelbase must be positive, got {wheelbase}.")

    x, y, yaw, v = state
    a, delta = control
    x_next = x + v * math.cos(yaw) * dt
    y_next = y + v * math.sin(yaw) * dt
    yaw_next = yaw + v / wheelbase * math.tan(delta) * dt
    v_next = v + a * dt
    return np.array([x_next, y_next, wrap_angle(yaw_next), v_next], dtype=float)


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def linearize_dynamics(
    state: Union[Sequence[float], np.ndarray],
    control: Union[Sequence[float], np.ndarray],
    dt: float,
    wheelbase: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the first-order Taylor (Jacobian) linearization of the bicycle model.

    Returns the discrete-time matrices ``A`` (4×4) and ``B`` (4×2) such that::

        x_{k+1} ≈ A @ x_k + B @ u_k + c

    where ``c`` absorbs the linearization residual.

    Args:
        state: Linearization point ``[x, y, yaw, v]``.
        control: Linearization control ``[a, delta]``.
        dt: Integration step size (seconds).
        wheelbase: Vehicle wheelbase (metres).

    Returns:
        Tuple ``(A, B)`` — state transition and input Jacobian matrices.
    """
    _, _, yaw, v = state
    _, delta = control
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_delta = math.cos(delta)
    cos_delta_sq = cos_delta * cos_delta

    A = np.eye(4, dtype=float)
    A[0, 2] = -v * sin_yaw * dt
    A[0, 3] = cos_yaw * dt
    A[1, 2] = v * cos_yaw * dt
    A[1, 3] = sin_yaw * dt
    A[2, 3] = math.tan(delta) / wheelbase * dt

    B = np.zeros((4, 2), dtype=float)
    B[0, 1] = -v * sin_yaw * dt
    B[1, 1] = v * cos_yaw * dt
    B[2, 1] = v / (wheelbase * cos_delta_sq) * dt
    B[3, 0] = dt

    return A, B


def linearize_obstacle_constraint(
    x_prev: Union[Sequence[float], np.ndarray],
    center: Union[Sequence[float], np.ndarray],
    radius: float,
    margin: float = 0.25,
) -> Tuple[np.ndarray, float]:
    """Linearize a circular obstacle avoidance constraint around *x_prev*.

    The linearized half-plane constraint has the form::

        direction · [x, y] ≥ threshold

    Args:
        x_prev: Current vehicle state ``[x, y, yaw, v]`` (only x, y used).
        center: Obstacle center ``[cx, cy]``.
        radius: Obstacle radius (metres). Must be ≥ 0.
        margin: Additional safety margin (metres). Defaults to ``0.25``.

    Returns:
        Tuple ``(direction, threshold)`` where *direction* is a normalized
        unit vector pointing from the obstacle centre to the vehicle, and
        *threshold* is the scalar RHS of the half-plane constraint.

    Raises:
        ValueError: If *radius* is negative.
    """
    if radius < 0.0:
        raise ValueError(f"Obstacle radius must be non-negative, got {radius}.")
    center = np.asarray(center, dtype=float)
    position = np.asarray(x_prev[:2], dtype=float)
    direction = normalize_vector(position - center)
    threshold = float(center.dot(direction)) + radius + margin
    return direction, threshold


# ---------------------------------------------------------------------------
# Reference trajectory builders
# ---------------------------------------------------------------------------

def build_circular_reference(
    n_points: int,
    radius: float,
    speed: float,
    dt: float,
    center: Tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
) -> np.ndarray:
    """Build a circular reference trajectory for the vehicle.

    Args:
        n_points: Number of trajectory steps (exclusive of the final duplicate).
        radius: Circle radius (metres). Must be > 0.
        speed: Constant reference speed (m/s). Must be > 0.
        dt: Time step (seconds). Must be > 0.
        center: Centre of the circle as ``(cx, cy)``. Defaults to origin.
        start_angle: Starting angle in radians. Defaults to ``0.0``.

    Returns:
        Array of shape ``(n_points + 1, 4)`` with columns ``[x, y, yaw, v]``.

    Raises:
        ValueError: If *radius*, *speed*, or *dt* are non-positive.
    """
    if radius <= 0.0:
        raise ValueError(f"Circular reference radius must be positive, got {radius}.")
    if speed <= 0.0:
        raise ValueError(f"Reference speed must be positive, got {speed}.")
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")

    logger.debug(
        "Building circular reference: n_points=%d, radius=%.2f, speed=%.2f, dt=%.3f",
        n_points, radius, speed, dt,
    )
    reference = []
    for k in range(n_points + 1):
        theta = start_angle + (speed / radius) * k * dt
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        yaw = wrap_angle(theta + math.pi / 2)
        reference.append([x, y, yaw, speed])
    return np.array(reference, dtype=float)


def build_waypoint_reference(
    waypoints: Union[Sequence[Sequence[float]], np.ndarray],
    speed: float,
    dt: float,
    n_points: Optional[int] = None,
    a_lat_max: float = 1.5,
) -> np.ndarray:
    """Build a reference trajectory by interpolating through *waypoints* with cubic splines.

    A curvature-aware speed profile is computed so that the vehicle slows down
    in tight corners (lateral acceleration ≤ *a_lat_max*), while always
    maintaining at least 30 % of the target *speed*.

    Args:
        waypoints: Sequence of ``[x, y]`` pairs defining the path.
        speed: Desired cruising speed (m/s). Must be > 0.
        dt: Time step (seconds). Must be > 0.
        n_points: Number of uniformly-spaced sample points along the path.
            Defaults to ``ceil(total_length / (speed * dt))``.
        a_lat_max: Maximum allowed lateral acceleration (m/s²). Controls
            how much speed is reduced in corners. Defaults to ``1.5``.

    Returns:
        Array of shape ``(n_points + 1, 4)`` with columns ``[x, y, yaw, v]``.

    Raises:
        ValueError: If *waypoints* are not a 2-D array of ``[x, y]`` pairs,
            or if *speed* / *dt* are non-positive.
    """
    if speed <= 0.0:
        raise ValueError(f"Reference speed must be positive, got {speed}.")
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")

    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Waypoints must be a list of [x, y] pairs (shape N×2), got shape {points.shape}."
        )

    if points.shape[0] < 2:
        logger.warning("Only one unique waypoint provided; returning a single-row reference.")
        return np.array([[points[0, 0], points[0, 1], 0.0, speed]], dtype=float)

    # Remove near-duplicate consecutive points to avoid singular spline parameterisation
    filtered: list = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - filtered[-1]) > 1e-4:
            filtered.append(p)
    points = np.array(filtered, dtype=float)

    if points.shape[0] < 2:
        logger.warning("All waypoints collapsed to a single point after deduplication.")
        return np.array([[points[0, 0], points[0, 1], 0.0, speed]], dtype=float)

    segments = np.diff(points, axis=0)
    distances = np.linalg.norm(segments, axis=1)
    total_length = float(np.sum(distances))

    if total_length <= 0.0:
        return np.array([[points[0, 0], points[0, 1], 0.0, speed]], dtype=float)

    cumulative = np.concatenate([[0.0], np.cumsum(distances)])

    # Cubic splines parametrised by arc length
    cs_x = CubicSpline(cumulative, points[:, 0], bc_type="natural")
    cs_y = CubicSpline(cumulative, points[:, 1], bc_type="natural")

    if n_points is None:
        n_points = int(max(1, math.ceil(total_length / (speed * dt))))

    logger.debug(
        "Building waypoint reference: total_length=%.2f, n_points=%d, speed=%.2f",
        total_length, n_points, speed,
    )

    sample_distances = np.linspace(0.0, total_length, n_points + 1)
    reference = []
    for s in sample_distances:
        x = float(cs_x(s))
        y = float(cs_y(s))
        dx = float(cs_x(s, 1))
        dy = float(cs_y(s, 1))
        ddx = float(cs_x(s, 2))
        ddy = float(cs_y(s, 2))

        heading = math.atan2(dy, dx)

        # Curvature κ = |x'y'' − y'x''| / (x'² + y'²)^(3/2)
        denom = (dx ** 2 + dy ** 2) ** 1.5
        kappa = abs(dx * ddy - dy * ddx) / denom if denom > 1e-6 else 0.0

        # Lateral-acceleration-limited speed profile
        v_ref = min(speed, math.sqrt(a_lat_max / kappa)) if kappa > 1e-3 else speed
        v_ref = max(0.3 * speed, v_ref)  # always keep ≥ 30 % of target speed

        reference.append([x, y, wrap_angle(heading), v_ref])

    return np.array(reference, dtype=float)


def build_figure8_reference(
    n_points: int,
    speed: float,
    dt: float,
    size_x: float = 12.0,
    size_y: float = 6.0,
) -> np.ndarray:
    """Build a figure-8 (Lissajous) reference trajectory.

    The trajectory is generated by sampling a parametric Lissajous curve and
    then delegating to :func:`build_waypoint_reference` for arc-length
    parametrisation and speed profiling.

    Args:
        n_points: Number of uniformly-spaced trajectory steps.
        speed: Desired cruising speed (m/s). Must be > 0.
        dt: Time step (seconds). Must be > 0.
        size_x: Half-amplitude of the figure-8 along the x-axis. Defaults to ``12.0``.
        size_y: Half-amplitude of the figure-8 along the y-axis. Defaults to ``6.0``.

    Returns:
        Array of shape ``(n_points + 1, 4)`` with columns ``[x, y, yaw, v]``.
    """
    logger.debug(
        "Building figure-8 reference: n_points=%d, speed=%.2f, size_x=%.1f, size_y=%.1f",
        n_points, speed, size_x, size_y,
    )
    theta = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
    waypoints = [[size_x * math.sin(t), size_y * math.sin(2 * t)] for t in theta]

    # Repeat to guarantee sufficient arc length for the requested number of points
    waypoints = waypoints * 5
    waypoints.append(waypoints[0])

    return build_waypoint_reference(waypoints, speed, dt, n_points=n_points)


def build_lane_change_reference(
    n_points: int,
    speed: float,
    dt: float,
    lane_width: float = 3.5,
    straight_length: float = 20.0,
    transition_length: float = 15.0,
) -> np.ndarray:
    """Build a smooth double-lane-change (S-curve) reference trajectory.

    The path consists of three sections:
      1. A straight initial approach.
      2. A ``tanh``-blended lateral transition of width *lane_width*.
      3. A straight section in the new lane.

    The resulting waypoints are passed to :func:`build_waypoint_reference` for
    arc-length parametrisation, heading computation, and curvature-aware speed
    profiling.

    Args:
        n_points: Number of uniformly-spaced trajectory steps.
        speed: Desired cruising speed (m/s). Must be > 0.
        dt: Time step (seconds). Must be > 0.
        lane_width: Lateral displacement between the two lanes (metres). Defaults to ``3.5``.
        straight_length: Length of the initial and final straight sections (metres). Defaults to ``20.0``.
        transition_length: Longitudinal length of the lateral transition (metres). Defaults to ``15.0``.

    Returns:
        Array of shape ``(n_points + 1, 4)`` with columns ``[x, y, yaw, v]``.

    Raises:
        ValueError: If *lane_width*, *straight_length*, *transition_length*, *speed*, or *dt* are
            non-positive.
    """
    if speed <= 0.0:
        raise ValueError(f"Reference speed must be positive, got {speed}.")
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")
    if lane_width <= 0.0:
        raise ValueError(f"lane_width must be positive, got {lane_width}.")
    if straight_length <= 0.0:
        raise ValueError(f"straight_length must be positive, got {straight_length}.")
    if transition_length <= 0.0:
        raise ValueError(f"transition_length must be positive, got {transition_length}.")

    logger.debug(
        "Building lane-change reference: lane_width=%.2f, straight=%.1f, transition=%.1f",
        lane_width, straight_length, transition_length,
    )

    total_length = straight_length + transition_length + straight_length
    n_samples = max(200, n_points * 4)
    xs = np.linspace(0.0, total_length, n_samples)

    waypoints = []
    for x_val in xs:
        # Normalised longitudinal coordinate centred on the transition section
        x_rel = (x_val - straight_length) / transition_length
        # Smooth tanh blend: 0 at start, 1 at end of transition
        blend = 0.5 * (1.0 + math.tanh(4.0 * (x_rel - 0.5)))
        blend = float(np.clip(blend, 0.0, 1.0))
        y_val = lane_width * blend
        waypoints.append([x_val, y_val])

    return build_waypoint_reference(waypoints, speed, dt, n_points=n_points)


def compute_tracking_metrics(history: dict) -> dict:
    """Compute summary performance metrics from a recorded simulation history.

    The *history* dict is expected to have the same structure produced by
    ``mpc_demo_nosim.main()``::

        {
            "x":     list of np.ndarray([x, y, yaw, v]),   # actual states
            "xref":  list of np.ndarray([x, y, yaw, v]),   # reference states
            "u":     list of np.ndarray([a, delta]),        # applied controls
            "error": list of float,                         # positional tracking error
        }

    Args:
        history: Simulation history dictionary as described above.

    Returns:
        Dictionary with the following keys:

        * ``mean_error`` (float) — mean positional tracking error (m).
        * ``max_error`` (float) — maximum positional tracking error (m).
        * ``rms_error`` (float) — root-mean-square positional tracking error (m).
        * ``mean_speed`` (float) — mean vehicle speed over the trajectory (m/s).
        * ``total_accel_effort`` (float) — total variation of acceleration (sum of |Δa|).
        * ``total_steer_effort`` (float) — total variation of steering angle (sum of |Δδ|).
        * ``n_steps`` (int) — number of recorded simulation steps.
    """
    errors = np.asarray(history.get("error", []), dtype=float)
    states = np.asarray(history.get("x", []), dtype=float)
    controls = np.asarray(history.get("u", []), dtype=float)

    n_steps = int(len(errors))

    if n_steps == 0:
        return {
            "mean_error": float("nan"),
            "max_error": float("nan"),
            "rms_error": float("nan"),
            "mean_speed": float("nan"),
            "total_accel_effort": float("nan"),
            "total_steer_effort": float("nan"),
            "n_steps": 0,
        }

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    rms_error = float(np.sqrt(np.mean(errors ** 2)))

    mean_speed = float(np.mean(states[:, 3])) if states.ndim == 2 and states.shape[1] >= 4 else float("nan")

    if controls.ndim == 2 and controls.shape[0] > 1:
        delta_u = np.diff(controls, axis=0)
        total_accel_effort = float(np.sum(np.abs(delta_u[:, 0])))
        total_steer_effort = float(np.sum(np.abs(delta_u[:, 1])))
    else:
        total_accel_effort = float("nan")
        total_steer_effort = float("nan")

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "rms_error": rms_error,
        "mean_speed": mean_speed,
        "total_accel_effort": total_accel_effort,
        "total_steer_effort": total_steer_effort,
        "n_steps": n_steps,
    }
