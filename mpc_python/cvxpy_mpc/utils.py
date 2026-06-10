import math
import numpy as np
import yaml


def load_yaml(path):
    """Load YAML configuration from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def wrap_angle(angle):
    """Normalize an angle to the interval [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bicycle_model(state, control, dt, wheelbase):
    """Simple bicycle kinematic model for one integration step."""
    x, y, yaw, v = state
    a, delta = control
    x_next = x + v * math.cos(yaw) * dt
    y_next = y + v * math.sin(yaw) * dt
    yaw_next = yaw + v / wheelbase * math.tan(delta) * dt
    v_next = v + a * dt
    return np.array([x_next, y_next, wrap_angle(yaw_next), v_next], dtype=float)


def linearize_dynamics(state, control, dt, wheelbase):
    """Compute state and input Jacobians for the bicycle model."""
    x, y, yaw, v = state
    a, delta = control
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


def normalize_vector(vector, eps=1e-6):
    """Return a normalized 2D vector, with a safe fallback for zero magnitude."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.array([1.0, 0.0], dtype=float)
    return vector / norm


def linearize_obstacle_constraint(x_prev, center, radius, margin=0.25):
    """Linearize a circular obstacle constraint around the current state."""
    center = np.asarray(center, dtype=float)
    position = np.asarray(x_prev[:2], dtype=float)
    direction = normalize_vector(position - center)
    threshold = center.dot(direction) + radius + margin
    return direction, threshold


def build_circular_reference(n_points, radius, speed, dt, center=(0.0, 0.0), start_angle=0.0):
    """Build a circular reference trajectory for the vehicle."""
    reference = []
    for k in range(n_points + 1):
        theta = start_angle + (speed / radius) * k * dt
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        yaw = wrap_angle(theta + math.pi / 2)
        reference.append([x, y, yaw, speed])
    return np.array(reference, dtype=float)


def build_waypoint_reference(waypoints, speed, dt, n_points=None):
    """Build a reference trajectory by interpolating through waypoints."""
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Waypoints must be a list of [x, y] pairs.")
    if points.shape[0] < 2:
        yaw = 0.0
        return np.array([[points[0, 0], points[0, 1], yaw, speed]], dtype=float)

    segments = np.diff(points, axis=0)
    distances = np.linalg.norm(segments, axis=1)
    total_length = np.sum(distances)
    if total_length <= 0.0:
        yaw = math.atan2(segments[-1, 1], segments[-1, 0])
        return np.array([[points[0, 0], points[0, 1], wrap_angle(yaw), speed]], dtype=float)

    if n_points is None:
        n_points = int(max(1, math.ceil(total_length / (speed * dt))))

    sample_distances = np.linspace(0.0, total_length, n_points + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])

    reference = []
    segment_index = 0
    for s in sample_distances:
        while segment_index < len(distances) - 1 and s > cumulative[segment_index + 1]:
            segment_index += 1

        segment_dist = distances[segment_index]
        offset = s - cumulative[segment_index]
        ratio = offset / max(segment_dist, 1e-6)
        position = points[segment_index] + ratio * segments[segment_index]
        heading = math.atan2(segments[segment_index, 1], segments[segment_index, 0])
        reference.append([position[0], position[1], wrap_angle(heading), speed])

    return np.array(reference, dtype=float)
