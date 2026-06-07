import math
from pathlib import Path
import numpy as np
import yaml


def load_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bicycle_model(state, control, dt, wheelbase):
    x, y, yaw, v = state
    a, delta = control
    x_next = x + v * math.cos(yaw) * dt
    y_next = y + v * math.sin(yaw) * dt
    yaw_next = yaw + v / wheelbase * math.tan(delta) * dt
    v_next = v + a * dt
    return np.array([x_next, y_next, wrap_angle(yaw_next), v_next], dtype=float)


def linearize_dynamics(state, control, dt, wheelbase):
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


def build_circular_reference(n_points, radius, speed, dt, center=(0.0, 0.0), start_angle=0.0):
    reference = []
    for k in range(n_points + 1):
        theta = start_angle + (speed / radius) * k * dt
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        yaw = wrap_angle(theta + math.pi / 2)
        reference.append([x, y, yaw, speed])
    return np.array(reference, dtype=float)
