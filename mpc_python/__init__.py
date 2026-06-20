"""mpc_python package."""

from .cvxpy_mpc import (
    IterativeMPC,
    load_yaml,
    wrap_angle,
    bicycle_model,
    linearize_dynamics,
    linearize_obstacle_constraint,
    build_circular_reference,
    build_waypoint_reference,
    build_figure8_reference,
    build_lane_change_reference,
    compute_tracking_metrics,
    tracking_error_stats,
    control_effort,
    constraint_satisfaction,
    summarize,
    MPCRunner,
)

__all__ = [
    "IterativeMPC",
    "load_yaml",
    "wrap_angle",
    "bicycle_model",
    "linearize_dynamics",
    "linearize_obstacle_constraint",
    "build_circular_reference",
    "build_waypoint_reference",
    "build_figure8_reference",
    "build_lane_change_reference",
    "compute_tracking_metrics",
    "tracking_error_stats",
    "control_effort",
    "constraint_satisfaction",
    "summarize",
    "MPCRunner",
]
