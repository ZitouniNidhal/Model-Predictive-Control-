"""Performance metrics module for the MPC bicycle-model controller.

Provides functions to evaluate trajectory quality, control effort, and
constraint satisfaction from a recorded simulation history dict.
All functions accept the ``history`` dict produced by
:func:`mpc_python.mpc_demo_nosim.main` or :class:`planners.MPCRunner`.
"""

import logging
from typing import Optional

import numpy as np

from .utils import compute_tracking_metrics

logger = logging.getLogger(__name__)


def tracking_error_stats(history: dict) -> dict:
    """Compute positional tracking error statistics.

    Args:
        history: Simulation history dict with an ``"error"`` key containing a
            list/array of per-step positional errors (metres).

    Returns:
        Dict with keys ``mean``, ``max``, ``rms``, ``n_steps``.
    """
    errors = np.asarray(history.get("error", []), dtype=float)
    n = len(errors)
    if n == 0:
        return {"mean": float("nan"), "max": float("nan"), "rms": float("nan"), "n_steps": 0}
    return {
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "rms": float(np.sqrt(np.mean(errors ** 2))),
        "n_steps": n,
    }


def control_effort(u_history) -> dict:
    """Compute total variation of control inputs as a proxy for control effort.

    Args:
        u_history: Array-like of shape ``(T, 2)`` where columns are
            ``[acceleration, steering_angle]``.

    Returns:
        Dict with keys ``total_accel`` (sum of |Δa|) and ``total_steer`` (sum of |Δδ|).
    """
    u = np.asarray(u_history, dtype=float)
    if u.ndim != 2 or u.shape[1] < 2 or u.shape[0] < 2:
        return {"total_accel": float("nan"), "total_steer": float("nan")}
    delta = np.diff(u, axis=0)
    return {
        "total_accel": float(np.sum(np.abs(delta[:, 0]))),
        "total_steer": float(np.sum(np.abs(delta[:, 1]))),
    }


def constraint_satisfaction(u_history, constraints_cfg: dict) -> dict:
    """Compute the fraction of steps where each control constraint is satisfied.

    Args:
        u_history: Array-like of shape ``(T, 2)`` — ``[a, delta]`` per step.
        constraints_cfg: Dict with keys ``a_min``, ``a_max``, ``delta_min``,
            ``delta_max`` (same structure as ``config["constraints"]``).

    Returns:
        Dict mapping each constraint name to the fraction of steps where it
        is satisfied (0.0 – 1.0), plus ``"overall"`` (all constraints jointly).
    """
    u = np.asarray(u_history, dtype=float)
    if u.ndim != 2 or u.shape[1] < 2 or u.shape[0] == 0:
        return {
            "a_min": float("nan"),
            "a_max": float("nan"),
            "delta_min": float("nan"),
            "delta_max": float("nan"),
            "overall": float("nan"),
        }

    a = u[:, 0]
    delta = u[:, 1]
    T = float(len(a))

    a_min = float(constraints_cfg.get("a_min", -3.0))
    a_max = float(constraints_cfg.get("a_max", 2.0))
    d_min = float(constraints_cfg.get("delta_min", -0.5))
    d_max = float(constraints_cfg.get("delta_max", 0.5))

    ok_a_min = np.sum(a >= a_min - 1e-6) / T
    ok_a_max = np.sum(a <= a_max + 1e-6) / T
    ok_d_min = np.sum(delta >= d_min - 1e-6) / T
    ok_d_max = np.sum(delta <= d_max + 1e-6) / T

    all_ok = np.all(
        (a >= a_min - 1e-6) & (a <= a_max + 1e-6)
        & (delta >= d_min - 1e-6) & (delta <= d_max + 1e-6)
    )
    return {
        "a_min": float(ok_a_min),
        "a_max": float(ok_a_max),
        "delta_min": float(ok_d_min),
        "delta_max": float(ok_d_max),
        "overall": float(np.mean(
            (a >= a_min - 1e-6) & (a <= a_max + 1e-6)
            & (delta >= d_min - 1e-6) & (delta <= d_max + 1e-6)
        )),
    }


def summarize(history: dict, constraints_cfg: Optional[dict] = None) -> None:
    """Print a formatted performance summary table to stdout.

    Args:
        history: Simulation history dict (same format as
            :func:`compute_tracking_metrics`).
        constraints_cfg: Optional constraint config dict (same as
            ``config["constraints"]``). If provided, constraint satisfaction
            rates are shown.
    """
    metrics = compute_tracking_metrics(history)
    u_hist = history.get("u", [])
    effort = control_effort(u_hist)

    sep = "=" * 52
    print(sep)
    print("  MPC PERFORMANCE SUMMARY")
    print(sep)
    print(f"  Steps simulated   : {metrics['n_steps']}")
    print(f"  Mean track. error : {metrics['mean_error']:.4f} m")
    print(f"  Max track. error  : {metrics['max_error']:.4f} m")
    print(f"  RMS track. error  : {metrics['rms_error']:.4f} m")
    print(f"  Mean speed        : {metrics['mean_speed']:.3f} m/s")
    print(f"  Accel effort (ΣΔa): {effort['total_accel']:.4f}")
    print(f"  Steer effort (ΣΔδ): {effort['total_steer']:.4f}")

    if constraints_cfg and u_hist:
        sat = constraint_satisfaction(u_hist, constraints_cfg)
        print("-" * 52)
        print("  Constraint satisfaction rates:")
        print(f"    a_min          : {sat['a_min']*100:.1f} %")
        print(f"    a_max          : {sat['a_max']*100:.1f} %")
        print(f"    delta_min      : {sat['delta_min']*100:.1f} %")
        print(f"    delta_max      : {sat['delta_max']*100:.1f} %")
        print(f"    overall (all)  : {sat['overall']*100:.1f} %")

    print(sep)
