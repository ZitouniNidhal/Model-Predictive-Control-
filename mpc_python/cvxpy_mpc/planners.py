"""High-level MPC planning helpers.

This module provides :class:`MPCRunner`, a convenience wrapper around
:class:`~mpc_python.cvxpy_mpc.IterativeMPC` that bundles the simulation
loop, warm-start shifting, and history recording into a single reusable
object.  This simplifies both demo scripts and test code.
"""

import logging
from typing import List, Optional, Union

import numpy as np

from .cvxpy_mpc import IterativeMPC
from .utils import bicycle_model

logger = logging.getLogger(__name__)

# Type alias mirroring cvxpy_mpc
ObstacleDict = dict


class MPCRunner:
    """High-level runner that wraps :class:`IterativeMPC` with a simulation loop.

    Manages warm-start shifting, state propagation, fallback handling, and
    history recording.  After calling :meth:`run`, the recorded ``history``
    attribute is ready for plotting or metrics analysis.

    Args:
        mpc: A configured :class:`IterativeMPC` instance.
        reference: Reference trajectory array of shape ``(M, 4)`` where
            ``M >= n_steps + horizon + 1``.
        x0: Initial vehicle state ``[x, y, yaw, v]``.
        wheelbase: Vehicle wheelbase in metres. Defaults to ``0.16``.

    Attributes:
        history (dict): Recorded simulation history with keys
            ``"x"``, ``"u"``, ``"xref"``, ``"error"``.
    """

    def __init__(
        self,
        mpc: IterativeMPC,
        reference: np.ndarray,
        x0: Union[list, np.ndarray],
        wheelbase: float = 0.16,
    ) -> None:
        self.mpc = mpc
        self.reference = np.asarray(reference, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.wheelbase = float(wheelbase)
        self.horizon = mpc.horizon
        self.dt = mpc.dt

        self.history: dict = {
            "x": [self.x.copy()],
            "u": [],
            "xref": [self.reference[0].copy()],
            "error": [],
        }
        self._u_prev = np.zeros((self.horizon, 2), dtype=float)

    def run(
        self,
        n_steps: int,
        obstacles: Optional[Union[List[ObstacleDict], List[List[ObstacleDict]]]] = None,
    ) -> dict:
        """Run the MPC simulation loop for *n_steps* steps.

        Args:
            n_steps: Number of simulation steps to execute.
            obstacles: Optional obstacle specification forwarded to
                :meth:`IterativeMPC.solve` at every step. Pass a callable
                ``obstacles(step)`` to supply per-step obstacle sequences.

        Returns:
            The ``history`` dict (also stored in :attr:`history`).
        """
        for step in range(n_steps):
            ref_segment = self.reference[step : step + self.horizon + 1]
            if ref_segment.shape[0] < self.horizon + 1:
                logger.warning("Reference too short at step %d; stopping early.", step)
                break

            # Resolve per-step obstacles
            if callable(obstacles):
                step_obs = obstacles(step)
            else:
                step_obs = obstacles

            x_pred, u_opt = self.mpc.solve(
                self.x, ref_segment, u_init=self._u_prev, obstacles=step_obs
            )

            if u_opt is None or len(u_opt) == 0:
                a_min = self.mpc.a_min
                u_cmd = np.array([a_min, 0.0], dtype=float)
                logger.warning(
                    "MPC failed at step %d; applying safety fallback a=%.2f, delta=0.", step, a_min
                )
                self._u_prev = np.zeros((self.horizon, 2), dtype=float)
            else:
                u_cmd = np.asarray(u_opt[0], dtype=float)
                self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]]) if len(u_opt) > 1 else np.array(u_opt)

            self.x = bicycle_model(self.x, u_cmd, self.dt, self.wheelbase)

            self.history["x"].append(self.x.copy())
            self.history["u"].append(u_cmd.copy())
            self.history["xref"].append(ref_segment[1].copy())
            self.history["error"].append(float(np.linalg.norm(self.x[:2] - ref_segment[1, :2])))

            if (step + 1) % max(1, n_steps // 10) == 0:
                logger.info(
                    "Step %3d | x=%.2f,%.2f yaw=%.2f v=%.2f error=%.3f",
                    step + 1, self.x[0], self.x[1], self.x[2], self.x[3],
                    self.history["error"][-1],
                )

        return self.history

    def reset(self, x0: Optional[Union[list, np.ndarray]] = None) -> None:
        """Reset internal state for a fresh run.

        Args:
            x0: New initial state. If ``None``, uses the original ``x0``
                passed at construction (i.e. ``self.reference[0]``).
        """
        if x0 is not None:
            self.x = np.asarray(x0, dtype=float).copy()
        else:
            self.x = np.asarray(self.reference[0], dtype=float).copy()
        self._u_prev = np.zeros((self.horizon, 2), dtype=float)
        self.history = {
            "x": [self.x.copy()],
            "u": [],
            "xref": [self.reference[0].copy()],
            "error": [],
        }
