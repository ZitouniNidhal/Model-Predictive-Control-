"""Iterative (Sequential) Linear MPC controller using CVXPY and OSQP.

The :class:`IterativeMPC` class solves a finite-horizon quadratic programme
at each control step by linearising the nonlinear bicycle-model dynamics
around the previous solution trajectory.  Obstacle avoidance is handled via
convex half-plane constraints that are linearised around the current state.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np

from .utils import bicycle_model, linearize_dynamics, linearize_obstacle_constraint

logger = logging.getLogger(__name__)

# Type alias used for an obstacle dictionary
ObstacleDict = Dict[str, Union[List[float], float]]


class IterativeMPC:
    """Iterative (SQP-style) Model Predictive Controller for a kinematic bicycle model.

    The controller solves a finite-horizon QP at each time step by
    sequentially linearising the nonlinear vehicle dynamics around the
    previous solution, iterating up to :attr:`max_iters` times per call.

    Args:
        config: Configuration dictionary with the following optional keys:

            * ``dt`` (float) — time step in seconds. Default: ``0.1``.
            * ``horizon`` (int) — prediction horizon. Default: ``10``.
            * ``max_iters`` (int) — SQP iterations per solve call. Default: ``5``.
            * ``wheelbase`` (float) — vehicle wheelbase in metres. Default: ``0.16``.
            * ``obstacle_margin`` (float) — safety margin beyond obstacle radius. Default: ``0.25``.
            * ``obstacle_slack`` (bool) — use slack variables for obstacle constraints. Default: ``True``.
            * ``obstacle_slack_weight`` (float) — penalty weight for slack. Default: ``100.0``.
            * ``solver_options`` (dict) — keyword arguments forwarded to OSQP.
            * ``weights`` (dict) — state/control cost weights (see below).
            * ``constraints`` (dict) — physical limits (see below).

    Weight keys (all optional, with defaults):
        ``q_x=5``, ``q_y=5``, ``q_yaw=1``, ``q_v=0.1``,
        ``r_a=1``, ``r_delta=5``, ``r_da=15``, ``r_ddelta=10``.

    Constraint keys (all optional, with defaults):
        ``v_min=0``, ``v_max=3``, ``a_min=-3``, ``a_max=2``,
        ``delta_min=-0.5``, ``delta_max=0.5``, ``ddelta_max=0.3``.
    """

    def __init__(self, config: dict) -> None:
        self.dt: float = float(config["dt"])
        self.horizon: int = int(config["horizon"])
        self.max_iters: int = int(config.get("max_iters", 5))
        self.wheelbase: float = float(config.get("wheelbase", 0.16))
        self.obstacle_margin: float = float(config.get("obstacle_margin", 0.25))
        self.obstacle_slack: bool = bool(config.get("obstacle_slack", True))
        self.obstacle_slack_weight: float = float(config.get("obstacle_slack_weight", 100.0))
        self.solver_options: dict = config.get("solver_options", {})

        # Validate key parameters
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}.")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {self.max_iters}.")
        if self.wheelbase <= 0.0:
            raise ValueError(f"wheelbase must be positive, got {self.wheelbase}.")

        weights = config.get("weights", {})
        self.Q: np.ndarray = np.diag([
            float(weights.get("q_x", 5.0)),
            float(weights.get("q_y", 5.0)),
            float(weights.get("q_yaw", 1.0)),
            float(weights.get("q_v", 0.1)),
        ])
        self.Qf: np.ndarray = self.Q
        self.R: np.ndarray = np.diag([
            float(weights.get("r_a", 1.0)),
            float(weights.get("r_delta", 5.0)),
        ])
        self.Rd: np.ndarray = np.diag([
            float(weights.get("r_da", 15.0)),
            float(weights.get("r_ddelta", 10.0)),
        ])

        constraints = config.get("constraints", {})
        self.v_min: float = float(constraints.get("v_min", 0.0))
        self.v_max: float = float(constraints.get("v_max", 3.0))
        self.a_min: float = float(constraints.get("a_min", -3.0))
        self.a_max: float = float(constraints.get("a_max", 2.0))
        self.delta_min: float = float(constraints.get("delta_min", -0.5))
        self.delta_max: float = float(constraints.get("delta_max", 0.5))
        self.ddelta_max: float = float(constraints.get("ddelta_max", 0.3))

        logger.info(
            "IterativeMPC initialised: horizon=%d, dt=%.3f, max_iters=%d, wheelbase=%.3f",
            self.horizon, self.dt, self.max_iters, self.wheelbase,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_problem(
        self,
        x0: np.ndarray,
        ref_traj: np.ndarray,
        x_prev: np.ndarray,
        u_prev: np.ndarray,
        obstacles: Optional[Union[List[ObstacleDict], List[List[ObstacleDict]]]] = None,
    ) -> Tuple[cp.Problem, cp.Variable, cp.Variable]:
        """Construct and return the CVXPY optimisation problem for one SQP iteration.

        Args:
            x0: Initial state ``[x, y, yaw, v]`` of shape ``(4,)``.
            ref_traj: Reference trajectory of shape ``(N+1, 4)``.
            x_prev: Previous state trajectory used for linearisation, shape ``(N+1, 4)``.
            u_prev: Previous control sequence used for linearisation, shape ``(N, 2)``.
            obstacles: Optional obstacle specification.  Accepted formats:

                * ``None`` — no obstacles.
                * A list of ``N+1`` lists of obstacle dicts (per-step).
                * A single list of obstacle dicts (replicated for all steps).

        Returns:
            Tuple ``(problem, x_var, u_var)`` ready to be solved.
        """
        N = self.horizon
        x = cp.Variable((N + 1, 4))
        u = cp.Variable((N, 2))

        cons: list = [x[0, :] == x0]
        cost = 0.0
        slack_vars: list = []

        # Normalise obstacle input into a per-step sequence
        if obstacles is None:
            obstacle_sequence: list = [None] * (N + 1)
        elif isinstance(obstacles, list) and len(obstacles) == N + 1:
            obstacle_sequence = obstacles
        else:
            obstacle_sequence = [obstacles] * (N + 1)

        for k in range(N):
            A, B = linearize_dynamics(x_prev[k], u_prev[k], self.dt, self.wheelbase)
            c = x_prev[k + 1] - A.dot(x_prev[k]) - B.dot(u_prev[k])
            cons += [x[k + 1, :] == A @ x[k, :] + B @ u[k, :] + c]
            cons += [self.v_min <= x[k, 3], x[k, 3] <= self.v_max]
            cons += [self.a_min <= u[k, 0], u[k, 0] <= self.a_max]
            cons += [self.delta_min <= u[k, 1], u[k, 1] <= self.delta_max]

            # Steering-rate limit
            if k == 0:
                cons += [cp.abs(u[0, 1] - u_prev[0, 1]) <= self.ddelta_max]
            else:
                cons += [cp.abs(u[k, 1] - u[k - 1, 1]) <= self.ddelta_max]

            # Obstacle constraints at step k
            if obstacle_sequence[k] is not None:
                for obstacle in obstacle_sequence[k]:
                    direction, threshold = linearize_obstacle_constraint(
                        x_prev[k],
                        obstacle["center"],
                        float(obstacle["radius"]),
                        margin=self.obstacle_margin,
                    )
                    if self.obstacle_slack:
                        epsilon = cp.Variable(nonneg=True)
                        slack_vars.append(epsilon)
                        cons += [direction[0] * x[k, 0] + direction[1] * x[k, 1] + epsilon >= threshold]
                    else:
                        cons += [direction[0] * x[k, 0] + direction[1] * x[k, 1] >= threshold]

            # Stage cost
            dx = x[k, :] - ref_traj[k, :]
            cost += cp.quad_form(dx, self.Q)
            cost += cp.quad_form(u[k, :], self.R)
            if k > 0:
                du = u[k, :] - u[k - 1, :]
                cost += cp.quad_form(du, self.Rd)

        # Obstacle constraint at terminal step N
        if obstacle_sequence[N] is not None:
            for obstacle in obstacle_sequence[N]:
                direction, threshold = linearize_obstacle_constraint(
                    x_prev[N],
                    obstacle["center"],
                    float(obstacle["radius"]),
                    margin=self.obstacle_margin,
                )
                if self.obstacle_slack:
                    epsilon = cp.Variable(nonneg=True)
                    slack_vars.append(epsilon)
                    cons += [direction[0] * x[N, 0] + direction[1] * x[N, 1] + epsilon >= threshold]
                else:
                    cons += [direction[0] * x[N, 0] + direction[1] * x[N, 1] >= threshold]

        if self.obstacle_slack and slack_vars:
            cost += self.obstacle_slack_weight * cp.sum(slack_vars)

        # Terminal cost
        dx_terminal = x[N, :] - ref_traj[N, :]
        cost += cp.quad_form(dx_terminal, self.Qf)

        return cp.Problem(cp.Minimize(cost), cons), x, u

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: Union[Sequence, np.ndarray],
        ref_traj: np.ndarray,
        u_init: Optional[np.ndarray] = None,
        obstacles: Optional[Union[List[ObstacleDict], List[List[ObstacleDict]]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve the MPC problem and return the optimal state and control trajectories.

        The method performs up to :attr:`max_iters` SQP iterations, each time
        linearising the dynamics around the current warm-start trajectory.

        Args:
            x0: Initial vehicle state ``[x, y, yaw, v]``.
            ref_traj: Reference state trajectory of shape ``(N+1, 4)`` where
                ``N = self.horizon``.
            u_init: Optional warm-start control sequence of shape ``(N, 2)``.
                If ``None`` or wrong shape, zeros are used.
            obstacles: Optional obstacle specification (same format as
                :meth:`_build_problem`).

        Returns:
            Tuple ``(x_traj, u_traj)`` where:

            * ``x_traj`` — optimal state trajectory of shape ``(N+1, 4)``.
            * ``u_traj`` — optimal control sequence of shape ``(N, 2)``.

            Both are ``None`` if the solver fails or returns an infeasible /
            unbounded status.
        """
        N = self.horizon
        x0 = np.asarray(x0, dtype=float)

        if ref_traj.shape != (N + 1, 4):
            logger.error(
                "ref_traj shape mismatch: expected (%d, 4), got %s.", N + 1, ref_traj.shape
            )
            return None, None

        # Initialise warm-start control sequence
        if u_init is None or np.shape(u_init) != (N, 2):
            u_prev = np.zeros((N, 2), dtype=float)
            logger.debug("No valid u_init provided; using zero warm-start.")
        else:
            u_prev = np.array(u_init, dtype=float)

        # Rollout initial state trajectory from warm-start controls
        x_prev = np.zeros((N + 1, 4), dtype=float)
        x_prev[0, :] = x0
        for k in range(N):
            x_prev[k + 1, :] = bicycle_model(x_prev[k, :], u_prev[k, :], self.dt, self.wheelbase)

        for iteration in range(self.max_iters):
            logger.debug("SQP iteration %d / %d", iteration + 1, self.max_iters)
            problem, x_var, u_var = self._build_problem(x0, ref_traj, x_prev, u_prev, obstacles=obstacles)

            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **self.solver_options)
            except cp.error.SolverError as exc:
                logger.warning("CVXPY SolverError on iteration %d: %s", iteration + 1, exc)
                return None, None

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                logger.warning(
                    "Solver returned non-optimal status '%s' on iteration %d.",
                    problem.status, iteration + 1,
                )
                return None, None

            if u_var.value is None:
                logger.warning("u_var.value is None after solve on iteration %d.", iteration + 1)
                return None, None

            u_solution = np.array(u_var.value, dtype=float)
            if np.any(np.isnan(u_solution)):
                logger.warning("NaN values detected in solution on iteration %d.", iteration + 1)
                return None, None

            # Update warm-start for next SQP iteration
            u_prev = u_solution
            x_prev[0, :] = x0
            for k in range(N):
                x_prev[k + 1, :] = bicycle_model(x_prev[k, :], u_prev[k, :], self.dt, self.wheelbase)

        logger.debug("MPC solve completed successfully.")
        return x_prev, u_prev
