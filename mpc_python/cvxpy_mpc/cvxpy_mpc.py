import numpy as np
import cvxpy as cp
from .utils import bicycle_model, linearize_dynamics, linearize_obstacle_constraint


class IterativeMPC:
    def __init__(self, config):
        self.dt = float(config["dt"])
        self.horizon = int(config["horizon"])
        self.max_iters = int(config.get("max_iters", 5))
        self.wheelbase = float(config.get("wheelbase", 0.16))
        self.obstacle_margin = float(config.get("obstacle_margin", 0.25))
        self.solver_options = config.get("solver_options", {})

        weights = config.get("weights", {})
        self.Q = np.diag([
            float(weights.get("q_x", 5.0)),
            float(weights.get("q_y", 5.0)),
            float(weights.get("q_yaw", 1.0)),
            float(weights.get("q_v", 0.1)),
        ])
        self.Qf = self.Q
        self.R = np.diag([
            float(weights.get("r_a", 1.0)),
            float(weights.get("r_delta", 5.0)),
        ])
        self.Rd = np.diag([
            float(weights.get("r_da", 15.0)),
            float(weights.get("r_ddelta", 10.0)),
        ])

        constraints = config.get("constraints", {})
        self.v_min = float(constraints.get("v_min", 0.0))
        self.v_max = float(constraints.get("v_max", 3.0))
        self.a_min = float(constraints.get("a_min", -3.0))
        self.a_max = float(constraints.get("a_max", 2.0))
        self.delta_min = float(constraints.get("delta_min", -0.5))
        self.delta_max = float(constraints.get("delta_max", 0.5))
        self.ddelta_max = float(constraints.get("ddelta_max", 0.3))

    def _build_problem(self, x0, ref_traj, x_prev, u_prev, obstacles=None):
        N = self.horizon
        x = cp.Variable((N + 1, 4))
        u = cp.Variable((N, 2))

        constraints = [x[0, :] == x0]
        cost = 0.0

        if obstacles is None:
            obstacle_sequence = [None] * (N + 1)
        elif isinstance(obstacles, list) and len(obstacles) == N + 1:
            obstacle_sequence = obstacles
        else:
            obstacle_sequence = [obstacles] * (N + 1)

        for k in range(N):
            A, B = linearize_dynamics(x_prev[k], u_prev[k], self.dt, self.wheelbase)
            c = x_prev[k + 1] - A.dot(x_prev[k]) - B.dot(u_prev[k])
            constraints += [x[k + 1, :] == A @ x[k, :] + B @ u[k, :] + c]
            constraints += [self.v_min <= x[k, 3], x[k, 3] <= self.v_max]
            constraints += [self.a_min <= u[k, 0], u[k, 0] <= self.a_max]
            constraints += [self.delta_min <= u[k, 1], u[k, 1] <= self.delta_max]

            if k > 0:
                constraints += [cp.abs(u[k, 1] - u[k - 1, 1]) <= self.ddelta_max]

            if obstacle_sequence[k] is not None:
                for obstacle in obstacle_sequence[k]:
                    direction, threshold = linearize_obstacle_constraint(
                        x_prev[k], obstacle["center"], float(obstacle["radius"]), margin=self.obstacle_margin
                    )
                    constraints += [direction[0] * x[k, 0] + direction[1] * x[k, 1] >= threshold]

            dx = x[k, :] - ref_traj[k, :]
            cost += cp.quad_form(dx, self.Q)
            cost += cp.quad_form(u[k, :], self.R)
            if k > 0:
                du = u[k, :] - u[k - 1, :]
                cost += cp.quad_form(du, self.Rd)

        if obstacle_sequence[N] is not None:
            for obstacle in obstacle_sequence[N]:
                direction, threshold = linearize_obstacle_constraint(
                    x_prev[N], obstacle["center"], float(obstacle["radius"]), margin=self.obstacle_margin
                )
                constraints += [direction[0] * x[N, 0] + direction[1] * x[N, 1] >= threshold]

        dx_terminal = x[N, :] - ref_traj[N, :]
        cost += cp.quad_form(dx_terminal, self.Qf)

        return cp.Problem(cp.Minimize(cost), constraints), x, u

    def solve(self, x0, ref_traj, u_init=None, obstacles=None):
        N = self.horizon
        if u_init is None or np.shape(u_init) != (N, 2):
            u_prev = np.zeros((N, 2), dtype=float)
        else:
            u_prev = np.array(u_init, dtype=float)

        x_prev = np.zeros((N + 1, 4), dtype=float)
        x_prev[0, :] = x0
        for k in range(N):
            x_prev[k + 1, :] = bicycle_model(x_prev[k, :], u_prev[k, :], self.dt, self.wheelbase)

        for _ in range(self.max_iters):
            problem, x_var, u_var = self._build_problem(x0, ref_traj, x_prev, u_prev, obstacles=obstacles)
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **self.solver_options)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                break

            if u_var.value is None:
                break

            u_solution = np.array(u_var.value, dtype=float)
            if np.any(np.isnan(u_solution)):
                break

            u_prev = u_solution
            x_prev[0, :] = x0
            for k in range(N):
                x_prev[k + 1, :] = bicycle_model(x_prev[k, :], u_prev[k, :], self.dt, self.wheelbase)

        return x_prev, u_prev
