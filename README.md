# mpc_python

Iterative Model Predictive Control (MPC) for a bicycle-like vehicle, built to support research, learning, and rapid prototyping.

This repository demonstrates an iterative convex MPC workflow using CVXPY and OSQP. It includes a headless simulation mode for analysis and logging, plus an optional MuJoCo visualization path for interactive control experiments.

## Why this repository exists

The goal is to provide a compact, research-friendly implementation of:

- a simple bicycle kinematic model,
- iterative linearization-based MPC,
- quadratic cost optimization over a finite horizon,
- optional obstacle avoidance via linearized constraints,
- interpretable reference generation for circular and waypoint trajectories.

This repository is useful for AI and control research because it keeps the core algorithm readable while supporting reproducibility and experiment variation.

## Key features

- Iterative model predictive control with warm-started optimization
- Headless no-simulation demo with plot output and CSV logging
- Obstacle avoidance through convexified linear constraints
- Optional MuJoCo demo for visualization and interactive playback
- Lightweight utility functions for dynamics, linearization, and reference generation
- Simple test coverage for solver integration and logging utilities

## Repository structure

- `config/`
  - `mpc.yaml` — MPC hyperparameters, weights, constraints, solver options
  - `simulation.yaml` — reference path, track, obstacle, and simulation settings
- `mpc_python/cvxpy_mpc/`
  - `cvxpy_mpc.py` — iterative convex MPC implementation
  - `utils.py` — bicycle model, linearization, obstacle handling, and reference builders
- `mpc_python/models/mushr/mushr.xml` — lightweight MuJoCo model used by the visual demo
- `mpc_python/mpc_demo_nosim.py` — headless simulation entry point
- `mpc_python/mpc_demo_mujoco.py` — MuJoCo visualization entry point
- `tests/` — small unit tests for MPC integration and CSV export

## Research context

This demo is designed for researchers and students exploring how iterative MPC behaves under:

- finite-horizon trajectory tracking,
- reference path following,
- actuator and state constraints,
- convexified obstacle avoidance,
- solver performance and feasibility.

It is especially well suited for experiments in:

- control theory education,
- robotics and autonomous driving prototyping,
- algorithm comparison against other MPC formulations,
- sensitivity analysis of weights, horizons, and constraints.

## Installation

Recommended Python version: `3.11+`.

Install dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

Or, create a Conda environment from the provided YAML:

```bash
conda env create -f env.yml
conda activate mpc_python
python -m pip install -r requirements.txt
```

Install the package locally for importable usage:

```bash
python -m pip install -e .
```

## Running demos

### Headless demo

```bash
python mpc_python/mpc_demo_nosim.py
```

This mode runs the MPC controller in simulation, plots the executed trajectory, control signals, and tracking error, and can optionally save history to CSV.

### Headless demo with CSV output

```bash
python mpc_python/mpc_demo_nosim.py --save-log logs/mpc_history.csv
```

### Obstacle avoidance

```bash
python mpc_python/mpc_demo_nosim.py --obstacle-avoidance
```

### MuJoCo visualization demo

```bash
python mpc_python/mpc_demo_mujoco.py
```

This requires a working MuJoCo installation and proper MuJoCo environment variables.

## Command-line options

`mpc_demo_nosim.py` supports:

- `--config` — path to `config/mpc.yaml`
- `--simulation` — path to `config/simulation.yaml`
- `--reference-mode` — override reference mode: `circle` or `waypoints`
- `--obstacle-avoidance` — enable linearized obstacle constraints
- `--start-offset` — initial lateral offset from the reference path
- `--start-speed` — initial vehicle speed
- `--save-log` — save state, control, and error history as CSV

`mpc_demo_mujoco.py` supports:

- `--config` — path to `config/mpc.yaml`
- `--simulation` — path to `config/simulation.yaml`
- `--reference-mode` — override reference mode for MuJoCo demo

## Configuration details

### `config/mpc.yaml`

Key sections:

- `dt` — control time step
- `horizon` — prediction horizon length
- `weights` — tracking and control penalties (`q_x`, `q_y`, `q_yaw`, `q_v`, `r_a`, `r_delta`, `r_da`, `r_ddelta`)
- `constraints` — velocity, acceleration, steering, and steering rate limits
- `solver_options` — optional CVXPY solver settings
- `obstacle_margin` — safe distance for obstacle avoidance
- `obstacle_slack` — allow soft obstacle constraints with slack variables
- `obstacle_slack_weight` — penalty weight for slack violations

### `config/simulation.yaml`

Key sections:

- `reference` — type and speed definitions for `circle` or `waypoints`
- `track` — circular track parameters: center, radius, start angle
- `obstacles` — static and moving obstacle definitions
- `simulation` — runtime options such as `max_steps` and `log_interval`

## Core algorithm overview

1. Generate a reference trajectory from either a circular path or a waypoint sequence.
2. Initialize the vehicle state and a warm-start control sequence.
3. Linearize the bicycle dynamics around the current predicted trajectory.
4. Formulate a convex quadratic program with state and input constraints.
5. Solve the QP with OSQP and update the control sequence.
6. Apply the first control command, simulate forward, and repeat.

This iterative linearization scheme makes the controller easier to inspect while still supporting nonlinear vehicle kinematics.

## Package API

The package exports:

- `mpc_python.cvxpy_mpc.IterativeMPC`
- `mpc_python.cvxpy_mpc.utils.load_yaml`
- `mpc_python.cvxpy_mpc.utils.bicycle_model`
- `mpc_python.cvxpy_mpc.utils.linearize_dynamics`
- `mpc_python.cvxpy_mpc.utils.build_circular_reference`
- `mpc_python.cvxpy_mpc.utils.build_waypoint_reference`

This allows the core solver and utilities to be reused in research scripts and notebooks.

## Testing

Run tests with:

```bash
python -m pytest -q
```

The repository includes lightweight tests for the MPC interface and the CSV export helper.

## Troubleshooting

### CVXPY missing

If the MPC import fails because `cvxpy` is not installed:

```bash
python -m pip install cvxpy osqp
```

### MuJoCo issues

If the MuJoCo demo fails to launch, verify that:

- the `mujoco` Python package is installed,
- MuJoCo binaries are installed on your machine,
- required environment variables such as `MUJOCO_PY_MJPRO_PATH` or newer MuJoCo paths are configured.

## Research guidance

To use this repository for AI/control research:

- vary `config/mpc.yaml` weights and examine the resulting trajectory quality,
- change `config/simulation.yaml` obstacle layouts to test robustness,
- compare this iterative MPC approach against a nonlinear MPC baseline,
- log results to CSV and analyze tracking error, control effort, and constraint satisfaction,
- add new reference generators or dynamics models for system identification studies.

## Contribution

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch
3. Add tests for new behavior
4. Open a pull request with a clear description and evaluation results

## License

This project is provided as-is for demonstration and educational use. Add a license such as MIT if you publish it publicly.
