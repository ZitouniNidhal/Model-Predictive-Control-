# Model Predictive Control (MPC) Demo

Iterative Model Predictive Control for a bicycle-like vehicle, built for research, learning, and rapid prototyping.

This repository demonstrates an iterative convex MPC workflow using CVXPY and OSQP. It includes a headless simulation mode for analysis and logging, plus an optional MuJoCo visualization demo.

## Why this repository exists

The goal is to provide a compact, research-friendly implementation of:

- a bicycle kinematic model,
- iterative linearization-based MPC,
- quadratic cost optimization over a prediction horizon,
- obstacle avoidance through convexified constraints,
- reference generation for circular and waypoint trajectories.

The implementation is designed for rapid experimentation, reproducibility, and easy extension.

## Key features

- Iterative MPC with warm-started optimization
- Headless simulation demo with plotting and CSV logging
- Optional obstacle avoidance via linearized constraints
- MuJoCo visualization demo for interactive playback
- Lightweight utility modules for dynamics, linearization, and reference generation
- Basic unit tests for solver and utility behavior

## Repository structure

- `config/`
  - `mpc.yaml` — MPC configuration and solver weights
  - `simulation.yaml` — reference path, track, and obstacle settings
- `mpc_python/cvxpy_mpc/`
  - `cvxpy_mpc.py` — iterative convex MPC implementation
  - `utils.py` — bicycle dynamics, linearization, obstacle handling, and references
- `mpc_python/models/mushr/mushr.xml` — MuJoCo model for the visualization demo
- `mpc_python/mpc_demo_nosim.py` — headless simulation entry point
- `mpc_python/mpc_demo_mujoco.py` — MuJoCo visualization entry point
- `tests/` — automated tests

## Installation

Recommended Python version: `3.11+`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Or create a Conda environment:

```bash
conda env create -f env.yml
conda activate mpc_python
python -m pip install -r requirements.txt
```

Install the package locally:

```bash
python -m pip install -e .
```

## Running demos

### Headless demo

```bash
python mpc_python/mpc_demo_nosim.py
```

This runs the MPC controller in simulation, plots trajectory and control signals, and optionally saves history to CSV.

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

MuJoCo requires a working installation and proper environment variables.

## Command-line options

`mpc_demo_nosim.py` supports:

- `--config` — path to `config/mpc.yaml`
- `--simulation` — path to `config/simulation.yaml`
- `--reference-mode` — `circle` or `waypoints`
- `--obstacle-avoidance` — enable obstacle avoidance
- `--start-offset` — initial lateral offset from the reference
- `--start-speed` — initial speed
- `--save-log` — CSV file path for history

`mpc_demo_mujoco.py` supports:

- `--config` — path to `config/mpc.yaml`
- `--simulation` — path to `config/simulation.yaml`
- `--reference-mode` — `circle` or `waypoints`

## Configuration details

### `config/mpc.yaml`

Key options:

- `dt` — control time step
- `horizon` — prediction horizon length
- `weights` — penalties on tracking and control effort
- `constraints` — velocity, acceleration, steering, and steering rate limits
- `solver_options` — optional CVXPY solver settings
- `obstacle_margin` — safety buffer for obstacle avoidance
- `obstacle_slack` — enable soft obstacle constraints
- `obstacle_slack_weight` — penalty for slack violation

### `config/simulation.yaml`

Key options:

- `reference` — circle or waypoints reference mode
- `track` — circular track parameters
- `obstacles` — static and moving obstacle definitions
- `simulation` — `max_steps` and `log_interval`

## Core algorithm overview

1. Create a reference trajectory from a circular path or waypoints.
2. Initialize vehicle state and warm-start inputs.
3. Linearize the bicycle dynamics around the predicted trajectory.
4. Build a convex QP with state, control, and obstacle constraints.
5. Solve the QP using OSQP and update the control sequence.
6. Apply the first command, simulate one step, and repeat.

## Package API

Main exports:

- `mpc_python.IterativeMPC`
- `mpc_python.load_yaml`
- `mpc_python.bicycle_model`
- `mpc_python.linearize_dynamics`
- `mpc_python.build_circular_reference`
- `mpc_python.build_waypoint_reference`

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

## Troubleshooting

### Missing CVXPY

Install CVXPY and OSQP if the solver import fails:

```bash
python -m pip install cvxpy osqp
```

### MuJoCo issues

If MuJoCo fails to launch, verify the installation and environment variables.


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
