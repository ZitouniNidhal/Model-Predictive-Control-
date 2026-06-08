# mpc_python

An iterative Model Predictive Control demo built with CVXPY and an optional MuJoCo visualization path.

This repository is designed to help control practitioners move from basic kinematic tracking to a real-time convex optimization workflow.

## Structure

- `config/`
  - `mpc.yaml` — MPC weights, horizon, and constraints
  - `simulation.yaml` — demo configuration for the reference path and obstacles
- `mpc_python/cvxpy_mpc/`
  - `cvxpy_mpc.py` — iterative convex MPC implementation
  - `utils.py` — bicycle model, linearization, and reference generation helpers
- `mpc_python/models/mushr/`
  - `mushr.xml` — lightweight MuJoCo placeholder model for the demo
- `mpc_python/mpc_demo_mujoco.py` — MuJoCo demo entry point
- `mpc_python/mpc_demo_nosim.py` — headless no-physics demo entry point
# mpc_python — Iterative Model Predictive Control (MPC) Demo

This repository demonstrates an iterative Linear-Quadratic Model Predictive Control workflow for a simple bicycle-like vehicle. The controller is implemented with CVXPY (problem formulation) and OSQP (QP solver). A headless demo produces plotting outputs and an optional MuJoCo demo is available for visualization when MuJoCo is installed.

**Goals**
- Provide a clear, educational iterative-MPC implementation
- Support obstacle avoidance via convexified linear constraints
- Offer an easy headless demo and optional MuJoCo visualization
- Keep the codebase lightweight so helper utilities can be imported without heavy dependencies

**Contents**
- `config/` — `mpc.yaml` and `simulation.yaml` (controller + simulation settings)
- `mpc_python/cvxpy_mpc/` — core MPC implementation and helpers
- `mpc_python/mpc_demo_nosim.py` — headless demo (plots, optional CSV log)
- `mpc_python/mpc_demo_mujoco.py` — MuJoCo visualization demo (requires MuJoCo)
- `tests/` — lightweight unit tests

## Quick Start

Prerequisites: Python 3.11+ recommended. Create an isolated environment first.

Install dependencies with pip (recommended):

```bash
python -m pip install -r requirements.txt
```

Or with Conda using an environment YAML (if provided):

```bash
conda env create -f env.yml
conda activate mpc_python
python -m pip install -r requirements.txt
```

Run the headless demo (plots appear):

```bash
python mpc_python/mpc_demo_nosim.py
```

Save simulation history to CSV:

```bash
python mpc_python/mpc_demo_nosim.py --save-log logs/mpc_history.csv
```

Enable linearized obstacle avoidance:

```bash
python mpc_python/mpc_demo_nosim.py --obstacle-avoidance
```

Run the MuJoCo demo (MuJoCo must be installed and configured):

```bash
python mpc_python/mpc_demo_mujoco.py
```

Install package locally for import-style usage:

```bash
python -m pip install -e .
```

## Configuration

- `config/mpc.yaml` — MPC hyperparameters: `horizon`, `dt`, weights (`q_x`, `q_y`, ...), constraints (velocity, steering limits), solver settings.
- `config/simulation.yaml` — reference type (`circle` or `waypoints`), track parameters, static and moving obstacles, `simulation.max_steps`, `log_interval`.

Adjust `config/simulation.yaml` to change reference paths and obstacle placements.

## Development & Testing

Run unit tests with `pytest`:

```bash
python -m pytest -q
```

Notes for running tests in lightweight environments:
- Tests that exercise the full CVXPY solver are skipped when `cvxpy` is not installed. This lets helpers (e.g., CSV export) be tested without heavy dependencies.

## Troubleshooting

- Missing `cvxpy` error when importing the MPC module: install dependencies via `pip install -r requirements.txt` or install `cvxpy` separately:

```bash
python -m pip install cvxpy osqp
```

- MuJoCo demo errors: ensure MuJoCo is installed, `MUJOCO_PY_MJPRO_PATH` / `MUJOCO_KEY_PATH` (or newer mujoco env vars) are set according to your MuJoCo installation.

## Examples

- Run a short experiment and save CSV:

```bash
python mpc_python/mpc_demo_nosim.py --simulation config/simulation.yaml --config config/mpc.yaml --save-log out/history.csv
```

- Run headless with obstacle avoidance and start offset:

```bash
python mpc_python/mpc_demo_nosim.py --obstacle-avoidance --start-offset -0.2 --start-speed 0.5
```

## Contributing

Contributions welcome — open issues or PRs. Recommended workflow:

1. Fork and create a feature branch
2. Add tests for new features
3. Open a PR with a clear description and test results

## License & Credits

This project is provided as-is for demonstration and education. Add an appropriate license file if you plan to publish (e.g., MIT).

---
If you'd like, I can add a GitHub Actions workflow to run tests on push, include a CI badge in this README, or further expand the Usage section with sample `config/` snippets.
 
![CI](https://github.com/ZitouniNidhal/Model-Predictive-Control-/actions/workflows/python-app.yml/badge.svg)
