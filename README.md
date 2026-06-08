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

## Quick start

1. Create the Python environment:

```bash
conda env create -f env.yml
conda activate mpc_python
```

2. Run the headless demo:

```bash
python mpc_python/mpc_demo_nosim.py
```

3. Enable obstacle avoidance in the headless demo:

```bash
python mpc_python/mpc_demo_nosim.py --obstacle-avoidance
```

4. Save the simulation history to CSV:

```bash
python mpc_python/mpc_demo_nosim.py --save-log logs/mpc_history.csv
```

5. Run the MuJoCo demo if MuJoCo is installed:

```bash
python mpc_python/mpc_demo_mujoco.py
```

5. Install locally for package-style imports:

```bash
python -m pip install -e .
```

## Features

- Iterative linearization around the bicycle model using CVXPY and OSQP
- Support for quadratic state and input penalties
- Steering rate constraints and obstacle avoidance through convexified halfspace constraints
- Reference generation from circular or waypoint-based paths
- A headless demo with trajectory plots, control inputs, and tracking error visualization

## Configuration

- `config/mpc.yaml` contains MPC weights, solver settings, and constraint limits
- `config/simulation.yaml` defines the reference path, static/moving obstacles, and run settings

## Notes

- `mpc_demo_nosim.py` is the easiest way to evaluate the controller without MuJoCo.
- `mpc_demo_mujoco.py` is a demo wrapper for a placeholder MuJoCo vehicle.
- `config/simulation.yaml` now supports circle and waypoint reference paths.

## Testing

Run the basic code-level test with:

```bash
pytest tests/test_mpc_basic.py
```
