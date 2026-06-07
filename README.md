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

4. If MuJoCo is installed, run the MuJoCo demo:

```bash
python mpc_python/mpc_demo_mujoco.py
```

## Features

- Iterative linearization around the bicycle model using CVXPY and OSQP
- Support for quadratic state and input penalties
- Steering rate constraints and obstacle avoidance through convexified halfspace constraints
- A headless demo with trajectory plots, control input graphs, and tracking error visualization

## Configuration

- `config/mpc.yaml` contains MPC weighting and constraint parameters
- `config/simulation.yaml` includes the reference path and obstacle definitions

## Notes

- `mpc_demo_nosim.py` is the easiest way to run the controller without MuJoCo.
- `mpc_demo_mujoco.py` is a demo wrapper for a placeholder MuJoCo vehicle.
- The obstacle avoidance demo uses moving obstacle predictions and sequential convexification.
