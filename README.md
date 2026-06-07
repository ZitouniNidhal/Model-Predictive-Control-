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

3. If MuJoCo is installed, run the MuJoCo demo:

```bash
python mpc_python/mpc_demo_mujoco.py
```

## Notes

- `mpc_demo_nosim.py` provides an accessible way to evaluate the iterative MPC without requiring MuJoCo.
- `mpc_demo_mujoco.py` uses a lightweight placeholder model and will work when `mujoco` is installed.
- The project is intentionally structured so you can extend the controller to real MuJoCo vehicle models.
