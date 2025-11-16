# CarlaRL

CarlaRL is a small research codebase that trains and evaluates reinforcement learning agents (PPO and SAC variants) on visual steering control from the CARLA driving simulator.

This repository contains two main parts:
- a PPO implementation at the project root (training loop: `train.py`, resume: `trainExisting.py`, evaluation: `test.py`)
- a SAC implementation under the `SAC/` subfolder with a separate agent, replay buffer and training script (`SAC/train.py`).

## Repository layout

- `agent.py` - PPO agent and actor/critic network definitions used by root training scripts.
- `environment.py` - CARLA environment wrapper that provides image observations and step/reset logic.
- `model_config.py` - network architecture specifications used by the actor/critic models.
- `replay_buffer.py` - small replay/batch buffer used by the PPO implementation.
- `train.py` - main training loop for PPO (creates Agent, runs episodes, saves model & figures to `tmp/`).
- `trainExisting.py` - resume training or evaluate a saved PPO model.
- `test.py` - simple evaluation/test loop for PPO models.
- `SAC/` - an alternate SAC implementation with its own `agent.py`, `environment.py`, `model_config.py`, `replay_buffer.py` and `train.py`.
- `tmp/` - runtime artifacts (plots, trackers, saved checkpoints). This folder is ignored by git via `.gitignore`.

## Prerequisites

- CARLA simulator (matching version used during development). Start the CARLA server before running training. Typical steps:
	1. Download CARLA (https://carla.org) and unpack.
 2. Start the CARLA server (example):
```
./CarlaUE4.sh -opengl
```
or run the packaged server binary for your platform. Make sure server listens on `localhost:4000` (the default).

- Python 3.8+ (virtualenv recommended)
- PyTorch (a compatible CUDA build if you want GPU training)
- Common packages: numpy, matplotlib, opencv-python

Suggested quick-install (adjust to your Python environment and desired torch build):

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib opencv-python
```

Note: this repo does not include a pinned `requirements.txt`. If you want, create one after selecting your versions.

## Quick start - PPO (root scripts)

1. Start CARLA server and ensure it is reachable on localhost:4000.
2. Run training:

```
python train.py
```

Training saves intermediate artifacts to `tmp/` (checkpoint files like `actor_ppo` / `critic_ppo`, plots `figure.png`, and `tracker.txt`). Use `trainExisting.py` to load saved models and continue training, and `test.py` to run short evaluation runs.

Tips:
- The code expects 224x224 image observations. If you change camera sizes, update `environment.py` and adjust model input shapes accordingly.
- To use a GPU, ensure PyTorch detects CUDA (`torch.cuda.is_available()`) and that CARLA + data handling fit in GPU memory.

## Quick start - SAC

The `SAC/` folder contains a separate implementation. To run SAC training, `cd SAC` and run that folder's `train.py`:

```
python SAC/train.py
```

SAC training writes outputs into `tmp/` as well (in the repository root). Behavior and hyperparameters may differ from the PPO scripts.

## Code notes and caveats

- The codebase is a research / prototype code. Expect some rough edges: minimal argument parsing, limited error handling, and scripts that write temporary outputs to `tmp/`.
- Before training, ensure the CARLA server and the network/hyperparameter choices are appropriate for your machine.
- There are duplicated files between the root and `SAC/` — they are intentionally separate implementations.

## Suggested next improvements

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add CLI flags to training scripts (e.g., seed, device selection, number of episodes, checkpoint paths).
- Add unit tests and a small CI job to run basic syntax checks.

## Troubleshooting

- If the scripts hang at environment startup, check that CARLA is running and listening on the expected port.
- If CUDA is not used even when available, ensure you installed a CUDA-capable PyTorch wheel and that GPUs are available.

