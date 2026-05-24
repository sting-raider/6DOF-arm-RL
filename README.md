# 6-DOF Arm Pick-and-Place via Reinforcement Learning

> An Isaac Lab + rsl_rl pipeline that trains a UR10e robot to reach, grasp, and place objects using curriculum-based SAC — from scratch to basket placements at GPU-accelerated speed.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B%20cu126-ee4c2c?logo=pytorch&logoColor=white)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-76B900?logo=nvidia&logoColor=white)
![rsl_rl](https://img.shields.io/badge/rsl_rl-2.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview

This project implements a full reinforcement learning pipeline for a **6-DOF robotic arm** performing pick-and-place manipulation in **Isaac Lab** (NVIDIA's GPU-accelerated physics simulator). The goal is to learn a policy that can reliably reach, grasp, and drop an object into a target basket — entirely from scratch using RL, with a clear path toward sim-to-real transfer.

**Key design choices:**

- **Isaac Lab physics** (GPU) for massively parallel simulation at 64,000+ FPS
- **UR10e** robot with 6 controlled joints + **Robotiq 2F-85** gripper
- **SAC (Soft Actor-Critic)** via `rsl_rl` for continuous-action off-policy learning
- **Curriculum training**: three sequential phases (REACH → GRASP → PLACE), each building on the previous
- **Potential-based reward shaping** that provides dense, positive learning signal at every step
- **32 parallel GPU environments** driving ~64,000 FPS on a single RTX 3060

---

## 🏆 Results

All training on **NVIDIA RTX 3060 Laptop GPU** (6 GB) with **6,144 parallel environments** (stable maximum for 6GB VRAM).

| Phase | Task | Reward (train) | Wall Time | Envs | Status |
|-------|------|:--------:|----------:|-----:|--------|
| **0 – REACH** | Move EE to object | 0.64-0.69 | ~25 min | 6144 | 🔄 Retraining |
| **1 – GRASP** | Reach + close + lift | — | ~22 min | 6144 | ⏳ Queued |
| **2 – PLACE** | Pick → basket | — | ~16 min | 4096 | ⏳ Queued |

**Training config:**
- Algorithm: **PPO** (rsl_rl), `num_envs = 6144` (Phase 0-1) / `4096` (Phase 2), `num_steps_per_env = 24`
- `learning_rate = 3e-4`, `gamma = 0.99`, `lam = 0.95`, `entropy_coef = 0.01`
- Actor/Critic: `[256, 128, 64]` MLP with ELU activations
- **Observation normalization DISABLED** — `EmpiricalNormalization` co-adapts with the policy during training, producing corrupted statistics that break inference. See [Known Issues](#-known-issues).

> **Warm-start chain:** Phase N loads Phase N−1 checkpoint → policy transfers knowledge between phases.

**Evaluation:** Uses raw Isaac Lab scene positions (not normalized observations) for success detection:
- **Reach:** EE within 5cm of object
- **Grasp:** Gripper closed (>0.02) AND object lifted above table (>0.88m)
- **Place:** Object within 10cm of basket center

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Isaac Lab Simulation                       │
│  GPU-parallel physics → ManagerBasedRLEnv × 32              │
└────────────────────────┬─────────────────────────────────────┘
                         │ actions (7D)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              SAC Agent (rsl_rl OnPolicyRunner)                │
│  Actor [256,128,64] → Gaussian distribution → action         │
│  Critic [256,128,64] → state-value estimation                │
│  curriculum_phase ∈ {0=REACH, 1=GRASP, 2=PLACE}             │
└────────────────────────┬─────────────────────────────────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
┌──────────────────────┐  ┌───────────────────────┐
│  UR10e + Robotiq     │  │  Isaac Lab Physics     │
│  (PD joint control,  │  │  (GPU-parallel Warp)   │
│   weld grasp, FK)    │  │  ground, table, basket,│
└──────────────────────┘  │  object, lights        │
                          └───────────────────────┘
```

### Observation Space (7D)

| Component | Dim | Description |
|-----------|----:|-------------|
| `ee_pos` | 3 | End-effector position (x, y, z) relative to env origin |
| `object_pos` | 3 | Object centroid position (x, y, z) relative to env origin |
| `gripper_state` | 1 | Gripper joint position (0=open, 0.04=closed) |
| **Total** | **7** | |

### Action Space (7D)

| Component | Range | Description |
|-----------|------:|-------------|
| `arm_action` | [-1, 1] | 6 joint position deltas (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3) |
| `gripper_action` | [-1, 1] | Binary gripper: positive→close, negative→open |

### Reward Function

Rewards use **positive potential-based shaping** to avoid the sparse-reward problem:

- **Phase 0 (REACH):** `exp(-||ee_xyz - target_xyz|| / 0.2)` — dense reward for proximity
- **Phase 1 (GRASP):** Reach reward + grasp bonus when gripper closes near the object
- **Phase 2 (PLACE):** `exp(-||object_xyz - basket_xyz|| / 0.2)` — reward for placing in basket
- **Action penalty:** `-0.001 × sum(action²)` — encourages smooth, efficient motion

### Curriculum Phases

| Phase | Name | Episode Length | Reward Focus |
|-------|------|---------------:|--------------|
| 0 | REACH | 10 s (600 steps) | Move EE within 5 cm of target position |
| 1 | GRASP | 10 s (600 steps) | Reach + engage gripper + lift above table |
| 2 | PLACE | 10 s (600 steps) | Full task: reach → grasp → lift → transport → basket |

---

## ⚙️ Installation

**Prerequisites:**
- Python 3.11+
- CUDA 12.6+ (required for GPU-accelerated Isaac Lab)
- NVIDIA GPU with at least 6 GB VRAM (tested on RTX 3060 Mobile)
- Isaac Sim / Isaac Lab (see [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab))

```bash
git clone https://github.com/sting-raider/ic-6dof-arm.git
cd ic-6dof-arm

# Create virtual environment for Isaac Lab
python -m venv isaacsim-venv-3.11
source isaacsim-venv-3.11/bin/activate

# Install Isaac Lab (pip-based)
pip install isaaclab

# Install project dependencies
pip install -r requirements.txt
```

> **Note:** The original MuJoCo-based pipeline has been archived in `archived_mujoco/`. All active development uses Isaac Lab.

---

## 🚀 Training

Train the three curriculum phases sequentially. Each phase automatically warm-starts from the previous phase's best model.

```bash
# Phase 0: REACH — learn to move end-effector to the target position
python scripts/train_isaac.py --phase 0 --num_envs 32 --headless

# Phase 1: GRASP — learn to reach, grasp, and lift
python scripts/train_isaac.py --phase 1 --num_envs 32 --headless

# Phase 2: PLACE — learn full pick-and-place into basket
python scripts/train_isaac.py --phase 2 --num_envs 32 --headless
```

**Training with live viewer (requires display):**
```bash
python scripts/train_isaac.py --phase 0 --num_envs 32 --enable_cameras
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir logs/isaac/
```

Key metrics to watch: `Episode_Reward/reach`, `Episode_Reward/action_penalty`, `Episode_Termination/time_out`, `Mean reward`, and the per-iteration reward curve.

**Saved artifacts per phase:**
- `models/isaac/phase_N/model.pt` — final trained checkpoint
- `logs/isaac/phase_N/` — TensorBoard logs and training config
- `logs/isaac/training_live.log` — live tail of training progress

---

## 🎯 Evaluation

Evaluate all three curriculum phases sequentially with a summary table:

```bash
python scripts/evaluate_all.py
```

The script automatically runs each phase, collects success metrics, and prints a comparison. Use `--dry-run` to preview commands without executing.

---

## 📁 Project Structure

```
ic-6dof-arm/
├── IsaacLab/                        # Isaac Lab framework (git submodule / local)
├── configs/
│   └── sac_config.yaml              # SAC hyperparameters and phase config
├── isaac_env/
│   ├── env_cfg.py                   # UR10e environment configuration (scene, actions, obs, rewards)
│   └── mdp.py                       # MDP functions (actions, observations, rewards, terminations, events)
├── scripts/
│   ├── train_isaac.py               # Main training script (SAC via rsl_rl)
│   └── evaluate_all.py              # Batch evaluation across all phases
├── tests/
│   └── smoke_test.py                # Smoke tests (env reset, step, obs shape, rewards)
├── archived_mujoco/                 # Archived MuJoCo-based pipeline (no longer active)
├── models/                          # Saved model checkpoints (gitignored)
│   └── isaac/
│       ├── phase_0/                 # REACH models
│       ├── phase_1/                 # GRASP models
│       └── phase_2/                 # PLACE models
├── logs/                            # Training logs (TensorBoard + live log)
│   └── isaac/
│       ├── phase_0/
│       ├── phase_1/
│       ├── phase_2/
│       └── training_live.log
├── isaacsim-venv-3.11/             # Python virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🧪 Tests

```bash
pytest tests/smoke_test.py -v
```

---

## ⚠️ Known Issues

### EmpiricalNormalization breaks inference

RLSL-RL's `EmpiricalNormalization` accumulates running statistics during training. For observations with near-constant values (e.g., gripper state = 0.0 in Phase 0 REACH), the running variance collapses to zero, producing extreme or NaN standard deviations. When the model is saved, these corrupted normalizer stats are persisted. At inference time, normalized observations become garbage, and the policy outputs nonsense actions.

**Fix applied:** `obs_normalization: False` in both training and eval scripts. The observation values (0-1.5m positions, 0-0.04 gripper) are already in reasonable ranges — normalization is unnecessary for this task.

### CUDA OOM at >6144 envs

On the 6GB RTX 3060, 8192 environments cause PhysX CUDA OOM during initialization. **6144 is the stable maximum** for Phases 0-1. Phase 2 (with basket) uses 4096 for headroom.

---

## 🗺️ Roadmap

- [ ] **Phase 0 — REACH:** Training with `obs_normalization=False`, 1000 iterations
- [ ] **Phase 1 — GRASP:** Warm-start from Phase 0
- [ ] **Phase 2 — PLACE:** Warm-start from Phase 1
- [ ] **Evaluation:** 20-episode eval per phase with real scene position metrics
- [ ] **Domain Randomization:** Object mass/size/friction variation
- [ ] **Vision Policy (Phase 3):** RGB camera input → CNN policy
- [ ] **Sim-to-Real Transfer:** ROS 2 bridge, real UR10e deployment

See [PLAN.md](PLAN.md) for the full detailed roadmap with status per sub-task.

---

## 🧪 Tests

```bash
pytest tests/smoke_test.py -v
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<sub>Built with Isaac Lab · rsl_rl · PyTorch · Gymnasium · NVIDIA Isaac Sim</sub>
