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

All training performed on **NVIDIA RTX 3060 Laptop GPU** (6 GB) with **32 parallel environments** in Isaac Lab.

| Phase | Task | Steps | Wall Time | Steps/sec | Reach Rate |
|-------|------|------:|----------:|----------:|----------:|
| **0 – REACH** | Move EE to target | 500K | ~8 min | 64,142 | **97%** |
| **1 – GRASP** | Reach + close gripper + lift | 1M | ~16 min | 63,800 | — |
| **2 – PLACE** | Full end-to-end pick-and-place | 2M | ~32 min | 62,500 | — |

> **Phase 0** achieves a **97% reach success rate** — the end-effector reaches within 5 cm of the target position reliably. Phases 1 and 2 are in active training with ongoing improvements.

**Training config:**
- `num_envs = 32`, `num_steps_per_env = 24`, `num_learning_epochs = 8`
- `learning_rate = 3e-4`, `gamma = 0.99`, `lam = 0.95`
- Actor network: `[256, 128, 64]` with ELU activations
- Observation normalization via `EmpiricalNormalization`

> **Warm-start chain:** Each phase loads the previous phase's best checkpoint so the agent never learns from scratch.

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
├── PLAN.md                          # Full project roadmap and phase status
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🗺️ Roadmap

The Isaac Lab curriculum pipeline is **active through Phase 2**. Current and planned work:

- [x] **Phase 0 — REACH:** 97% reach rate achieved, model trained and saved
- [ ] **Phase 1 — GRASP:** In training — reach + grasp + lift with domain randomization
- [ ] **Phase 2 — PLACE:** In training — full end-to-end pick-and-place into basket
- [ ] **Robustness:** Extend training to 5M+ steps; push place success rate >30%
- [ ] **Domain Randomization:** Object mass/size/friction variation, joint noise, camera noise
- [ ] **Vision Policy (Phase 3):** Switch from 7D state observations to camera-based RGB input with CNN policy
- [ ] **Advanced Algorithms:** HER for sparse rewards, TD3, Dreamer v3, hierarchical RL
- [ ] **Sim-to-Real Transfer:** ROS 2 bridge, system identification, real UR10e deployment

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
