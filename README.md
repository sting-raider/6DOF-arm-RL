# 6-DOF Arm Pick-and-Place via Reinforcement Learning

> An Isaac Lab + rsl_rl pipeline that trains a UR10e robot to reach, grasp, and place objects using curriculum-based PPO — from scratch to basket placements at GPU-accelerated speed.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B%20cu126-ee4c2c?logo=pytorch&logoColor=white)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-76B900?logo=nvidia&logoColor=white)
![rsl_rl](https://img.shields.io/badge/rsl_rl-2.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview

This project implements a full reinforcement learning pipeline for a **6-DOF robotic arm** performing pick-and-place manipulation in **Isaac Lab** (NVIDIA's GPU-accelerated physics simulator). The goal is to learn a policy that can reliably reach, grasp, and drop an object into a target basket — entirely from scratch using RL, with a clear path toward sim-to-real transfer.

**Key design choices:**

- **Isaac Lab physics** (GPU) for massively parallel simulation at 45,000+ steps/sec
- **UR10e** robot with 6 active joints + **Robotiq 2F-85** gripper
- **PPO** via `rsl_rl` for continuous-action policy optimization
- **Curriculum training**: three sequential phases (REACH → GRASP → PLACE), each building on the previous
- **Potential-based reward shaping** that provides dense, positive learning signals
- **4096 parallel GPU environments** driving high throughput on a single RTX 3060 Laptop GPU
- **Complete 29D state observability** with stable, running observation normalization

---

## 🏆 Results

All training on **NVIDIA RTX 3060 Laptop GPU** (6 GB) with **4,096 parallel environments** (the stable maximum for 6GB VRAM).

| Phase | Task | Reward (train) | Wall Time | Envs | Status |
|-------|------|:--------:|----------:|-----:|--------|
| **0 – REACH** | Move EE to object | 0.32+ | ~54 min | 4096 | 🔄 Retraining (v14) |
| **1 – GRASP** | Reach + close + lift | — | TBD | 4096 | ⏳ Queued |
| **2 – PLACE** | Pick → basket | — | TBD | 4096 | ⏳ Queued |

**Training config:**
- Algorithm: **PPO** (rsl_rl), `num_envs = 4096`, `num_steps_per_env = 24`
- `learning_rate = 1e-4`, `gamma = 0.98`, `lam = 0.95`
- Actor/Critic: `[256, 128, 64]` MLP with ELU activations
- **Running Observation Normalization ENABLED** — Enabled on both actor and critic networks. Robotiq mimic joints are filtered out of the state space, ensuring stable variance calculations and smooth gradients.

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
│  GPU-parallel physics → ManagerBasedRLEnv × 4096            │
└────────────────────────┬─────────────────────────────────────┘
                         │ actions (7D)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PPO Agent (rsl_rl OnPolicyRunner)                │
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

### Observation Space (29D Complete State Observability)

| Component | Dim | Description |
|-----------|----:|-------------|
| `joint_pos` | 6 | Relative joint positions (radians) for the 6 active arm joints |
| `joint_vel` | 6 | Relative joint velocities (rad/s) for the 6 active arm joints |
| `ee_pos` | 3 | Scaled local end-effector position (x, y, z) |
| `gripper_state` | 1 | Scaled gripper position (0=open, 1.0=closed) |
| `object_pos` | 3 | Local object centroid position (x, y, z) |
| `relative_pos` | 3 | Local vector from end-effector to object (x, y, z) |
| `actions` | 7 | Previous action commands |
| **Total** | **29** | |

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

---

## ⚙️ Installation

**Prerequisites:**
- Python 3.11+
- CUDA 12.6+ (required for GPU-accelerated Isaac Lab)
- NVIDIA GPU with at least 6 GB VRAM
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

---

## 🚀 Training

Train the three curriculum phases sequentially. Each phase automatically warm-starts from the previous phase's best model.

```bash
# Phase 0: REACH — learn to move end-effector to the target position
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_isaac.py --phase 0 --num_envs 4096 --headless

# Phase 1: GRASP — learn to reach, grasp, and lift
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_isaac.py --phase 1 --num_envs 4096 --headless --checkpoint models/isaac/phase_0/model.pt

# Phase 2: PLACE — learn full pick-and-place into basket
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_isaac.py --phase 2 --num_envs 4096 --headless --checkpoint models/isaac/phase_1/model.pt
```

---

## 🎯 Evaluation

Evaluate the trained curriculum phases using the coordinate-aligned script:

```bash
python scripts/evaluate_isaac.py --phase 0 --model models/isaac/phase_0/model.pt --episodes 20 --num_envs 16
```

---

## ⚠️ Resolutions & Known Issues

### The Observation Normalizer Saga (RESOLVED)

**Problem:** In previous training sessions, enabling running observation normalizers caused policy collapse after 50-80 iterations. Attempting to run with normalizers disabled led directly to $4.5$ billion value function divergence due to GAE bootstrapping on unnormalized inputs with vastly different scales.

**Diagnostic & Solution:** We identified that the unactuated Robotiq mimic joints (5 out of 6 gripper joints) experienced massive instantaneous velocity spikes ($\ge 10,000$ rad/s) under contact due to PhysX constraint-solving forces. This blew up the normalizer's running variance, zeroing out all normal inputs. 
By **filtering out the unactuated gripper mimic joints** and observing strictly the **6 active arm joints**, the velocity spikes are completely eliminated. Running observation normalization is now **fully re-enabled and completely stable**, ensuring smooth gradient updates and preventing critic divergence.

### CUDA OOM at >6144 envs

On the 6GB RTX 3060 Mobile GPU, 6144 is the absolute maximum number of environments. **4096 environments** is the recommended VRAM sweet spot for training all phases.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<sub>Built with Isaac Lab · rsl_rl · PyTorch · Gymnasium · NVIDIA Isaac Sim</sub>
