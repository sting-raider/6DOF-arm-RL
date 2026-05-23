# 6-DOF Arm Pick-and-Place via Reinforcement Learning

> A MuJoCo + Stable-Baselines3 pipeline that trains a Kuka iiwa robot to reach, grasp, and place objects using curriculum-based SAC — from scratch to basket placements in under 2 hours of GPU time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B%20cu126-ee4c2c?logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-00bfa5?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PC9zdmc+)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview

This project implements a full reinforcement learning pipeline for a **6-DOF robotic arm** performing pick-and-place manipulation in simulation. The goal is to learn a policy that can reliably reach, grasp, and drop an object into a target basket — entirely from scratch using RL, with a clear path toward sim-to-real transfer.

**Key design choices:**

- **MuJoCo physics** (CPU) for fast, accurate contact simulation
- **Kuka iiwa 7** robot with 6 controlled joints + a magnetic weld-based gripper
- **SAC (Soft Actor-Critic)** via Stable-Baselines3 for continuous-action off-policy learning
- **Curriculum training**: three sequential phases (REACH → GRASP → PLACE), each warm-starting from the previous best checkpoint
- **Potential-based reward shaping** that provides dense, positive learning signal at every step
- **16 parallel vectorized environments** driving ~1,800 FPS on a single laptop GPU

---

## 🏆 Results

All training performed on **NVIDIA RTX 3060 Laptop GPU** with PyTorch 2.12.0+cu126 and MuJoCo 3.x.

| Phase | Task | Steps | Wall Time | Eval Reward | Reach | Grasp | Place |
|-------|------|------:|----------:|------------:|------:|------:|------:|
| **0 – REACH** | Move EE to object | 2M | 31 min | **-56 → +126** | 15% | 0% | 0% |
| **1 – GRASP** | Reach + close gripper + lift | 2M | 33 min | **→ +424** (peak) | 20% | 20% | 0% |
| **2 – PLACE** | Full end-to-end pick-and-place | 3M | 50 min | **→ +680** (max +2701) | 20% | 20% | **5%** |

**Hardware:** NVIDIA RTX 3060 Laptop GPU · **Simulation speed:** ~1,800 FPS (vs ~500 FPS on CPU — 3.4× speedup)

**Training config:**
- `N_ENVS = 16`, `batch_size = 1024`, `gradient_steps = 64` (`N_ENVS × 4`)
- `buffer_size = 1,000,000`, `learning_rate = 3e-4`, `γ = 0.99`

> **Warm-start chain:** Phase 0 → 1 → 2. Each phase loads the previous phase's best checkpoint so the agent never learns from scratch.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                         │
│  SubprocVecEnv (16 parallel) → VecNormalize → SAC (GPU) │
└───────────────────┬─────────────────────────────────────┘
                    │ actions (6D)
                    ▼
┌─────────────────────────────────────────────────────────┐
│              PickAndPlaceEnv (Gymnasium)                 │
│  curriculum_phase ∈ {0=REACH, 1=GRASP, 2=PLACE}        │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌──────────────────┐  ┌───────────────────┐
│  KukaRobot       │  │  MuJoCo Physics   │
│  (PD control,    │  │  pick_and_place_  │
│   weld grasp,    │  │  scene.xml        │
│   FK)            │  │  (table, basket,  │
└──────────────────┘  │   object)         │
                      └───────────────────┘
```

### Observation Space (20D)

| Component | Dim | Description |
|-----------|----:|-------------|
| `ee_pos` | 3 | End-effector position (x, y, z) |
| `obj_pos` | 3 | Object position (x, y, z) |
| `relative_pos` | 3 | Object position relative to EE |
| `joint_pos` | 5 | Joint angles for joints 0–4 |
| `joint_vel` | 5 | Joint velocities for joints 0–4 |
| `gripper_state` | 1 | Gripper open/closed state |
| **Total** | **20** | |

### Action Space (6D)

| Component | Range | Description |
|-----------|------:|-------------|
| `delta_joint_0..4` | [-1, 1] | Joint angle deltas (5 DOF) |
| `gripper_action` | [-1, 1] | Gripper open/close command |

### Reward Function

Rewards use **positive potential-based shaping** to avoid the sparse-reward problem:

- **Baseline:** `max(0, 1 - dist / max_workspace_dist)` — always positive, higher when closer
- **Temporal shaping:** `10 × (prev_dist - dist)` — rewards progress toward object
- **Phase bonuses:** reach (+5), grasp (+5), lift (up to +3), basket placement (+50)
- **Efficiency penalty:** `-0.005` per step to encourage shorter trajectories

### Curriculum Phases

| Phase | Name | Episode Length | Reward Focus |
|-------|------|---------------:|--------------|
| 0 | REACH | 200 steps | Move EE within 5 cm of object |
| 1 | GRASP | 300 steps | Reach + engage gripper + lift above table |
| 2 | PLACE | 400 steps | Full task: reach → grasp → lift → transport → basket |

---

## ⚙️ Installation

**Prerequisites:** Python 3.10+, CUDA 12.6+ (for GPU training)

```bash
git clone https://github.com/sting-raider/6DOF-arm-RL.git
cd 6DOF-arm-RL

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# GPU training (CUDA 12.6) — strongly recommended
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** CPU-only training works but is ~3.4× slower. Omit the last line to use the CPU-only PyTorch from `requirements.txt`.

---

## 🚀 Training

Train the three curriculum phases sequentially. Each phase automatically warm-starts from the previous phase's best model.

```bash
# Phase 0: REACH — learn to move end-effector to the object (31 min on RTX 3060)
python scripts/train.py --phase 0 --timesteps 2000000

# Phase 1: GRASP — learn to reach, grasp, and lift (33 min on RTX 3060)
python scripts/train.py --phase 1 --timesteps 2000000

# Phase 2: PLACE — learn full pick-and-place into basket (50 min on RTX 3060)
python scripts/train.py --phase 2 --timesteps 3000000
```

Saved artifacts per phase:
- `models/phase_N/best_model.zip` — best checkpoint by eval reward
- `models/phase_N/final_model.zip` — final checkpoint after all timesteps
- `models/phase_N/vec_normalize.pkl` — observation normalization statistics

**Monitor training with TensorBoard:**

```bash
tensorboard --logdir logs/
```

Key metrics to watch: `reward/r_reach_bonus`, `reward/r_grasp_bonus`, `reward/r_place_bonus`, `reward/reach_success`, `reward/grasp_success`, `reward/place_success`.

**Override config via CLI or `configs/sac_config.yaml`:**

```bash
python scripts/train.py --phase 2 --timesteps 5000000 --config configs/sac_config.yaml
```

---

## 🎯 Evaluation

```bash
python scripts/evaluate_comprehensive.py \
    --model models/phase_2/best_model \
    --phase 2 \
    --episodes 20
```

> **Known issue:** The evaluation script has a VecNormalize loading bug that can cause early episode termination (object falls at step 1–23). This is a measurement artifact; training-time success rates are accurate. Fix is tracked in PLAN.md Phase 2.0.

---

## 📁 Project Structure

```
6DOF-arm-RL/
├── configs/
│   └── sac_config.yaml          # SAC hyperparameters and phase config
├── envs/
│   └── pick_and_place_env.py    # Gymnasium environment (20D obs, curriculum)
├── robots/
│   └── kuka_iiwa.py             # KukaRobot wrapper (PD control, weld grasp, FK)
├── scenes/
│   └── pick_and_place_scene.xml # MuJoCo XML scene (arm, table, basket, object)
├── scripts/
│   ├── train.py                 # Main training script (SAC, warm-start chain)
│   ├── evaluate_comprehensive.py# Multi-episode evaluation with success metrics
│   └── web_demo.py              # Browser-based visualization (needs update)
├── sensors/
│   └── camera.py                # Overhead camera (HSV detection, pixel→world)
├── isaac_env/                   # Isaac Lab migration scaffolding (WIP)
│   ├── env_cfg.py
│   └── mdp.py
├── tests/
│   └── smoke_test.py            # 13 pytest tests (env reset, step, reward, obs shape)
├── utils/
│   └── constants.py             # Shared constants (workspace bounds, basket pos, etc.)
├── models/                      # Saved model checkpoints (gitignored)
│   ├── phase_0/                 # REACH models
│   ├── phase_1/                 # GRASP models
│   └── phase_2/                 # PLACE models
├── logs/                        # TensorBoard logs (gitignored)
├── archive/                     # Archived/obsolete scripts
├── PLAN.md                      # Full project roadmap and phase status
└── requirements.txt             # Python dependencies
```

---

## 🗺️ Roadmap

The MuJoCo curriculum pipeline is **complete through Phase 2**. Next steps:

- [ ] **Phase 2 — Robustness:** Retrain with 5M+ steps; push place success from 5% → 30%+
- [ ] **Domain Randomization:** Object mass/size/friction variation for sim-to-real readiness
- [ ] **Vision Policy (Phase 3):** Switch from state observations to 64×64 RGB camera inputs with a CNN policy
- [ ] **Isaac Lab Migration (Phase 4):** GPU-parallel physics at 10,000+ FPS using 128+ environments
- [ ] **Advanced Algorithms (Phase 5):** HER for sparse rewards, TD3, Dreamer v3, hierarchical RL
- [ ] **Sim-to-Real Transfer (Phase 7):** ROS 2 bridge, system identification, real Kuka iiwa 7 deployment

See [PLAN.md](PLAN.md) for the full detailed roadmap with status per sub-task.

---

## 🧪 Tests

```bash
pytest tests/smoke_test.py -v
```

Runs 13 tests covering environment reset, step, observation shape, reward sign, and termination conditions.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<sub>Built with MuJoCo · Stable-Baselines3 · PyTorch · Gymnasium</sub>
