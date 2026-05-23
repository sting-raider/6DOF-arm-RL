# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-23 (Session 2 — Hermes continuation)
> **Current State**: MuJoCo RL pipeline fully rebuilt. Phase 0+1 models exist. Phase 2 model MISSING (needs retrain). Domain randomization implemented. CI/CD running on GitHub. Eval script fixed.

> [!NOTE]
> **Security Issue (RESOLVED)**: `scripts/setup_cloud.sh` contained a hardcoded NGC API key. Archived. Key should still be rotated on NVIDIA dashboard.

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| MuJoCo scene (`scenes/pick_and_place_scene.xml`) | ✅ Working | 6-DOF arm, table, basket, gripper, magnetic weld |
| Robot wrapper (`robots/kuka_iiwa.py`) | ✅ Fixed | PD control, weld-based grasp, FK, cached renderer |
| Gym environment (`envs/pick_and_place_env.py`) | ✅ Rewritten | 20D obs, positive potential-based rewards, DR, per-phase limits |
| Training script (`scripts/train.py`) | ✅ Rewritten | YAML config, GPU, EvalCallback, warm-start, reward logging |
| Phase 0 model (`models/phase_0/`) | ✅ Trained | 2M steps, eval reward +126, 15% reach success |
| Phase 1 model (`models/phase_1/`) | ✅ Trained | 2M steps, eval reward +424 peak, 20% grasp success |
| Phase 2 model (`models/phase_2/`) | ❌ MISSING | Was deleted for retrain — retrain killed at 5.3M/5M steps |
| Evaluation script (`scripts/evaluate_comprehensive.py`) | ✅ Fixed | VecNormalize loading fixed, success flags persist |
| Domain randomization | ✅ Implemented | Mass/size/friction/obs noise. Active by default. |
| Web demo (`scripts/web_demo.py`) | ✅ Updated | 20D obs format, live dashboard, VecNormalize loading |
| CI/CD (`.github/workflows/ci.yml`) | ✅ Active | Smoke tests + flake8 on push to main |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | HSV detection, pixel→world — not integrated into obs yet |
| Isaac Lab integration (`isaac_env/`) | ❌ Scaffolding only | env_cfg.py + mdp.py exist, untested |
| Tests (`tests/smoke_test.py`) | ✅ Rewritten | 13 pytest tests, all passing |
| README.md | ✅ Comprehensive | Install, training, results, architecture |

---

## Training Results

All training on **RTX 3060 Laptop GPU**, PyTorch 2.12.0+cu126, MuJoCo 3.x, SB3 SAC.

| Phase | Task | Steps | Wall Time | Eval Reward | Reach | Grasp | Place |
|-------|------|-------|-----------|-------------|-------|-------|-------|
| **0 – REACH** | Move EE to object | 2M | 31 min | **+126** (was -56) | 15% | 0% | 0% |
| **1 – GRASP** | Reach + close gripper + lift | 2M | 33 min | **+424** peak | 20% | 20% | 0% |
| **2 – PLACE** | Full task end-to-end | **NEEDS RETRAIN** | — | — | — | — | — |

> Warm-start chain: Phase 0 → 1 → 2 (each phase loads the previous best model).

**Training config:**
- `N_ENVS = 16`, `batch_size = 1024`, `gradient_steps = 64`
- `buffer_size = 1,000,000`, `learning_rate = 3e-4`
- Domain randomization: ON (mass 0.05-0.20kg, size 0.015-0.030m, friction 0.5-2.0×, obs noise σ=0.002)

---

## Phase 1: Fix Training Pipeline ✅ COMPLETE

> **Goal**: Get working RL pipeline producing positive results.

### 1.1–1.4 All Fixes ✅
- [x] Reward function: positive potential-based shaping, per-component logging
- [x] Environment: 20D observations, per-phase episode limits, proper close()
- [x] Training infrastructure: YAML config, EvalCallback, warm-start, GPU
- [x] Cleanup & security: scripts archived, NGC API key removed

### 1.5 Training ✅
- [x] Phase 0 (REACH) — 2M steps, 31 min, +126 eval reward
- [x] Phase 1 (GRASP) — 2M steps, 33 min, +424 peak eval reward
- [~] Phase 2 (PLACE) — model was deleted; needs retrain with fixed reward

### 1.6 Post-Phase-1 Enhancements ✅
- [x] Eval script VecNormalize fix
- [x] Comprehensive README
- [x] Domain randomization (mass 0.05-0.20kg, size 0.015-0.030m, friction 0.5-2.0×, obs noise σ=0.002)
- [x] Web demo updated for 20D observations
- [x] Phase 2 reward fix (basket-dist baseline swap when grasped)
- [x] GitHub Actions CI (smoke tests + flake8)

---

## Phase 2: Improve Robustness & Generalization (NEXT)

> **Goal**: Push place success from 5% → 20%+ with domain randomization for sim-to-real readiness.

### 2.0 Fix Evaluation Issues ✅
- [x] VecNormalize loading fixed in eval script
- [x] Success flags persist across episode boundaries

### 2.1 Retrain Phase 2
- [ ] Retrain from scratch, warm-start Phase 1 → Phase 2
- [ ] 5M+ steps with N_ENVS=16, batch_size=1024
- [ ] Domain randomization active during training
- [ ] Target: ≥20% place success at eval time

### 2.2 Domain Randomization ✅
- [x] Object size randomization (0.015–0.030m)
- [x] Object mass randomization (0.05–0.20 kg)
- [x] Friction randomization (0.5×–2.0×)
- [x] Observation noise (Gaussian σ=0.002)

### 2.3 Evaluation & Deliverables
- [ ] Run comprehensive evaluation with all 3 phases
- [ ] Generate evaluation videos
- [ ] Push Phase 2 model + results to GitHub

---

## Phase 3: Vision-Based Policy

> **Prerequisite**: Phase 2 state-based policy ≥20% place success.

### 3.1 Image Observations
- [ ] Enable `use_vision=True` path (currently broken)
- [ ] 64×64 RGB stacked frames, CNN policy
- [ ] Frame stacking (3–4 frames)

### 3.2 Depth Camera
- [ ] Depth rendering via MuJoCo depth buffer
- [ ] RGB-D 4-channel input

### 3.3 State-Vision Distillation
- [ ] Teacher (state) → Student (vision) via DAgger

### 3.4 Camera Calibration
- [ ] Fix pixel_to_world mapping
- [ ] Proper intrinsic/extrinsic calibration

---

## Phase 4: Isaac Lab Migration

> **Priority**: After MuJoCo pipeline proven (Phase 1 done ✅, Phase 2 pending).

### 4.1 Fix Isaac Lab Environment
- [ ] Fix MDP functions in `isaac_env/mdp.py`
- [ ] Implement missing MDP functions
- [ ] Decide robot: UR10e or KUKA iiwa
- [ ] Test scene loading in Isaac Lab

### 4.2 Port Training
- [ ] Register custom env with Isaac Lab gym registry
- [ ] Use RSL-RL for GPU-native training
- [ ] Scale to 128+ parallel envs → 10,000+ FPS
- [ ] Warm-start from MuJoCo Phase 2 policy

### 4.3 Isaac Sim Rendering
- [ ] RTX photorealistic rendering for vision policies
- [ ] Synthetic data generation via Isaac Sim Replicator
- [ ] MuJoCo XML → USD scene export

---

## Phase 5: Advanced RL Algorithms

> **Can run in parallel with Phase 4.**

- [ ] PPO, TD3, DDPG+HER comparison
- [ ] DrQ-v2 for image-based RL
- [ ] Hierarchical RL (HIRO/HAC)
- [ ] Behavior cloning pre-training + RL fine-tuning

---

## Phase 6: Infrastructure & Code Quality

| Area | Status | Priority |
|---|---|---|
| ✅ 13 smoke tests | ✅ Done | — |
| ✅ CI/CD (GitHub Actions) | ✅ Done | — |
| ✅ README | ✅ Done | — |
| Unit tests (robot, env) | ❌ Not started | Medium |
| Type hints | ❌ Not started | Low |
| W&B experiment tracking | ❌ Not started | Low |
| Pre-commit hooks | ❌ Not started | Low |

---

## Phase 7: Sim-to-Real Transfer

**Prerequisites**: Phases 2–4 complete. Domain randomization working.

- [ ] Hardware selection (KUKA iiwa / UR5e / UR10e)
- [ ] ROS 2 bridge for policy inference
- [ ] Real-world fine-tuning with safety constraints

---

## Priority Graph

```
Phase 1 (Fix Pipeline) ─────────────────► ✅ COMPLETE
    │
    └── Phase 2 (Robustness) ───────────► 🔴 RETRAIN PHASE 2 NOW
            │
            ├── Phase 3 (Vision) ───────► After state-based policy solid
            ├── Phase 4 (Isaac Lab) ────► After MuJoCo pipeline proven
            └── Phase 5 (Algorithms) ───► Can parallelize with 3/4

Phase 6 (Infrastructure) ───────────────► Ongoing background
Phase 7 (Sim-to-Real) ─────────────────► After everything above
```
