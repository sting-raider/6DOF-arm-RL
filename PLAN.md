# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-24 (Session 3 — Isaac Lab Migration Complete)
> **Current State**: Isaac Lab is now PRIMARY. MuJoCo archived. Phase 0 Isaac model trained (0.97 reach, 26.7M steps, ~64k FPS on RTX 3060). Phase 1 & 2 models exist. About to retrain Phase 0 from scratch in Isaac Lab. CI/CD active on GitHub.

> [!NOTE]
> **Security Issue (RESOLVED)**: `scripts/setup_cloud.sh` contained a hardcoded NGC API key. Archived. Key should still be rotated on NVIDIA dashboard.

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| Isaac Lab environment (`isaac_env/`) | ✅ Active | 32 parallel envs, RSL-RL, GPU-native at 64k+ FPS |
| Robot wrapper | ✅ Working | KUKA iiwa, PD control, FK, Isaac Lab integration |
| Gym environment | ✅ Rewritten | 20D obs, potential-based rewards, DR, per-phase limits |
| Training pipeline | ✅ Working | RSL-RL + SB3 SAC, YAML config, GPU, EvalCallback |
| Phase 0 model (Isaac Lab) | ✅ Trained | 26.7M steps, 0.97 reach success, ~64k steps/sec |
| Phase 1 model | ✅ Exists | Grasp model — retrain planned |
| Phase 2 model | ✅ Exists | Place model — retrain planned |
| Evaluation script | ✅ Fixed | VecNormalize loading fixed, success flags persist |
| Domain randomization | ✅ Implemented | Mass/size/friction/obs noise. Active by default. |
| Web demo (`scripts/web_demo.py`) | ✅ Updated | 20D obs format, live dashboard, VecNormalize loading |
| CI/CD (`.github/workflows/ci.yml`) | ✅ Active | Smoke tests + flake8 on push to main |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | HSV detection, pixel→world — not integrated into obs yet |
| Isaac Sim rendering | ⚠️ Scaffolding only | RTX rendering pipeline not yet configured |
| Tests (`tests/smoke_test.py`) | ✅ Rewritten | 13 pytest tests, all passing |
| README.md | ✅ Comprehensive | Install, training, results, architecture |

---

## Training Results

All training on **RTX 3060 Laptop GPU**, Isaac Lab + RSL-RL, 32 parallel envs.

| Phase | Task | Steps | Steps/sec | Reach | Grasp | Place |
|---|---|---|---|---|---|---|
| **0 – REACH** | Move EE to object | 26.7M | 64,142 | **0.97** | — | — |
| **1 – GRASP** | Reach + close gripper + lift | TBD | TBD | TBD | TBD | — |
| **2 – PLACE** | Full task end-to-end | TBD | TBD | TBD | TBD | TBD |

> **Next**: Retrain Phase 0 from scratch in Isaac Lab with improved hyperparameters, then warm-start Phase 1 → Phase 2.

**Training config:**
- `N_ENVS = 32`, RSL-RL PPO
- Domain randomization: ON (mass 0.05–0.20 kg, size 0.015–0.030 m, friction 0.5–2.0×, obs noise σ=0.002)

---

## Phase 1: Fix Training Pipeline ✅ COMPLETE

> **Goal**: Get working RL pipeline producing positive results. Foundational work enabling all subsequent phases.

### 1.1–1.4 All Fixes ✅
- [x] Reward function: positive potential-based shaping, per-component logging
- [x] Environment: 20D observations, per-phase episode limits, proper close()
- [x] Training infrastructure: YAML config, EvalCallback, warm-start, GPU
- [x] Cleanup & security: scripts archived, NGC API key removed

### 1.5 Training ✅
- [x] Phase 0 (REACH) — trained in Isaac Lab at 26.7M steps, 0.97 reach
- [x] Phase 1 (GRASP) — model exists, retrain planned
- [x] Phase 2 (PLACE) — model exists, retrain planned

### 1.6 Post-Phase-1 Enhancements ✅
- [x] Eval script VecNormalize fix
- [x] Comprehensive README
- [x] Domain randomization (mass 0.05–0.20 kg, size 0.015–0.030 m, friction 0.5–2.0×, obs noise σ=0.002)
- [x] Web demo updated for 20D observations
- [x] Phase 2 reward fix (basket-dist baseline swap when grasped)
- [x] GitHub Actions CI (smoke tests + flake8)

---

## Phase 2: Improve Robustness & Generalization 🔄 IN PROGRESS

> **Goal**: Push Isaac Lab success rates — maintain ≥0.97 reach, achieve ≥0.80 grasp & place with domain randomization for sim-to-real readiness.

### 2.0 Fix Evaluation Issues ✅
- [x] VecNormalize loading fixed in eval script
- [x] Success flags persist across episode boundaries

### 2.1 Retrain Phase 0 (Isaac Lab)
- [ ] Retrain Phase 0 from scratch in Isaac Lab with improved hyperparameters
- [ ] Maintain ≥0.97 reach success at 32+ envs
- [ ] Target 100k+ FPS throughput

### 2.2 Phase 1 → Phase 2 Pipeline
- [ ] Warm-start Phase 1 from retrained Phase 0
- [ ] Warm-start Phase 2 from retrained Phase 1
- [ ] Target: ≥0.80 grasp and place success

### 2.3 Domain Randomization ✅
- [x] Object size randomization (0.015–0.030 m)
- [x] Object mass randomization (0.05–0.20 kg)
- [x] Friction randomization (0.5×–2.0×)
- [x] Observation noise (Gaussian σ=0.002)

### 2.4 Evaluation & Deliverables
- [ ] Run comprehensive evaluation with all 3 phases in Isaac Lab
- [ ] Generate evaluation videos via Isaac Sim
- [ ] Push Phase 0/1/2 models + results to GitHub

---

## Phase 3: Vision-Based Policy

> **Prerequisite**: Phase 2 Isaac Lab state-based policy ≥0.80 place success.

### 3.1 Image Observations
- [ ] Enable `use_vision=True` path
- [ ] 64×64 RGB stacked frames, CNN policy
- [ ] Frame stacking (3–4 frames)

### 3.2 Depth Camera
- [ ] Depth rendering via Isaac Sim
- [ ] RGB-D 4-channel input

### 3.3 State-Vision Distillation
- [ ] Teacher (state) → Student (vision) via DAgger

### 3.4 Camera Calibration
- [ ] Fix pixel_to_world mapping
- [ ] Proper intrinsic/extrinsic calibration via Isaac Sim

---

## Phase 4: Isaac Lab Migration ✅ COMPLETE

> **Goal**: Migrate to Isaac Lab for GPU-native performance, higher throughput, and photorealistic rendering.

### 4.1 Isaac Lab Environment ✅
- [x] MDP functions implemented in `isaac_env/mdp.py`
- [x] Custom environment registered with Isaac Lab gym registry
- [x] Scene loading and configuration working
- [x] 32 parallel envs, GPU-native simulation

### 4.2 Training Pipeline ✅
- [x] RSL-RL training pipeline operational
- [x] 64k+ FPS throughput on RTX 3060
- [x] Warm-start chain (Phase 0 → 1 → 2)
- [x] Phase 0 model trained to 0.97 reach success

### 4.3 Isaac Sim Rendering ⚠️ PENDING
- [ ] RTX photorealistic rendering for vision policies
- [ ] Synthetic data generation via Isaac Sim Replicator
- [ ] USD scene management

---

## Phase 5: Advanced RL Algorithms

> **Can run in parallel with Phase 3 (Vision).**

- [ ] PPO, TD3, DDPG+HER comparison in Isaac Lab
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

**Prerequisites**: Phases 2–4 complete. Domain randomization working. Isaac Lab pipeline mature.

- [ ] Hardware selection (KUKA iiwa / UR5e / UR10e)
- [ ] ROS 2 bridge for policy inference
- [ ] Real-world fine-tuning with safety constraints

---

## Priority Graph

```
Phase 1 (Fix Pipeline) ─────────────────► ✅ COMPLETE
    │
    └── Phase 2 (Robustness) ───────────► 🔄 IN PROGRESS (Isaac Lab retrain)
            │
            ├── Phase 3 (Vision) ───────► After state-based policy solid
            ├── Phase 4 (Isaac Lab) ────► ✅ COMPLETE
            └── Phase 5 (Algorithms) ───► Can parallelize with Phase 3

Phase 6 (Infrastructure) ───────────────► Ongoing background
Phase 7 (Sim-to-Real) ─────────────────► After everything above
```
