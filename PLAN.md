# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-27 (Session 4 — PPO Training Stabilization & Normalization Saga Resolved)  
> **Current State**: Retraining Phase 0 (REACH) from scratch with complete 29D state observability and stable running observation normalization. Active joint filtering completely resolved Robotiq mimic joint velocity spikes. 150-iteration sanity check succeeded (Value loss: 0.0130). Full retraining run v14 is actively running at 4096 environments.

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| Isaac Lab environment (`isaac_env/`) | ✅ Active | 4096 parallel envs, RSL-RL, GPU-native |
| Robot wrapper | ✅ Working | KUKA iiwa, PD control, FK, active joint filtering |
| Gym environment | ✅ Stable | 29D obs (complete observability), potential-based rewards |
| Training pipeline | ✅ Stabilized | Restored standard unfrozen observation normalizers |
| Phase 0 model (Isaac Lab) | 🔄 Retraining | Retraining in progress (v14); sanity check passed |
| Phase 1 model | ⏳ Queued | Grasp model — queued for warm-start |
| Phase 2 model | ⏳ Queued | Place model — queued for warm-start |
| Evaluation script | ✅ Fixed | Normalizer config synced, local coordinate frames corrected |
| Domain randomization | ✅ Implemented | Mass/size/friction/obs noise. Active by default. |
| Web demo (`scripts/web_demo.py`) | ✅ Updated | 20D/29D format compatible, live dashboard |
| CI/CD (`.github/workflows/ci.yml`) | ✅ Active | Smoke tests + flake8 on push to main |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | Not integrated into obs yet |
| Isaac Sim rendering | ⚠️ Scaffolding only | RTX rendering pipeline pending |
| Tests (`tests/smoke_test.py`) | ✅ Passing | 13 pytest tests passing |
| README.md | ✅ Updating | Synced to normalization resolution and 29D obs |

---

## Training Results

All training on **RTX 3060 Laptop GPU**, Isaac Lab + RSL-RL, 4096 parallel envs.

| Phase | Task | Steps | Steps/sec | Reach | Grasp | Place | Status |
|---|---|---|---|---|---|---|---|
| **0 – REACH** | Move EE to object | 1500 iter | ~45k | TBD | — | — | 🔄 Retraining (v14) |
| **1 – GRASP** | Reach + close + lift | TBD | TBD | TBD | TBD | — | ⏳ Queued |
| **2 – PLACE** | Full task end-to-end | TBD | TBD | TBD | TBD | TBD | ⏳ Queued |

> **Next**: Complete Phase 0 retraining, verify success rate is $\ge 90\%$, then warm-start Phase 1 → Phase 2.

---

## Phase 1: Fix Training Pipeline ✅ COMPLETE

- [x] **Reward function**: positive potential-based shaping, per-component logging
- [x] **Environment**: 29D complete state observability, per-phase episode limits
- [x] **Training infrastructure**: YAML config, EvalCallback, warm-start, GPU
- [x] **Cleanup & security**: scripts archived, NGC API key removed

---

## Phase 2: Improve Robustness & Generalization 🔄 IN PROGRESS

> **Goal**: Achieve $\ge 90\%$ REACH success and $\ge 80\%$ GRASP & PLACE success in Isaac Lab under domain randomization.

### 2.1 Resolve Normalizer Saga & Retrain Phase 0 ✅ IN PROGRESS
- [x] Diagnose Robotiq mimic joint velocity spikes skewing normalizer variance
- [x] Filter observation space to strictly 6 active arm joints, removing mimic joints
- [x] Re-enable running observation normalization (`obs_normalization: True`) without freezing
- [x] Succeeded in 150-iteration sanity check (Value loss: 0.0130, Reach reward: 0.3243)
- [x] Launch full 1500-iteration Phase 0 retraining at 4096 envs (v14 active)

### 2.2 Phase 1 → Phase 2 Pipeline
- [ ] Warm-start Phase 1 from retrained stable Phase 0
- [ ] Warm-start Phase 2 from retrained stable Phase 1
- [ ] Target: $\ge 80\%$ grasp and place success

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
- [ ] Enable `use_vision=True` path
- [ ] 64×64 RGB stacked frames, CNN policy
- [ ] Frame stacking (3–4 frames)
- [ ] Depth rendering via Isaac Sim
- [ ] State-Vision Distillation (Teacher → Student via DAgger)
- [ ] Camera intrinsic/extrinsic calibration via Isaac Sim
