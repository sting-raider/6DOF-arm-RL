# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-27 (Session 5 — Training Tuning & Phase 2 Prep)
> **Current State**: Phase 0 (REACH) retraining (Run 49) actively running at 8192 envs. Corrected robot base position to ground (0,0,0), lifted start pose, increased action scale to 0.07, and relaxed eval threshold to 8cm. Reach reward currently plateauing around 0.50 (~14cm distance).

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| Isaac Lab environment (`isaac_env/`) | ✅ Active | 8192 parallel envs, RSL-RL, GPU-native |
| Robot wrapper | ✅ Working | UR10e, PD control, FK, active joint filtering |
| Gym environment | ✅ Stable | 29D obs, potential-based rewards |
| Training pipeline | ✅ Stabilized | Standard running normalization restored |
| Phase 0 model | 🔄 Training | Run 49 active (8192 envs, plateau at ~0.50 reach) |
| Phase 1 model | ⏳ Queued | Will train from Phase 0 best model with corrected geometry |
| Phase 2 model | ⏳ Queued | Pending hollow basket with collisions |
| Evaluation script | ✅ Fixed | Local coordinate frames corrected, debug distances added |
| Domain randomization | ✅ Implemented | Mass/size/friction/obs noise. Active by default. |
| Web demo (`scripts/web_demo.py`) | ✅ Updated | 20D/29D format compatible, live dashboard |
| CI/CD (`.github/workflows/ci.yml`) | ✅ Active | Smoke tests + flake8 on push to main |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | Not integrated into obs yet |
| Isaac Sim rendering | ⚠️ Scaffolding only | RTX rendering pipeline pending |
| Tests (`tests/smoke_test.py`) | ✅ Passing | 13 pytest tests passing |
| README.md | ✅ Updating | Synced to 29D obs and corrected thresholds |

---

## Training Results

All training on **RTX 3060 Laptop GPU**, Isaac Lab + RSL-RL.

| Phase | Task | Steps | Steps/sec | Reach | Grasp | Place | Status |
|---|---|---|---|---|---|---|---|
| **0 – REACH** | Move EE to object | 500 iter | ~37k | 0.50 | — | — | 🔄 Training (Run 49) |
| **1 – GRASP** | Reach + close + lift | TBD | TBD | TBD | TBD | — | ⏳ Queued |
| **2 – PLACE** | Full task end-to-end | TBD | TBD | TBD | TBD | TBD | ⏳ Queued |

---

## Phase 1: Fix Training Pipeline ✅ COMPLETE

- [x] **Reward function**: positive potential-based shaping, per-component logging
- [x] **Environment**: 29D complete state observability, per-phase episode limits
- [x] **Training infrastructure**: YAML config, EvalCallback, warm-start, GPU
- [x] **Cleanup & security**: scripts archived, NGC API key removed, old logs/models purged

---

## Phase 2: Improve Robustness & Generalization 🔄 IN PROGRESS

> **Goal**: Achieve $\ge 90\%$ REACH success and $\ge 80\%$ GRASP & PLACE success in Isaac Lab under domain randomization.

### 2.1 Fine-tune Phase 0 (REACH) 🔄 IN PROGRESS
- [x] Correct robot base position to (0,0,0) (ground level)
- [x] Lift starting pose to reach over table (`shoulder_lift=-1.4`, `elbow=1.7`)
- [x] Bump action scale from 0.05 to 0.07 for faster motion
- [x] Increase PhysX buffer sizes for 8192 envs (`8M` found_lost, `32K` pairs)
- [ ] Push reach reward past 0.50 plateau (may need reward shaping tweaks or larger action scale)
- [ ] Evaluate best checkpoint and check real EE-to-object distances

### 2.2 Phase 1 → Phase 2 Pipeline ⏳ QUEUED
- [ ] Warm-start Phase 1 from retrained stable Phase 0
- [ ] **Phase 2 Prep**: Replace visual-proxy basket with a 5-part hollow rigid-body basket with collisions enabled
- [ ] Warm-start Phase 2 from retrained stable Phase 1
- [ ] Target: $\ge 80\%$ grasp and place success

### 2.3 Domain Randomization ✅ COMPLETE
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
