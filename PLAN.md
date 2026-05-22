# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-22
> **Current State**: MuJoCo RL pipeline fully rebuilt and trained through all 3 curriculum phases. Reward went from **-56 → +680**. Agent can reach, grasp, lift, and place objects in the basket. Isaac Lab migration is Phase 4 (untouched).

> [!NOTE]
> **Security Issue (RESOLVED)**: `scripts/setup_cloud.sh` contained a hardcoded NGC API key. Moved to `archive/scripts/`. NGC API key should still be rotated on the NVIDIA dashboard.

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| MuJoCo scene (`scenes/pick_and_place_scene.xml`) | ✅ Working | 6-DOF arm, table, basket, gripper, magnetic weld |
| Robot wrapper (`robots/kuka_iiwa.py`) | ✅ Fixed | PD control, weld-based grasp, FK, **cached renderer** |
| Gym environment (`envs/pick_and_place_env.py`) | ✅ Rewritten | 20D obs, positive potential-based rewards, per-phase episode limits |
| Training script (`scripts/train.py`) | ✅ Rewritten | YAML config, GPU, EvalCallback, warm-start, reward logging |
| Trained models (`models/phase_*/`) | ✅ All 3 phases | Phase 0/1/2 complete — best models saved |
| Evaluation script (`scripts/evaluate_comprehensive.py`) | ⚠️ Works but buggy | VecNormalize not loaded correctly → early termination in some episodes |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | HSV detection, pixel→world — not integrated into obs yet |
| Web demo (`scripts/web_demo.py`) | ⚠️ Needs update | Needs update for new 20D obs format |
| Isaac Lab integration (`isaac_env/`) | ❌ Scaffolding only | env_cfg.py + mdp.py exist, untested, missing MDP funcs |
| Isaac Gym repos (`IsaacGymEnvs/`, `IsaacLab/`) | ❌ Cloned, gitignored | Standard upstream repos, no custom envs registered |
| Tests (`tests/smoke_test.py`) | ✅ Rewritten | 13 pytest tests, all passing |
| `tasks/` directory | ❌ Empty | Only `__init__.py` |
| Duplicate scripts | ✅ Archived | 11 scripts moved to `archive/scripts/` |

---

## Training Results (Session 1)

All training done on **NVIDIA RTX 3060 Laptop GPU**, PyTorch 2.12.0+cu126, MuJoCo 3.x.

| Phase | Task | Steps | Wall Time | Eval Reward | Reach | Grasp | Place |
|-------|------|-------|-----------|-------------|-------|-------|-------|
| **0 – REACH** | Move EE to object | 2M | 31 min | **+126** (was -56) | 15% | 0% | 0% |
| **1 – GRASP** | Reach + close gripper + lift | 2M | 33 min | **+424** peak | 20% | 20% | 0% |
| **2 – PLACE** | Full task end-to-end | 3M | 50 min | **+680** (max +2701) | 20% | 20% | **5%** |

> Warm-start chain: Phase 0 → 1 → 2 (each phase loads the previous best model).

**Final training config** (as of session end):
- `N_ENVS = 16`, `batch_size = 1024`, `gradient_steps = N_ENVS × 4 = 64`
- `buffer_size = 1_000_000`, `learning_rate = 3e-4`
- GPU FPS: ~1,800 (vs ~500 on CPU)

---

## Phase 1: Fix Training Pipeline ✅ COMPLETE

> **Goal**: Get working RL pipeline producing positive results.

### 1.1 Reward Function ✅
- [x] Identified flat negative penalty as root cause of 0% success
- [x] Switched to positive potential-based shaping: `baseline + shaping`
- [x] Store `_prev_dist` and `_prev_dist_to_basket` for temporal delta shaping
- [x] Increased `MAX_EPISODE_STEPS` to 200/300/400 per phase (was 100)
- [ ] Verify observation normalization — VecNormalize stats seem misaligned at eval time (see Phase 2.0)

### 1.2 Environment ✅
- [x] Observation: 8D → **20D** (added relative_pos, joint_pos, joint_vel, gripper_state)
- [x] Per-component reward logging to info dict (reach, grasp, lift, place bonuses)
- [x] Proper `close()` — cleans up robot and renderer resources
- [x] Cache `mujoco.Renderer` in `render_image()` — was creating/destroying per call
- [x] Add `close()` method to `KukaRobot`
- [ ] Action scaling verification — confirm arm can traverse workspace in per-phase step budgets
- [ ] Grasp weld threshold verification — test with actual EE geometry

### 1.3 Training Infrastructure ✅
- [x] `scripts/train.py` rewritten — YAML config, `EvalCallback`, `RewardInfoCallback`, warm-start
- [x] `configs/sac_config.yaml` wired up and actually loaded
- [x] `EvalCallback` runs every ~50K steps, saves best model
- [x] `RewardInfoCallback` logs all reward components to TensorBoard
- [x] Warm-start from previous phase model (Phase 0 → 1 → 2)
- [x] CUDA PyTorch installed (torch 2.12.0+cu126) — 3.4× speedup over CPU
- [x] N_ENVS doubled to 16, batch_size → 1024, gradient_steps → 64 for higher GPU utilization

### 1.4 Cleanup & Security ✅
- [x] 11 duplicate/obsolete scripts archived to `archive/scripts/`
- [x] NGC API key leak resolved (setup_cloud.sh archived)
- [x] `IsaacGymEnvs/`, `IsaacLab/`, venvs, models, logs all added to `.gitignore`
- [x] `isaac_pick_place_env_cfg.py` (root-level duplicate) moved to `archive/`

### 1.5 Testing ✅
- [x] `tests/smoke_test.py` rewritten with 13 proper pytest assertions (was a stub)

---

## Phase 2: Improve Robustness & Generalization

> **Goal**: Push place success from 5% → 30%+, add domain randomization for sim-to-real readiness.
> **Status**: Not started — pick up here next session.

### 2.0 Fix Evaluation Issues (Quick Win)
- [ ] Fix `scripts/evaluate_comprehensive.py` to load per-phase `VecNormalize` stats correctly
- [ ] Fix early-termination episodes (object falling at step 1-23) — VecNormalize mismatch
- [ ] Verify eval success rates match training-time rates

### 2.1 More Training (Quick Win)
- [ ] Retrain Phase 2 with 5M+ steps (was 3M) using new N_ENVS=16 / batch=1024 settings
- [ ] Monitor TensorBoard — `r_place_bonus` and `place_success` should start climbing
- [ ] Target: ≥20% place success rate at eval time

### 2.2 Domain Randomization
- [ ] Object size randomization (0.015–0.03m)
- [ ] Object mass randomization (0.05–0.2 kg)
- [ ] Friction randomization (±30%)
- [ ] Actuator noise — Gaussian noise on control signals
- [ ] Observation noise — small Gaussian to improve robustness

### 2.3 Richer Observations
- [x] Joint positions in observation ✅
- [x] Joint velocities ✅
- [x] Relative position (object - EE) ✅
- [ ] Add distance-to-basket as explicit feature
- [ ] Goal-conditioned observations (pass target pos as part of obs)

### 2.4 Improved Grasping
- [ ] Replace magnetic weld with contact-force-based grasping (more realistic)
- [ ] Check `mujoco.mj_contactForce()` for gripper-object contacts
- [ ] Add force/torque sensing via MuJoCo `sensordata`
- [ ] Gripper compliance — PD control with spring-damper model

### 2.5 Better Reward Engineering
- [ ] Hindsight Experience Replay (HER) — SB3 HER wrapper for sparse reward settings
- [ ] Curriculum auto-progression — auto-advance phase when success > 70% over 100 episodes

---

## Phase 3: Vision-Based Policy

> **Goal**: Train policies from camera observations instead of privileged state.
> **Prerequisites**: Phase 2 state-based policy working well (≥30% place success).

### 3.1 Image Observations
- [ ] Enable `use_vision=True` path in `PickAndPlaceEnv` (currently broken)
- [ ] 64×64 RGB (or grayscale) stacked frames observation space
- [ ] CNN feature extractor — SB3 `CnnPolicy` or custom `NatureCNN` / `ResNet` encoder
- [ ] Frame stacking (3–4 frames) for temporal information

### 3.2 Depth Camera
- [ ] Depth rendering in `sensors/camera.py` using MuJoCo depth buffer
- [ ] RGB-D 4-channel input
- [ ] (Optional) Point cloud → PointNet-style policy

### 3.3 State-Vision Distillation
- [ ] Train teacher policy on privileged state (Phase 2 result)
- [ ] Train student policy on images, supervised by teacher (DAgger-style)
- [ ] Compare end-to-end RL vs. distillation

### 3.4 Camera Calibration
- [ ] Fix `pixel_to_world` mapping — validate pinhole model against MuJoCo ground truth
- [ ] Proper intrinsic/extrinsic calibration from MuJoCo camera params
- [ ] Multi-view setup (add side camera)

---

## Phase 4: Isaac Lab Migration

> **Goal**: GPU-accelerated physics with 1000s of parallel envs for 10,000+ FPS.
> **Priority**: After MuJoCo pipeline is proven (Phases 1–2 complete ✅).

### 4.1 Fix Isaac Lab Environment
- [ ] Fix MDP function signatures in `isaac_env/mdp.py` — `env._prev_dist` not set anywhere
- [ ] Implement missing MDP functions: `joint_pos_delta_action`, `gripper_action`, `ee_position`, `object_position`, `gripper_state`, `time_out`, reward functions
- [ ] Decide robot: keep UR10e (Isaac Lab native support) or port KUKA iiwa from MuJoCo
- [ ] Test scene loading and rendering in Isaac Lab
- [ ] Validate physics vs. MuJoCo (object dynamics, grasping)

### 4.2 Port Training
- [ ] Register custom environment with Isaac Lab gym registry
- [ ] Use RSL-RL or RL Games (GPU-native) instead of SB3
- [ ] Scale to 128+ parallel environments — target 10,000+ FPS
- [ ] Warm-start from MuJoCo Phase 2 policy weights

### 4.3 Isaac Sim Rendering
- [ ] RTX photorealistic rendering for vision policies
- [ ] Synthetic data generation via Isaac Sim Replicator
- [ ] MuJoCo XML → USD scene export

### 4.4 Cleanup
- [ ] Remove large venv clones (`isaaclab-venv/`, `isaacsim-venv-3.11/`) — already gitignored, delete locally
- [ ] Use pip install or git submodules for IsaacLab/IsaacGymEnvs instead of full clones

---

## Phase 5: Advanced RL Algorithms

> **Goal**: Move beyond vanilla SAC. Try algorithms better suited for manipulation.
> **Can run in parallel with Phase 4.**

### 5.1 Algorithm Exploration
- [ ] PPO with clipping — often more stable with shaped rewards
- [ ] TD3 — compare with SAC for continuous control
- [ ] DDPG + HER — goal-conditioned sparse reward
- [ ] DrQ-v2 — data-regularized Q-learning for image-based RL
- [ ] Dreamer v3 — model-based RL for sample efficiency

### 5.2 Hierarchical RL
- [ ] Sub-policies for REACH → GRASP → LIFT → TRANSPORT → PLACE
- [ ] HIRO / HAC — hierarchical actor-critic

### 5.3 Imitation Learning Hybrid
- [ ] Generate IK demonstrations (expert trajectories)
- [ ] Behavior cloning pre-training, then RL fine-tuning
- [ ] GAIL / DAgger

---

## Phase 6: Infrastructure & Code Quality

> **Goal**: Production-quality codebase, CI/CD, reproducibility.
> **Status**: Ongoing.

### 6.1 Testing
- [x] 13 smoke tests (env reset, step, reward, obs shape) ✅
- [ ] Unit tests for `KukaRobot` (FK, action, weld grasp/release)
- [ ] Unit tests for `PickAndPlaceEnv` (reset, step, reward, termination)
- [ ] Integration test — full training loop for 1000 steps, no crash
- [ ] Regression test — known-good checkpoint, verify eval score doesn't degrade

### 6.2 Code Cleanup
- [x] 11 duplicate scripts archived ✅
- [x] `configs/sac_config.yaml` wired up ✅
- [x] `.gitignore` updated — models, logs, venvs, Isaac repos excluded ✅
- [ ] Add `__init__.py` exports to all packages
- [ ] Type hints everywhere
- [ ] Docstring coverage for all public functions

### 6.3 Logging & Experiment Tracking
- [ ] Weights & Biases integration (`wandb` is in requirements.txt but unused)
- [ ] Save full config (hyperparams + env settings + git hash) with each run
- [ ] Model registry — track versions with metadata (phase, timesteps, success rate)

### 6.4 CI/CD
- [ ] GitHub Actions — smoke test + linting on every push
- [ ] Pre-commit hooks — `black`, `isort`, `flake8`, `mypy`
- [ ] Docker container — reproducible training environment

### 6.5 Documentation
- [ ] Update `README.md` — currently 6 lines. Add overview, install, usage, architecture
- [ ] Training guide — reproduce training from scratch
- [ ] Architecture diagram — env ↔ robot ↔ sim ↔ policy data flow (Mermaid)

---

## Phase 7: Sim-to-Real Transfer

> **Goal**: Deploy trained policy on a physical 6-DOF arm.
> **Prerequisites**: Phases 2–4 complete. Domain randomization working.

### 7.1 Hardware Setup
- [ ] Select physical robot — KUKA iiwa 7 R800 or UR5e/UR10e
- [ ] Gripper — Robotiq 2F-85 or custom
- [ ] Camera — Intel RealSense D435 or ZED 2 (RGB-D)
- [ ] Workspace matching simulation dimensions

### 7.2 ROS 2 Bridge
- [ ] ROS 2 package wrapping policy inference node
- [ ] Joint command interface → `JointTrajectoryController`
- [ ] Sensor subscribers (camera + joint state → observation vector)
- [ ] Safety layer (velocity/torque limits, workspace bounds, e-stop)

### 7.3 Sim-to-Real Techniques
- [ ] System identification — measure real robot dynamics, tune MuJoCo model
- [ ] Automatic Domain Randomization (ADR)
- [ ] Real-world fine-tuning with safety constraints
- [ ] Residual policy learning

---

## Priority Order

```
Phase 1 (Fix Training) ──────────────────────► ✅ COMPLETE
    │
    ├── Phase 2 (Robustness) ────────────────► ← START HERE NEXT SESSION
    │       │
    │       ├── Phase 3 (Vision) ────────────► After state-based works
    │       │
    │       └── Phase 5 (Algorithms) ────────► Can parallelize
    │
    ├── Phase 4 (Isaac Lab) ─────────────────► After MuJoCo pipeline proven
    │
    ├── Phase 6 (Infrastructure) ────────────► Ongoing
    │
    └── Phase 7 (Sim-to-Real) ──────────────► After everything else
```

---

## Next Session — Pick Up Here

1. **Fix eval script VecNormalize bug** — episodes dying at step 1-23 are a measurement artifact
2. **Retrain Phase 2 with 5M steps** — N_ENVS=16, batch=1024 settings ready to go
3. **Add domain randomization** — object mass/size/friction variation
4. **Update README.md** — at minimum add install + training instructions
