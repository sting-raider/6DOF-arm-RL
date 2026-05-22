# 6-DOF Arm Pick-and-Place — Project Roadmap

> **Last Updated**: 2026-05-22
> **Current State**: MuJoCo pipeline **rebuilt** — new reward function (positive potential-based), richer 20D observations, per-phase episode lengths. Training Phase 0 (REACH) in progress. Isaac Lab scaffolding untouched (Phase 4).

> [!NOTE]
> **Security Issue (RESOLVED)**: `scripts/setup_cloud.sh` (with hardcoded NGC API key) has been moved to `archive/scripts/`. Key should still be rotated on NVIDIA NGC.

---

## Project Health Summary

| Component | Status | Notes |
|---|---|---|
| MuJoCo scene (`scenes/pick_and_place_scene.xml`) | ✅ Working | 6-DOF arm, table, basket, gripper, magnetic weld |
| Robot wrapper (`robots/kuka_iiwa.py`) | ✅ Fixed | PD control, weld-based grasp, FK, **cached renderer** |
| Gym environment (`envs/pick_and_place_env.py`) | ✅ Rewritten | **20D obs**, positive rewards, per-phase episode limits |
| Training script (`scripts/train.py`) | ✅ Rewritten | YAML config, EvalCallback, warm-start, reward logging |
| Trained models (`models/phase_*/`) | 🔄 Training... | Phase 0 (REACH) in progress |
| Vision pipeline (`sensors/camera.py`) | ⚠️ Built, unused | HSV detection, pixel→world, not integrated into obs |
| Web demo (`scripts/web_demo.py`) | ⚠️ Needs update | Needs update for new 20D obs format |
| Isaac Lab integration (`isaac_env/`) | ❌ Scaffolding only | env_cfg.py + mdp.py exist, untested, missing MDP funcs |
| Isaac Gym integration (`IsaacGymEnvs/`, `IsaacLab/`) | ❌ Cloned repos only | Standard upstream repos, no custom envs registered |
| Tests (`tests/smoke_test.py`) | ✅ Rewritten | **13 pytest tests, all passing** |
| `tasks/` directory | ❌ Empty | Only `__init__.py` |
| `analysis/` directory | ❌ Empty | Only `__init__.py` |
| Duplicate scripts | ✅ Archived | 11 scripts moved to `archive/scripts/` |

### Critical Issue: Trained Models Produce 0% Success

The evaluation results show:
```
Mean reward: -56.12 ± 10.46
Success rates: reach=0/5, grasp=0/5, place=0/5
```
This means the current SAC training pipeline is **not producing useful policies**. This is the #1 priority to fix.

### Additional Bugs Discovered

| Bug | File | Details |
|-----|------|---------|
| `train_cloud.py` EE is hardcoded | `scripts/train_cloud.py` | `_get_ee_pos()` returns `[0.35, 0.0, 1.0]` always — agent can never learn |
| `train_v2.py` wrong prim | `scripts/train_v2.py` | `ee_prim` points to `/World/Ground`; xform ops accumulate |
| Renderer created/destroyed per frame | `robots/kuka_iiwa.py` | ✅ **FIXED** — renderer now cached |
| Camera euler may be wrong | `scenes/pick_and_place_scene.xml:130` | `-1.5708 0 0` rotates around X → needs verification |
| Joint 6 never actuated | `robots/kuka_iiwa.py:62` | `_target_joint_pos = HOME_ANGLES[:5]` — only 5 joints controlled |
| `WELD_BREAK_DISTANCE` unused | `utils/constants.py:25` | Defined but never referenced |
| `FRAME_SKIP` unused | `utils/constants.py:10` | Defined but never imported |
| `close()` is no-op | `envs/pick_and_place_env.py` | ✅ **FIXED** — now cleans up robot/renderer |
| `use_vision=True` broken | `envs/pick_and_place_env.py` | Still broken (Phase 3 work) |
| Config YAML not loaded | `scripts/train.py` | ✅ **FIXED** — now loads from sac_config.yaml |
| Buggy Isaac Sim scripts | `scripts/train_cloud.py` etc. | ✅ **ARCHIVED** to `archive/scripts/` |

---

## Phase 1: Fix Training Pipeline (CRITICAL)

> **Goal**: Get Phase 0 (REACH) to ≥80% success, Phase 1 (GRASP) to ≥50%.
> **Timeline**: 1–2 weeks

### 1.1 Diagnose Reward Function Issues

- [x] **Reward magnitude analysis**: Identified flat negative penalty as root cause.
- [x] **Switch to positive potential-based shaping**: Implemented `baseline + shaping` formula.
- [x] **Store `prev_dist` in env state** — tracked in `_prev_dist` and `_prev_dist_to_basket`.
- [x] **Increase `MAX_EPISODE_STEPS`** — now 200/300/400 per phase (was 100).
- [ ] **Verify observation normalization**: Confirm `VecNormalize` stats stabilize during training.

### 1.2 Fix Environment Mechanics

- [x] **Richer observations**: 8D → 20D (added relative_pos, joint_pos, joint_vel).
- [x] **Per-component reward logging**: All reward terms logged to info dict.
- [x] **Proper close()**: Cleans up robot and renderer resources.
- [ ] **Action scaling verification**: Confirm arm can traverse workspace in 200 steps.
- [ ] **Grasp weld threshold verification**: Test with actual EE geometry.

### 1.3 Hyperparameter Tuning

- [x] **Reduce `batch_size`** to 256.
- [x] **`ent_coef="auto"`** verified.
- [x] **Add EvalCallback** — runs every ~50K steps, saves best model.
- [x] **Increase training timesteps**: 2M for REACH, 2M for GRASP, 3M for PLACE.
- [ ] **Monitor training curves** and adjust if needed.

### 1.4 Training Diagnostics

- [x] **Log per-episode metrics** to TensorBoard via RewardInfoCallback.
- [ ] **Create `scripts/analyze_training.py`**: Parse TensorBoard logs.
- [ ] **Visualize policy rollouts**: Record videos of trained model.

### 1.5 Consolidate Training Scripts

- [x] **Clean up script proliferation**: 11 duplicate scripts archived to `archive/scripts/`.
- [x] **Make `configs/sac_config.yaml` actually used**: train.py now loads from YAML.

---

## Phase 2: Improve Robustness & Generalization

> **Goal**: Domain randomization, better grasping, richer observations.
> **Timeline**: 2–3 weeks (after Phase 1 success)

### 2.1 Domain Randomization (MuJoCo)

- [ ] **Object spawn randomization**: Already done ✅ (random x, y on table).
- [ ] **Object size randomization**: Vary cube size from 0.015 to 0.03m.
- [ ] **Object mass randomization**: Vary mass from 0.05 to 0.2 kg.
- [ ] **Friction randomization**: Vary `friction` attribute ±30%.
- [ ] **Lighting randomization**: Vary ambient/directional light (for vision pipeline later).
- [ ] **Actuator noise**: Add Gaussian noise to control signals to simulate motor imprecision.
- [ ] **Observation noise**: Add small Gaussian noise to observations to improve robustness.

### 2.2 Richer Observations

- [x] **Add joint positions to observation**: ✅ Done in v2 env rewrite (20D obs).
- [x] **Add joint velocities**: ✅ Done.
- [x] **Add relative position** (object - EE): ✅ Done.
- [ ] **Add distance to basket** as explicit feature for Phase 2.
- [ ] **Goal-conditioned observations**: Pass the target position as part of observation for generalization.

### 2.3 Improved Grasping

- [ ] **Replace magnetic weld with contact-based grasping**: The current weld constraint is unrealistic. Implement contact force thresholding:
  - Check `mujoco.mj_contactForce()` for contacts between gripper fingers and object.
  - Grasp is successful when both fingers have sufficient normal force.
- [ ] **Add force/torque sensing**: Use MuJoCo `sensordata` for 6-axis F/T at the wrist.
- [ ] **Gripper compliance**: Add PD control for gripper fingers with compliance (spring-damper model).

### 2.4 Better Reward Engineering

- [ ] **Hindsight Experience Replay (HER)**: Integrate SB3's HER wrapper for sparse reward settings.
- [ ] **Reward decomposition logging**: Log each reward component separately (distance, reach bonus, grasp bonus, etc.) to diagnose which phases are failing.
- [ ] **Curriculum auto-progression**: Automatically advance from REACH → GRASP → PLACE based on success rate thresholds (e.g., advance when success > 70% over 100 episodes).

---

## Phase 3: Vision-Based Policy

> **Goal**: Train policies from camera observations instead of privileged state.
> **Timeline**: 3–4 weeks

### 3.1 Image Observation Integration

- [ ] **Enable `use_vision=True`** path in `PickAndPlaceEnv`.
- [ ] **Define image observation space**: Use 64×64 RGB (or grayscale) stacked frames.
- [ ] **CNN feature extractor**: Use SB3's `CnnPolicy` or a custom `NatureCNN` / `ResNet` encoder.
- [ ] **Frame stacking**: Stack 3–4 frames for temporal information (velocity estimation).

### 3.2 Depth Camera

- [ ] **Implement depth rendering** in `sensors/camera.py` using MuJoCo's depth buffer.
- [ ] **RGB-D observation**: Concatenate RGB and depth as 4-channel input.
- [ ] **Point cloud processing** (optional): Convert depth to 3D points for PointNet-style policies.

### 3.3 State-Vision Distillation

- [ ] **Train teacher policy** using privileged state (Phase 2 result).
- [ ] **Train student policy** on images, supervised by teacher actions (DAgger-style).
- [ ] **Compare end-to-end RL vs. distillation** performance.

### 3.4 Camera Calibration

- [ ] **Fix `pixel_to_world` mapping**: The current implementation uses a simple pinhole model. Validate against ground truth MuJoCo positions.
- [ ] **Intrinsic/extrinsic calibration**: Compute proper camera matrices from MuJoCo camera parameters.
- [ ] **Multi-view setup**: Add side camera for depth disambiguation.

---

## Phase 4: Isaac Lab / Isaac Sim Migration

> **Goal**: Move from MuJoCo to Isaac Lab for GPU-accelerated training with 1000s of parallel envs.
> **Timeline**: 3–4 weeks

### 4.1 Fix Isaac Lab Environment

The `isaac_env/` directory has scaffolding that needs significant work:

- [ ] **Fix MDP function signatures**: `mdp.py` references `env._prev_dist` which isn't set anywhere. Add proper state tracking.
- [ ] **Implement missing MDP functions**: `joint_pos_delta_action`, `gripper_action`, `ee_position`, `object_position`, `gripper_state`, `time_out`, `reset_joints_by_offset`, `reset_root_state_uniform` — these are referenced in `env_cfg.py` but not all are implemented in `mdp.py`.
- [ ] **Switch robot model**: `env_cfg.py` uses `UR10e_ROBOTIQ_2F_85_CFG` (from `isaaclab_assets`). Decide if you want to keep UR10e or port the KUKA iiwa from MuJoCo. UR10e is a reasonable choice since Isaac Lab has native support.
- [ ] **Test scene loading**: Verify the scene builds and renders in Isaac Lab without errors.
- [ ] **Validate physics**: Compare object falling, grasping, and arm dynamics between MuJoCo and Isaac Lab.

### 4.2 Port Training to Isaac Lab

- [ ] **Register custom environment** with Isaac Lab's gym registry.
- [ ] **Use RSL-RL or RL Games** instead of SB3 for GPU-native training.
- [ ] **Scale to 128+ parallel environments** — target 10,000+ FPS.
- [ ] **Warm-start from MuJoCo policy**: Load SB3 weights and fine-tune in Isaac Lab.

### 4.3 Isaac Sim Rendering

- [ ] **RTX rendering**: Enable photorealistic rendering for vision policies.
- [ ] **Synthetic data generation**: Use Isaac Sim's replicator for large-scale domain-randomized image datasets.
- [ ] **USD scene export**: Convert MuJoCo XML → USD for Isaac Sim compatibility.

### 4.4 Clean Up Isaac Repos

- [ ] **Remove or gitignore `IsaacGymEnvs/` and `IsaacLab/`**: These are full upstream repo clones (with their own `.git` directories) taking up massive space. Use `pip install` or git submodules instead.
- [ ] **Clean `isaacsim-venv-3.11/` and `isaaclab-venv/`**: These venvs are adding ~34K files to the workspace. Consider using a shared venv or docker container.

---

## Phase 5: Advanced RL Algorithms

> **Goal**: Move beyond vanilla SAC. Try algorithms better suited for manipulation.
> **Timeline**: 2–3 weeks (can run in parallel with Phase 4)

### 5.1 Algorithm Exploration

- [ ] **PPO with clipping**: Often more stable than SAC for manipulation with shaped rewards.
- [ ] **TD3**: Twin delayed DDPG — compare with SAC for continuous control.
- [ ] **DDPG + HER**: Specifically designed for goal-conditioned sparse reward tasks.
- [ ] **DrQ-v2**: Data-regularized Q-learning for image-based RL.
- [ ] **Dreamer v3**: Model-based RL for sample-efficient learning.

### 5.2 Hierarchical RL

- [ ] **Options framework**: Define sub-policies for REACH, GRASP, LIFT, TRANSPORT, PLACE.
- [ ] **Goal-conditioned HRL**: High-level policy selects goals, low-level executes.
- [ ] **HIRO / HAC**: Hierarchical actor-critic for multi-step manipulation.

### 5.3 Imitation Learning Hybrid

- [ ] **Generate demonstrations**: Use inverse kinematics to create expert trajectories.
- [ ] **Behavior cloning pre-training**: Pre-train policy on demos, then fine-tune with RL.
- [ ] **GAIL / DAgger**: Adversarial imitation or interactive imitation from demonstrations.

---

## Phase 6: Infrastructure & Code Quality

> **Goal**: Production-quality codebase, CI/CD, reproducibility.
> **Timeline**: Ongoing

### 6.1 Testing

- [ ] **Unit tests for `KukaRobot`**: Test FK, action application, weld grasp/release.
- [ ] **Unit tests for `PickAndPlaceEnv`**: Test reset, step, reward computation, termination.
- [ ] **Integration tests**: Full training loop for 1000 steps, verify no crashes.
- [ ] **Regression tests**: Save a known-good model checkpoint and verify eval score doesn't degrade.

### 6.2 Code Cleanup

- [x] **Remove duplicate scripts**: 11 scripts archived to `archive/scripts/`.
- [x] **Wire up `configs/sac_config.yaml`**: train.py now loads from config.
- [ ] **Add `__init__.py` files** to all packages with proper exports.
- [ ] **Type hints everywhere**: Add comprehensive type annotations.
- [ ] **Docstring coverage**: Ensure all public functions have docstrings.

### 6.3 Logging & Experiment Tracking

- [ ] **Set up Weights & Biases**: The `wandb` dependency is in `requirements.txt` but never used. Integrate it.
- [ ] **Experiment configs**: Save full config (hyperparams + env settings + git hash) with each run.
- [ ] **Model registry**: Track model versions with metadata (phase, timesteps, success rate).

### 6.4 CI/CD

- [ ] **GitHub Actions**: Run smoke test + linting on every push.
- [ ] **Pre-commit hooks**: `black`, `isort`, `flake8`, `mypy`.
- [ ] **Docker container**: Reproducible training environment with pinned dependencies.

### 6.5 Documentation

- [ ] **Update `README.md`**: Currently 6 lines. Add project overview, installation, usage, architecture diagram.
- [ ] **API documentation**: Generate docs from docstrings (Sphinx or MkDocs).
- [ ] **Training guide**: How to reproduce training from scratch.
- [ ] **Architecture diagram**: Mermaid or draw.io showing env ↔ robot ↔ sim ↔ policy data flow.

---

## Phase 7: Sim-to-Real Transfer

> **Goal**: Deploy trained policy on a physical 6-DOF arm.
> **Timeline**: 4–6 weeks (after Phases 2–4)

### 7.1 Hardware Setup

- [ ] **Select physical robot**: KUKA iiwa 7 R800, or UR5e/UR10e (depends on availability).
- [ ] **Gripper selection**: Robotiq 2F-85 (matches Isaac Lab config) or custom.
- [ ] **Camera setup**: Intel RealSense D435 or ZED 2 for RGB-D.
- [ ] **Workspace setup**: Table + basket matching simulation dimensions.

### 7.2 ROS 2 Bridge

- [ ] **Create ROS 2 package**: Wrap the policy inference in a ROS 2 node.
- [ ] **Joint command interface**: Map RL actions to `JointTrajectoryController` commands.
- [ ] **Sensor subscribers**: Camera image + joint state → observation vector.
- [ ] **Safety layer**: Joint velocity/torque limits, workspace bounds, emergency stop.

### 7.3 Sim-to-Real Techniques

- [ ] **System identification**: Measure real robot dynamics, tune MuJoCo model.
- [ ] **Automatic Domain Randomization (ADR)**: Progressively increase randomization until transfer works.
- [ ] **Real-world fine-tuning**: Online RL on the real robot with safety constraints.
- [ ] **Residual policy learning**: Train a residual correction on top of sim policy.

---

## Priority Order

```
Phase 1 (Fix Training) ──────────────────────► MUST DO FIRST
    │
    ├── Phase 2 (Robustness) ────────────────► After REACH works
    │       │
    │       ├── Phase 3 (Vision) ────────────► After state-based works
    │       │
    │       └── Phase 5 (Algorithms) ────────► Can parallelize
    │
    ├── Phase 4 (Isaac Lab) ─────────────────► After MuJoCo pipeline proven
    │
    ├── Phase 6 (Infrastructure) ────────────► Ongoing, start now
    │
    └── Phase 7 (Sim-to-Real) ──────────────► After everything else
```

---

## Quick Wins (Do This Week)

1. ~~**Fix reward function**~~ ✅ Done — positive potential-based shaping
2. ~~**Add `prev_dist` tracking**~~ ✅ Done
3. ~~**Increase `MAX_EPISODE_STEPS`**~~ ✅ Done (200/300/400 per phase)
4. ~~**Reduce `batch_size` to 256**~~ ✅ Done
5. ~~**Add `EvalCallback`**~~ ✅ Done
6. **Retrain Phase 0 (REACH)** — 🔄 In progress (2M timesteps)
7. ~~**Clean up duplicate training scripts**~~ ✅ Done (11 scripts archived)
8. **Update `README.md`** — TODO
