# 6-DOF Arm Pick-and-Place — Implementation Task Tracker

## 1. Environment Config & Package Structure ✅ COMPLETE
- [x] Implement `HERMES_DISABLE_DR` in `PickPlaceEnvCfg.__post_init__` inside `env_cfg.py`
- [x] Verify `isaac_env/__init__.py` imports and package structure

## 2. MDP Definitions & Gripper Semantics ✅ COMPLETE
- [x] Update `gripper_state()` in `mdp.py` to dynamically index `"finger_joint"`
- [x] Update `gripper_state_scaled()` in `mdp.py` to scale `"finger_joint"` to `[0, 1]`
- [x] Update `reach_reward()` in `mdp.py` to use `"finger_joint"` closedness
- [x] Delete unused `joint_pos_delta_action()` and `gripper_action()` in `mdp.py`
- [x] Delete unused `_debug_printed` latch variables in `mdp.py`
- [x] Align hardcoded basket target center in `mdp.py` to `[0.6, 0.0, 0.85]`

## 3. Training Script Cleanup ✅ COMPLETE
- [x] Clean up unused imports (`gymnasium`, `datetime`) in `train_isaac.py`
- [x] Update misleading "zero-initialized critic" comments in `train_isaac.py`

## 4. Evaluators Synchronization ✅ COMPLETE
- [x] Sync header comments and `obs_normalization: True` settings in `evaluate_isaac.py`
- [x] Remove 20-step normalizer warmup block in `evaluate_isaac.py`
- [x] Wrap rollout inference loop with `with torch.no_grad():` in `evaluate_isaac.py`
- [x] Align basket center coordinate in `evaluate_isaac.py` to `[0.6, 0.0, 0.85]`
- [x] Align grasp success detection to `"finger_joint"` position (>0.40 rad) in `evaluate_isaac.py`
- [x] Point `PHASES` models to `models/isaac/phase_{phase}/model.pt` in `evaluate_all.py`
- [x] Point called entrypoint to `scripts/evaluate_isaac.py` in `evaluate_all.py`

## 5. Verification & Testing ⏳ QUEUED
- [ ] Run evaluation of Phase 0 with `HERMES_DISABLE_DR=1`
- [ ] Run batch evaluation check with `evaluate_all.py --dry-run`
- [ ] Push clean changes to GitHub repository
