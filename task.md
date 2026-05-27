# 6-DOF Arm Pick-and-Place — Task Tracker

> **Status**: PPO Training Stabilization milestones achieved. Sanity check run of 150 iterations succeeded (Value loss: 0.0130).
> **Current**: Retraining Phase 0 (REACH) in NVIDIA Isaac Lab at 4096 environments with 29D state observability and running normalizers.

## Phase 1: Cleanup & Transition ✅ COMPLETE

- [x] Project cleaned up, MuJoCo training artifacts archived
- [x] Isaac Lab environment verified as primary simulation backend
- [x] Resolved NGC API key hardcoding in `setup_cloud.sh`
- [x] Configured 29D complete state observability in `env_cfg.py`

## Phase 2: Isaac Lab Retraining — IN PROGRESS

All curriculum phases retrained from scratch on Isaac Lab with stable PPO configs.

### Phase 0 — Reach (TARGET: $\ge 90\%$ reach success)
- [x] Succeeded in PPO normalizer sanity check (512 envs, 150 iterations; value loss: 0.0130, reward: 0.3243)
- [x] Resolved Robotiq multi-joint articulation action mapping mismatch (6 PhysX joints)
- [/] Train full-scale REACH policy at 4096 envs for 1500 iterations (v14 active, task-1794)
- [ ] Evaluate success rate via coordinate-aligned tracking script
- [ ] **Status: IN PROGRESS (v14 active)**

### Phase 1 — Grasp (TARGET: $\ge 80\%$ grasp success)
- [ ] Train grasp policy, warm-started from Phase 0 best checkpoint
- [ ] Target: 1500 iterations at 4096 envs
- [ ] **Status: QUEUED**

### Phase 2 — Place (TARGET: $\ge 80\%$ place success)
- [ ] Train place policy, warm-started from Phase 1 best checkpoint
- [ ] Target: 1500 iterations at 4096 envs
- [ ] **Status: QUEUED**

## Key Metrics Dashboard

| Training Target | Isaac Lab Goal | Sanity Check (PPO v14) |
|---|---|---|
| Parallel Envs | 4,096 | 512 |
| Mean Value Loss | < 1.0 | **0.0130** (Super Stable) |
| Reach Reward | Dense Potential | **0.3243** (Climbing) |
| Physics Jitter | Resolved | Mimic joints filtered out |
