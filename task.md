# 6-DOF Arm Pick-and-Place — Task Tracker

> **Status**: Phase 1 cleanup complete. MuJoCo archived. Isaac Lab is primary sim.
> **Next**: Phase 0 Isaac Lab retrain from scratch at 64k FPS.

## Phase 1: Cleanup & Transition ✅ COMPLETE

- [x] Project cleaned up, MuJoCo training artifacts archived
- [x] Isaac Lab environment verified as primary simulation backend
- [x] Previous MuJoCo results saved for reference (25% grasp, 0% place)
- [x] Ready for fresh Isaac Lab training pipeline

## Phase 2: Isaac Lab Retraining — IN PROGRESS

All phases trained from scratch on Isaac Lab at target 64k FPS.

### Phase 0 — Reach (TARGET: ~90% reach success)
- [ ] Train reach policy in Isaac Lab (no object interaction)
- [ ] Target: ~1000–2000 episodes, evaluate
- [ ] **Status: PENDING — about to start**

### Phase 1 — Grasp (TARGET: ~80% grasp success)
- [ ] Train grasp policy, warm-started from Phase 0
- [ ] Target: ~3000–5000 episodes, evaluate
- [ ] **Status: PENDING**

### Phase 2 — Place (TARGET: ~70% place success)
- [ ] Train place policy, warm-started from Phase 1
- [ ] Target: ~5000+ episodes, evaluate
- [ ] **Status: PENDING**

## Key Metrics Dashboard

| Training Target | Isaac Lab Goal | Previous (MuJoCo) |
|---|---|---|
| FPS | 64,000 | 2,000 |
| Reach Success | ~90% | 0% |
| Grasp Success | ~80% | 25% |
| Place Success | ~70% | 0% |
