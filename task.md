# 6-DOF Arm Pick-and-Place — Task Tracker

> **Status**: Phase 2 retrained and evaluated. Phase 2 model at 25% grasp success (with DR).
> **Next**: Improve Phase 2 place success (0%) with longer training or reward tuning.

## Phase 1: Fix Training Pipeline ✅ COMPLETE

All fixes done, all enhancements committed.

## Phase 2: Improve Robustness & Generalization ✅ COMPLETE

### 2.0 Fix Evaluation Issues ✅
- [x] VecNormalize loading fixed in eval script
- [x] Success flags persist across episode boundaries

### 2.1 Retrain Phase 2 ✅
- [x] Retrained from scratch — warm-start Phase 1 → 5M new Phase 2 steps
- [x] Training time: 42 min @ 1,997 FPS on RTX 3060
- [x] ep_rew_mean: **575** final (147 at start — 3.9× improvement)
- [x] Entropy: 0.000193 — well converged

### 2.2 Domain Randomization ✅
- [x] Object size randomization (0.015–0.030m)
- [x] Object mass randomization (0.05–0.20 kg)
- [x] Friction randomization (0.5×–2.0×)
- [x] Observation noise (Gaussian σ=0.002)

### 2.3 Evaluation Results ✅

| Metric | Phase 0 (REACH) | Phase 1 (GRASP) | Phase 2 (PLACE) |
|--------|:-:|:-:|:-:|
| Mean Reward | 112 ± 43 | 392 ± 536 | **666 ± 934** |
| Episode Length | 183 ± 53 | 271 ± 86 | **361 ± 117** |
| Reach Success | 0% | 20% | **25%** |
| Grasp Success | 0% | 20% | **25%** |
| Place Success | 0% | 0% | 0% |
| DR Active | No | No | Yes |

**Key findings:**
- Phase 2 model reaches+grasps at **25%** (beats Phase 1's 20%) — even with DR noise
- Reward pattern shows clear bimodal distribution: ~200 (failed grasp) vs 1800-2700 (successful grasp)
- **0% place success** — the basket transport is the remaining hard problem
- The high training ep_rew_mean (575) reflects the 25% grasp rate across 16 parallel envs

### 2.4 Why Place Is Still 0%

The Phase 2 reward fix (basket-dist baseline swap when grasped) is correct, but:
1. The policy needs more training to consistently associate "grasped → go to basket"
2. The basket placement requires precise alignment that's harder with DR noise
3. SAC may be plateauing — exploring basket transport needs more diverse experience

**Solutions available:**
- Train longer (7-10M Phase 2 steps) — most straightforward
- Increase place shaping weight (currently 15×, could go to 25×)
- Add basket-approach curriculum sub-phase (Phase 2a: transport, Phase 2b: place)
- Reduce DR during early Phase 2 training, increase gradually

## Phase 3+: Vision, Isaac Lab, Advanced Algorithms — PENDING

See PLAN.md for the full roadmap.
