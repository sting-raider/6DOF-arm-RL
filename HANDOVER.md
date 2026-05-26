# Handover: 6-DOF Arm RL → Antigravity

**Date:** May 26, 2026
**From:** Hermes (DeepSeek)
**To:** Antigravity
**Project:** `~/ic-6dof-arm/` — UR10e Pick-and-Place via Isaac Lab + PPO

---

## TL;DR

Training a 6-DOF UR10e arm with curriculum RL (REACH → GRASP → PLACE) in NVIDIA Isaac Lab. The core challenge was a buggy `EmpiricalNormalization` that poisons inference. The fix is in place but training needs to finish.

---

## Project State

### What works
- **Environment:** Isaac Lab scene — UR10e + Robotiq gripper, table, object, basket. 4096 parallel envs stable on RTX 3060 6GB.
- **Reward functions:** Phase-aware `mdp.reach_reward()` dispatches REACH/GRASP/PLACE automatically via `env.cfg.curriculum_phase`.
- **Eval script:** `scripts/evaluate_isaac.py` — uses raw Isaac Lab scene positions for reach/grasp/place success metrics. Includes 20-step normalizer warmup.
- **Code is pushed:** `sting-raider/6DOF-arm-RL` on GitHub.

### What needs to happen (in order)

```
Phase 0 (REACH):  python scripts/train_isaac.py --phase 0 --num_envs 4096 --headless --max_iterations 1000
Phase 1 (GRASP):  python scripts/train_isaac.py --phase 1 --num_envs 4096 --headless --max_iterations 1000 --checkpoint models/isaac/phase_0/model.pt
Phase 2 (PLACE):  python scripts/train_isaac.py --phase 2 --num_envs 4096 --headless --max_iterations 1000 --checkpoint models/isaac/phase_1/model.pt
Eval all:         python scripts/evaluate_isaac.py --phase 0/1/2 --model models/isaac/phase_X/model.pt --episodes 20 --num_envs 16
```

**Always use the isaacsim venv:** `source ~/ic-6dof-arm/isaacsim-venv-3.11/bin/activate`

---

## The Normalization Saga (critical context)

### Problem
RSL-RL's `EmpiricalNormalization` tracks running mean/std during training. For near-constant observations (e.g., gripper=0 in Phase 0), the variance collapses → std becomes NaN or 4 billion → saved normalizer stats corrupt → inference produces garbage actions (±57 radian joint commands).

### What we tried
1. **obs_normalization=False** → policy can't learn (reach 0.006 vs 0.69 with it ON). The MLP needs normalized inputs.
2. **Manual scaling** (EE/1.5, gripper×25) → also can't learn (reach 0.003).
3. **Sanitize normalizer before save** → crashed because PyTorch inference mode blocks inplace tensor ops.
4. **strict=False loading** → MLP weights trained with normalization but eval uses no normalization → garbage.

### Current approach (in scripts now)
- **Training:** `obs_normalization: True` (required for learning)
- **Eval:** `obs_normalization: True` + **20-step warmup** after load to adapt normalizer stats to eval env

### If it still doesn't work
- Train with normalization, save model
- Before eval, load model, run 50-100 warmup steps with the SAME number of envs
- OR: train and eval with identical `num_envs` so normalizer stats match
- OR: port to Stable-Baselines3 which has better normalization handling

---

## Key Files

| File | Purpose |
|---|---|
| `scripts/train_isaac.py` | PPO training entry point. `--phase`, `--num_envs`, `--checkpoint` |
| `scripts/evaluate_isaac.py` | Eval with real scene positions, warmup, reach/grasp/place metrics |
| `isaac_env/env_cfg.py` | Scene, robot, rewards, observations config |
| `isaac_env/mdp.py` | Reward functions, observations, terminations, domain rand |
| `models/isaac/phase_X/` | Saved models (currently stale from failed runs) |
| `logs/isaac/phase_X_v7.log` | Latest training logs |

---

## Hardware Constraints

- **GPU:** RTX 3060 Laptop 6GB
- **Max envs:** 4096 with normalization (6144 OOMs during PhysX init)
- **Phase 2** (with basket) needs 4096 or fewer
- **FPS:** ~46K at 4096 envs, ~55K at 6144

---

## Run Command (copy-paste ready)

```bash
cd ~/ic-6dof-arm && source isaacsim-venv-3.11/bin/activate

# Phase 0
python scripts/train_isaac.py --phase 0 --num_envs 4096 --headless --max_iterations 1000

# Phase 1 (warm-start)
python scripts/train_isaac.py --phase 1 --num_envs 4096 --headless --max_iterations 1000 \
  --checkpoint models/isaac/phase_0/model.pt

# Phase 2 (warm-start)
python scripts/train_isaac.py --phase 2 --num_envs 4096 --headless --max_iterations 1000 \
  --checkpoint models/isaac/phase_1/model.pt

# Eval
python scripts/evaluate_isaac.py --phase 0 --model models/isaac/phase_0/model.pt --episodes 20
python scripts/evaluate_isaac.py --phase 1 --model models/isaac/phase_1/model.pt --episodes 20
python scripts/evaluate_isaac.py --phase 2 --model models/isaac/phase_2/model.pt --episodes 20
```

---

## Observation Space (7D)

| Dim | Component | Raw Range | Source |
|-----|-----------|-----------|--------|
| 0-2 | EE position (x,y,z) | [-0.5, 1.5]m | `wrist_3_link` body_pos |
| 3-5 | Object position (x,y,z) | [0.2-0.5, ±0.15, 0.85]m | `RigidObject.root_pos_w` |
| 6 | Gripper state | [0, 0.04] | `finger_joint` position |

## Action Space (7D)

| Dim | Component | Scale |
|-----|-----------|-------|
| 0-5 | 6 arm joint deltas | ±0.05 rad |
| 6 | Gripper open/close | binary |

---

## Git

```
Remote:  https://github.com/sting-raider/6DOF-arm-RL.git
Branch:  main
Last commit: 96e81b5 "fix: re-enable obs_normalization + eval warmup adapts normalizer"
```

Everything is pushed. No uncommitted changes.
