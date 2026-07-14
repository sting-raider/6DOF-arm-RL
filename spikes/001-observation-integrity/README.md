# Observation-integrity spike

## Question

Given the current Isaac Lab environment, when it is stepped with controlled
actions, do the 30 raw policy observations stay finite and within physically
plausible ranges?

This isolates simulator/action instability from the running-normalizer code.
The trained Phase 0 checkpoint contains impossible statistics (for example,
object-position standard deviations of 32--88 m in a sub-meter workspace).

## Commands

```powershell
python inspect_observations.py --headless --num_envs 256 --steps 300 --action_mode zero
python inspect_observations.py --headless --num_envs 256 --steps 300 --action_mode random
python inspect_observations.py --headless --num_envs 256 --steps 300 --action_mode random --action_clip 1
```

## Verdict: VALIDATED

### Evidence

- With zero actions, 256 environments over 300 steps had no implausible
  observation; active joint position and velocity stayed exactly at zero.
- With unconstrained Gaussian actions, the first bad value appeared at step 52;
  joint velocity reached 3,069 rad/s, gripper position reached 3,179 rad, and an
  object moved nearly 10 m from the workspace.
- Clipping raw actions alone did not solve the failure. Explicitly commanding
  all six Robotiq mimic/passive joints was the largest contributor.
- The final stack (drive joint only, bounded relative targets, raw-action clip,
  open gripper for Phase 0, and invalid-state reset) completed a 512-environment,
  1,000-step stress run without returning an implausible observation. It had 6
  safety resets beyond 3,072 expected timeouts: about 0.0012% of env-steps.
- After moving that stack into the production environment, a separate
  512-environment, 300-step run passed with no implausible observations and no
  extra resets beyond the 1,024 expected timeouts.

### Constraints and surprises

- The spike does not prove PPO task success; it proves that the environment
  boundary no longer feeds catastrophic outliers to PPO normalization.

### Recommendation

- Retrain Phase 0 from a fresh normalizer, then require 6 cm/5 cm success and a
  shuffled-target ablation before warm-starting Phase 1.
