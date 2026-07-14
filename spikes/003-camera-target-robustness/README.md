# Camera-target robustness probe

## Question

How accurate must the future camera-to-XYZ adapter be for the existing Phase 0
policy, without retraining the policy or rendering camera images during PPO?

## Method

`scripts/evaluate_isaac.py --target_obs_mode noisy` perturbs every object-derived
policy channel while rewards and success metrics continue to use simulator truth.
Each condition used the Phase 0 best checkpoint, seed 42, and 64 environments.

## Results

| Target perturbation | Reach at 5 cm | True median minimum error |
|---|---:|---:|
| 5 mm Gaussian/frame + (+5, -5, 0) mm bias | 64/64 (100%) | 9 mm |
| 15 mm Gaussian/frame + (+10, -10, +10) mm bias | 64/64 (100%) | 6 mm |
| static (+30, +30, 0) mm bias | 64/64 (100%) | 28 mm |
| static (+50, +50, 0) mm bias | 19/64 (30%) | 53 mm |

Zero-mean frame jitter is comparatively benign because closed-loop control
effectively averages it. Persistent calibration bias is the limiting factor.

## Decision

- Use a filtered target-estimate adapter instead of training from pixels.
- Require persistent tabletop XY error below 3 cm for reaching.
- Target roughly 1 cm or better for grasping a 4 cm-wide object.
- Reject stale, low-confidence, out-of-workspace, and implausibly jumping targets.
- Do not spend cloud GPU time on vision-policy training at this stage.

The hardware-independent implementation is `isaac_env/target_provider.py`; its
validation and hold/expiry behavior is covered by `tests/smoke_test.py`.
