# 6-DOF Arm Pick-and-Place

Laptop-first manipulation research with Isaac Lab, a UR10e arm, a Robotiq
2F-85 gripper, and RSL-RL PPO.

The current working system combines a learned, target-conditioned reaching
policy with a deterministic 6-DOF pose controller for descend, close, and
retract. This hybrid is cheaper and easier to debug than asking PPO to discover
the entire contact sequence from scratch.

## Current status

| Capability | Result | Checkpoint / path |
|---|---:|---|
| Phase 0 reach, strict 256-episode test | 100% at 5/6/8 cm | `models/isaac/phase_0/model_pregrasp_coupled_v2.pt` |
| Phase 0 median closest distance | 7 mm | same checkpoint |
| Phase 0 shuffled-target control | degrades sharply | confirms target conditioning |
| Hybrid Phase 1 lift, two 64-episode seeds | 78/128 (60.9%) | `models/isaac/phase_1/model_grasp_v1.pt` |
| Four-arm fixed-layout smoke test | 4/4 reach, 4/4 grasp-cycle entry, 2/4 strict lift | `--demo_layout` |
| Phase 0 with static 3 cm X + 3 cm Y target bias | 64/64 at 5 cm | `--target_obs_mode noisy` |
| Phase 0 with static 5 cm X + 5 cm Y target bias | 19/64 at 5 cm | calibration failure control |
| Phase 2 place | not started | depends on a reliable grasp |

The Phase 1 result includes one bounded, midpoint-corrected regrasp and uses a
4 x 4 x 10 cm upright starter object. The original
4 cm cube has a very narrow vertical contact band and is not yet reliable.

## What the policy sees

The current checkpoint does **not** consume raw video. In simulation, Isaac
provides the object's 3D centroid directly. The 34D policy observation is:

| Component | Dim | Description |
|---|---:|---|
| arm joint positions | 6 | six active UR10e joints |
| arm joint velocities | 6 | six active UR10e joints |
| wrist position | 3 | local Cartesian position |
| wrist orientation | 4 | quaternion |
| gripper state | 1 | Robotiq drive-joint state |
| object position | 3 | target centroid; simulator truth for now |
| wrist-to-target vector | 3 | relative target position |
| target distance | 1 | scalar distance |
| previous action | 7 | six arm commands plus gripper command |
| **Total** | **34** | |

The real-world design replaces only the simulator's object-position channel:

```text
camera frame -> object detector / pose estimate -> target XYZ
                                                   |
robot joint feedback ------------------------------+-> reach policy
                                                        |
                                                        v
                                      pose servo -> close -> retract
```

Robot joint feedback is still required for closed-loop control, but it normally
comes from the robot itself and is not a separate camera or training expense.
A camera-only raw-video-to-motor policy would cost more data, be harder to debug,
and transfer less reliably.

`isaac_env/target_provider.py` now defines the camera-independent estimate
contract and rejects low-confidence, stale, out-of-workspace, non-finite, or
implausibly jumping detections. The policy remained perfect at the 5 cm reach
criterion with a static 3 cm bias on both X and Y, but fell to 30% with 5 cm bias
on both axes. The grasping target is therefore about 1 cm calibrated accuracy,
with 3 cm treated as the outer reach-only bound.

## Run the validated laptop demo

These commands are for the existing Windows workspace and virtual environment.
The renderer flags select the stable D3D path on this laptop.

Headless 64-arm benchmark:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 1 `
  --model models\isaac\phase_1\model_grasp_v1.pt `
  --num_envs 64 --episodes 64 --seed 42 `
  --hybrid_phase1 --headless `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

Visible four-arm demo with a different object position for each arm:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 1 `
  --model models\isaac\phase_1\model_grasp_v1.pt `
  --num_envs 4 --episodes 4 --seed 42 `
  --hybrid_phase1 --demo_layout --realtime `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

Phase 0 regression benchmark:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 0 `
  --model models\isaac\phase_0\model_pregrasp_coupled_v2.pt `
  --num_envs 256 --episodes 256 --headless `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

## Training

Phase 0 does not need to be retrained before continuing. If a later experiment
does need PPO, warm-start only the actor and observation normalizer so the critic
and optimizer start cleanly:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe scripts\train_isaac.py `
  --phase 1 --num_envs 512 --max_iterations 100 --headless `
  --warm_start models\isaac\phase_0\model_pregrasp_coupled_v2.pt `
  --output_model models\isaac\phase_1\model_grasp_v1.pt
```

Use 256-512 environments for laptop experiments. The four-hour cloud GPU should
be reserved for a final residual-policy or domain-randomization run after the
local controller and camera adapter pass their tests.

## Why this will not translate perfectly by itself

Isaac validates control logic and contact hypotheses; it cannot guarantee a
perfect real-world transfer. Remaining gaps include camera calibration and
occlusion, object-pose error, unknown friction and mass, gripper compliance,
latency, and robot-specific safety limits. The low-cost transfer plan is:

1. keep the proven reach policy;
2. add a replaceable camera-to-target adapter with noise/confidence handling;
3. test that adapter against synthetic perturbations locally;
4. improve the midpoint-corrected regrasp beyond its current 61% lift rate;
5. use a short randomized residual-policy run only if the controller plateaus;
6. start Phase 2 only after grasp success is consistently high.

## Tests

The fast tests do not launch Isaac Sim:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Physical behavior must also be verified with `scripts/evaluate_isaac.py`; syntax
and smoke tests alone cannot validate contact dynamics.

## License

MIT. See `LICENSE` if present in the distribution.
