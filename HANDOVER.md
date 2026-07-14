# Project handover

Last updated: 2026-07-14

## Working baseline

- Environment: Isaac Lab `ManagerBasedRLEnv`, UR10e + Robotiq 2F-85.
- Policy observation: 34D normalized state.
- Policy action: six bounded relative arm commands plus one binary gripper command.
- Best Phase 0 checkpoint:
  `models/isaac/phase_0/model_pregrasp_coupled_v2.pt`.
- Phase 0 strict evaluation: 256/256 reach at 5 cm, 7 mm median minimum
  distance, 15.4 degree median minimum wrist error, 256/256 grasp-ready poses.
- Existing Phase 1 warm-start:
  `models/isaac/phase_1/model_grasp_v1.pt`.
- Recommended Phase 1 path: `scripts/evaluate_isaac.py --hybrid_phase1`.
- Hybrid strict lift with corrective regrasp: 41/64 on seed 19595 and
  37/64 on seed 42 (78/128, 60.9%).

Phase 0 should not be retrained. Its shuffled-target control degrades sharply,
which confirms that it actually uses the target coordinates rather than replaying
a fixed trajectory.

## Key diagnosis

The Robotiq asset is intentionally driven through `finger_joint`; its dependent
joints move through the asset's mimic/passive linkage. All six linkage joints
were observed moving, and the gripper can physically lift an object. Driving all
six joints directly is not the fix.

The original 4 cm cube has a narrow contact band. An 8 cm descent is too shallow
for repeatable side contact, while deeper descents collide with the table. A
4 x 4 x 10 cm starter block removes that ambiguity and proved the grasp pipeline.

Waiting for the gripper to close to 0.70 rad before lifting reduced success from
34% to 3%. A 0.45 rad transition allows closing to continue during retract. A
2 cm Cartesian retract request per control tick improved strict success to
44-47% across two seeds. One bounded retry uses the measured gripper-midpoint
miss to correct the next XY approach, and a calibrated +7 mm Y target correction
raises the final two-seed result to 60.9%.

The next limiting evidence is simulator contact integrity. On seed 42, the
current preset produced 19 gripper-integrity and 5 arm-integrity terminations
(terms can overlap), while still recording 37/64 successful lifts. Higher
damping increased resets; lower effort reduced resets but also reduced success.
Keep the current actuator defaults until this is handled as a focused stability
change rather than hidden inside policy training.

## Architecture decision

Do not train a raw-video-to-motor policy. Use:

```text
camera -> object pose/confidence -> target-position observation
robot joint state -------------> reach policy -> pose servo -> gripper sequence
```

Simulation currently supplies the target pose directly. The next integration
replaces that one source with a camera adapter. Joint feedback comes from the
robot controller and does not imply buying extra sensing hardware for this stage.

## Run notes

- Use `.venv\Scripts\python.exe`; system Python is not the Isaac environment.
- Each Isaac process needs `OMNI_KIT_ACCEPT_EULA=YES`.
- On this Windows laptop use:
  `--kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"`.
- Headless runs are most reliable in a PTY.
- Two protected stale Python processes may remain visible but have not blocked
  current evaluation runs.

See `README.md` for copy-paste commands and `spikes/002-grasp-geometry/README.md`
for the geometry evidence.

## Immediate next work

1. Add a replaceable camera/target-provider interface and perturbation tests.
2. Add confidence and stale-target handling before any real camera connection.
3. Improve grasp success from 61% toward the 80% starter-object gate.
4. Test object dimensions from easy block toward the 4 cm cube.
5. Reserve cloud GPU time for a short residual correction policy only if needed.
6. Begin Phase 2 after grasp success is consistently high.
