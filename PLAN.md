# 6-DOF arm roadmap

Last updated: 2026-07-14

## Constraints and decisions

- Prefer local RTX 3060 laptop runs with 256-512 environments for experiments.
- Do not require purchased hardware for the current development milestones.
- Phase 0 is complete and will not be retrained unless a regression invalidates
  its 256/256 strict reach result.
- The eventual task input is a camera feed converted into an object pose. Robot
  joint feedback still closes the control loop.
- Do not train a raw-video-to-motor policy. Use a detector/pose estimator, the
  existing reach policy, and deterministic contact control.
- The four-hour cloud GPU is gated behind a demonstrated deterministic-control
  plateau. If used, train only a small residual correction policy.

## Milestone 0: target-conditioned reach - complete

- [x] Stable 34D observation space and normalization.
- [x] Bounded arm actions and stable Robotiq drive action.
- [x] 256/256 strict 5 cm reach evaluation.
- [x] Target-shuffle ablation demonstrating target conditioning.
- [x] Four-arm visual evaluation at distinct positions.

## Milestone 1A: contact stability - immediate priority

- [x] Verify the simulated Robotiq linkage and physical colliders.
- [x] Instrument individual timeout, object, arm, and gripper termination terms.
- [x] Reject damping and effort changes that trade safety for misleading lifts.
- [x] Attribute integrity resets to exact grasp stages and successful/failed
  episodes.
- [x] Isolate free closure, object contact, and table clearance in a
  physics-step contact probe.
- [x] Decouple arm pose correction from finger closure, reducing gripper
  integrity resets to 1/128 across the two established seeds.
- [ ] Prevent mechanically invalid finger overshoot without reducing strict lift.
- [ ] Reduce combined arm/gripper integrity terminations below 2% in the nominal
  starter-object benchmark.
- [ ] Add a repeatable contact-stability regression matrix covering at least
  three seeds.

## Milestone 1B: reusable hybrid controller

- [x] Separate learned reaching from deterministic descend/close/retract control.
- [x] Add full-pose damped-least-squares control.
- [x] Add slip detection and one bounded, midpoint-corrected regrasp.
- [x] Benchmark the contact-safe preset at 76/128 (59.4%) strict lifts across
  two seeds.
- [ ] Move the hybrid state machine out of `scripts/evaluate_isaac.py` into a
  reusable controller module.
- [ ] Make training evaluation, visual demos, and future deployment use the same
  controller implementation.
- [ ] Add unit tests for state transitions, retry limits, stale targets, and safe
  open/stop behavior.

## Milestone 1C: Phase 1 exit gates

Phase 1 is complete only when all of these pass:

- [ ] At least 80% strict lift success on the 4 x 4 x 10 cm starter object.
- [ ] At least 256 evaluated episodes across at least three seeds.
- [ ] Less than 2% arm/gripper integrity terminations.
- [ ] Hold the lifted object for at least two seconds without dropping or
  exceeding joint limits.
- [ ] At least 70% strict lift under bounded pose, friction, and mass
  perturbations.
- [ ] Pass the fixed four-arm layout with four different object positions.

## Milestone 1D: object curriculum

Start only after the nominal Phase 1 exit gates pass.

- [ ] Shorten the starter block in measured stages: 10 cm to 8 cm to 6 cm to
  the original 4 cm cube.
- [ ] Add width, aspect-ratio, yaw, mass, and friction variation one factor at a
  time.
- [ ] Add multiple simple shapes before claiming arbitrary-object handling.
- [ ] Record per-object success rates instead of reporting one aggregate score.

## Milestone 2: camera-target robustness

- [x] Define a target estimate with XYZ, timestamp, and confidence.
- [x] Feed estimates through the same policy channel used by simulator truth.
- [x] Reject stale, low-confidence, out-of-workspace, and jumping detections.
- [x] Test position noise and persistent calibration bias in simulation.
- [ ] Test update latency, dropped frames, and bursty detections.
- [ ] Verify that stale or invalid targets stop target-directed motion and keep
  the gripper open.
- [ ] Pass the Phase 0 reach benchmark with the simulated camera stream before
  connecting an RGB detector.

## Milestone 3: low-cost RGB detector

Start after contact stability and simulated camera-stream tests pass.

- [ ] Use a pretrained or classical tabletop detector before considering custom
  vision training.
- [ ] Convert image detections into robot-frame XYZ through camera/table
  calibration.
- [ ] Target approximately 1 cm persistent position accuracy for grasping;
  treat 3 cm as a reach-only outer bound.
- [ ] Test recorded or synthetic video for occlusion, lighting, confidence, and
  calibration drift.
- [ ] Keep detector replacement independent of the reach/grasp controller.

## Milestone 4: Phase 2 transport and placement

Do not start until every Phase 1 exit gate passes.

- [ ] Add a stable hold-to-transport transition.
- [ ] Move a grasped object to the basket without integrity resets or excessive
  slip.
- [ ] Validate basket collision geometry.
- [ ] Add controlled release and post-release verification.
- [ ] Evaluate full reach-grasp-transport-place success across multiple seeds.

## Milestone 5: optional residual policy and cloud gate

Cloud training is authorized only if deterministic control has plateaued after
Milestones 1A-1C and the remaining error is clearly measurable.

- [ ] Write a residual-policy hypothesis and local baseline before using cloud
  time.
- [ ] Freeze the validated Phase 0 policy.
- [ ] Train only bounded corrections around deterministic grasp control.
- [ ] Time-box the cloud run to the available four hours.
- [ ] Keep the residual only if it improves held-out success without increasing
  integrity resets.

## Sim-to-real claim gate

Simulation success does not guarantee perfect real-world transfer. Before making
that claim, measure camera calibration error, pose-estimation error, latency,
friction, mass, gripper compliance, robot joint limits, emergency-stop behavior,
and safe handling of lost detections. Physical validation remains deferred until
hardware is available without expanding the current project budget.
