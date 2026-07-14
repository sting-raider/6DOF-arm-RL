# Active task tracker

## Completed

- [x] Stabilize the observation normalizer by excluding passive/mimic velocities.
- [x] Add wrist quaternion and distance for a 34D policy input.
- [x] Bound relative joint targets and gripper slew rate.
- [x] Train and validate target-conditioned Phase 0 reaching.
- [x] Warm-start the current Phase 1 checkpoint.
- [x] Verify Robotiq mimic linkage and physical contact.
- [x] Build and benchmark the hybrid grasp controller.
- [x] Add one bounded, midpoint-corrected regrasp (78/128 strict lifts).
- [x] Add a four-object visual layout.
- [x] Add target-shuffle and grasp-geometry evaluation controls.
- [x] Document the verified Windows laptop commands.

## In progress

- [x] Camera-target provider with confidence/staleness behavior.
- [x] Camera-like target noise and bias evaluation.
- [x] Reactive slip detection and regrasp.
- [ ] Attribute integrity resets to grasp stages and reduce them below 2%.
- [ ] Extract the hybrid state machine into a reusable controller module.
- [ ] Starter-object gate: at least 80% strict lift with a two-second hold.
- [ ] Latency/dropout target-sequence evaluation after contact stabilization.

## Deferred

- [ ] Shrink the grasp curriculum toward the original 4 cm cube.
- [ ] Phase 2 transport and basket placement.
- [ ] Short cloud residual-policy run, only if deterministic control plateaus.
