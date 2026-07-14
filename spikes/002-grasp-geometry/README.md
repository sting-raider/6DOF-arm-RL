# Grasp geometry probe

## Question

Does the simulated Robotiq 2F-85 physically capture the 4 cm cube when the
UR10e wrist is placed at the current 19 cm pre-grasp offset, or is Phase 1
failing because the grasp height is wrong?

## Minimal experiment

`probe_grasp.py` assigns one wrist-height offset to each parallel environment,
uses Isaac Lab differential IK to hold the desired top-down pose, closes the
real simulated gripper through the production action term, and retracts the
wrist by 12 cm.  It reports object lift and finger travel for every offset.

This controller is diagnostic only.  It bypasses PPO so the experiment tests
contact geometry rather than policy exploration.

## Verdict

The original 4 cm cube is a poor first grasp curriculum for this gripper. Across
the tested descent depths, its useful vertical contact band was too narrow:
shallower grasps closed above the cube, while deeper grasps introduced table
contact and instability. None of those cube probes produced repeatable lifts.

The same physical colliders and Robotiq mimic linkage successfully lifted a
4 x 4 x 10 cm upright block, proving that the simulator contact path works. The
best low-cost controller configuration was:

- 8 cm descent from the learned 19 cm wrist pre-grasp offset;
- 0.78 rad close target;
- 0.45 rad close-to-retract transition;
- 2 cm maximum Cartesian retract request per control tick;
- object friction 1.0;
- full 6-DOF damped-least-squares pose control during contact.

With the existing Phase 1 warm-start checkpoint, strict lift success was 30/64
(47%) on seed 19595 and 28/64 (44%) on seed 42. The corresponding one-switch
configuration is `scripts/evaluate_isaac.py --hybrid_phase1`.

Conclusion: Phase 1 was limited primarily by grasp geometry and sequencing, not
by a broken gripper collider or a need to retrain Phase 0. Continue with the tall
starter object, reactive retries, and camera-target perturbation tests before
spending more GPU time on PPO.

## Reactive follow-up

A failed first retract now returns to pre-grasp, opens, measures the gripper
midpoint's XY miss, and applies the opposite correction on one retry. A stable
+7 mm Y calibration offset is also applied from the first attempt. This raised
strict lift success to 37/64 (58%) on seed 42 and 41/64 (64%) on seed 19595:
78/128, or 60.9%, without additional training.

## Remaining limiter

On the final seed-42 preset run, episode endings included 37 timeouts, 11 object
falls, 5 arm-integrity resets, 19 gripper-integrity resets, and 1 object-integrity
reset (more than one term may fire together). Gripper damping values of 2 and 4
did not reduce resets; a 5-unit effort cap reduced gripper resets to 11 but also
reduced strict success to 53%. A 7.5-unit cap produced 16 gripper resets and 53%
success. The original damping and 10-unit effort limit therefore remain in the
preset while contact stability is treated as the next separate engineering task.

Other rejected ablations:

- 12 cm block with 9 cm descent: 0/64 lifts because closure was obstructed;
- 0.02 rad/tick gripper ramp: 17/64 lifts (27%);
- 0.70 rad terminal close: 35/64 lifts (55%), slightly below the 0.78 baseline;
- three attempts: 38/64 on seed 42, only one success above the two-attempt
  calibrated result while adding ten seconds.
