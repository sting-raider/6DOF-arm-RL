# Contact-stability attribution

## Question

Are the nominal Phase 1 integrity resets failed grasps, and which controller
stage and physical state produce them?

## Instrumentation

The evaluator now snapshots the grasp stage and number of completed closes
before Isaac resets an environment. It correlates every active termination term
with strict lift success and preserves the invalid joint values that would
otherwise be replaced by the next episode's initial state.

The aggregation code is hardware-independent and covered by the smoke suite.
Isaac termination terms may overlap, so term counts are not assumed to sum to
the episode count.

## Nominal seed-42 result

The unchanged hybrid preset reproduced its established result exactly:

- 37/64 strict lifts;
- 19 gripper-integrity terminations;
- 5 arm-integrity terminations;
- 1 object-integrity termination.

Fourteen of the 19 gripper terminations occurred after an episode had already
achieved a strict lift. The five pre-success gripper terminations consisted of
four during the first close and one during initial descent. All five arm
terminations occurred after a strict lift: two during second-attempt retract and
three during recovery after the first close.

The gripper termination is a real physics excursion, not an overly tight guard.
Although the commanded close target is 0.78 rad, invalid drive-joint positions
ranged from -2.834 to +7.183 rad. Every arm termination was the
`wrist_2_joint` slightly crossing the 12 rad guard (12.034 to 12.165 rad); its
velocity remained between 2.756 and 3.239 rad/s.

## Controlled ablations

Each row changed one setting from the deterministic seed-42 baseline.

| Change | Strict lifts | Gripper resets | Arm resets | Verdict |
|---|---:|---:|---:|---|
| Baseline | 37/64 | 19 | 5 | Keep |
| Apply external forces every solver iteration | 37/64 | 24 | 10 | Reject |
| Hold the gripper target after strict lift | 36/64 | 18 | 2 | Reject: reduced lift |
| Close target 0.75 rad | 40/64 | 23 | 6 | Reject: less stable |
| Gripper slew 0.005 rad/physics tick | 34/64 | 20 | 5 | Reject |
| Freeze arm during close, target 0.78 rad | 36/64 | 0 | 5 | Reject: reduced lift |
| Freeze arm during close, target 0.75 rad | 37/64 | 0 | 3 | Keep |
| Position-only retry recovery on kept close preset | 34/64 | 1 | 3 | Reject |

The external-force setting produced the worst finger excursion, reaching
-60.539 rad. The 0.75 rad close target improved lift count but increased both
integrity terms, so it is not a safe new preset.

## Physics-step contact probe

`probe_linkage.py` runs free closure, static object contact, and production-
relative table clearance side by side at one observation per PhysX step. All
three scenarios remained finite for 120 close steps. Free and table-clearance
closure reached the 0.78 rad command. The object-contact drive joint stopped at
0.584 rad, demonstrating that the cube was physically blocking the fingers
without destabilizing the linkage.

The probe also confirmed hard arm limits of +/-6.283 rad for wrist 2. The
roughly 12.1 rad values attributed during recovery are consistent with
continuous-coordinate wrap rather than the configured hard-limit value.

## Kept preset

Pausing Cartesian arm correction during finger closure and lowering the close
target to 0.75 rad became the hybrid default. The exact default produced:

| Seed | Strict lifts | Gripper resets | Arm resets |
|---:|---:|---:|---:|
| 42 | 37/64 | 0 | 3 |
| 19595 | 39/64 | 1 | 9 |
| Combined | 76/128 (59.4%) | 1/128 (0.8%) | 12/128 (9.4%) |

This trades two strict lifts versus the previous 78/128 result for a much
safer contact path. It does not satisfy the overall integrity gate because the
retry recovery controller still winds wrist 2 after successful lifts.
The sole remaining gripper reset occurred during descent before any close: the
drive reached 1.942 rad while the left inner-finger velocity peaked at 84.9
rad/s. It is therefore distinct from the nominal static-contact path.

## Verdict

Aggregate integrity counts previously mixed failed grasps with post-success
hold/recovery instability. Static closure and object contact are stable; the
explosion was observed only with simultaneous arm servo motion through contact.
Decoupling those actions nearly eliminates the gripper failure without
retraining.

The next contact-stability task is wrist-aware retry recovery. It should prevent
continuous-coordinate wrap without changing the successful first-attempt
trajectory. Do not spend cloud GPU time or retrain Phase 0 for this issue.
