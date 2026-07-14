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

The external-force setting produced the worst finger excursion, reaching
-60.539 rad. The 0.75 rad close target improved lift count but increased both
integrity terms, so it is not a safe new preset.

## Verdict

Aggregate integrity counts previously mixed failed grasps with post-success
hold/recovery instability. The gripper problem is nevertheless genuine: the
stock mimic linkage can leave its mechanical range by several radians during
contact. Small target, effort, damping, slew, and solver-force adjustments have
not fixed it without another regression.

The next experiment should isolate the Robotiq linkage in one environment and
compare free closure, object contact, and table contact while recording every
linked joint at physics-step resolution. Do not spend cloud GPU time or retrain
Phase 0 for this issue.
