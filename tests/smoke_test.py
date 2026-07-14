"""Smoke tests that run without Isaac Sim installed."""
import os
import ast
import yaml

from isaac_env.target_provider import TargetEstimate, TargetTracker
from isaac_env.evaluation_metrics import EpisodeEndAttribution, grasp_stage_label


def test_repo_structure():
    required = [
        "isaac_env/env_cfg.py",
        "isaac_env/mdp.py",
        "isaac_env/actions.py",
        "isaac_env/target_provider.py",
        "isaac_env/evaluation_metrics.py",
        "scripts/train_isaac.py",
        "scripts/evaluate_isaac.py",
        "requirements.txt",
    ]
    for path in required:
        assert os.path.exists(path), f"missing: {path}"


def test_python_syntax():
    for path in [
        "isaac_env/env_cfg.py",
        "isaac_env/mdp.py",
        "isaac_env/actions.py",
        "isaac_env/target_provider.py",
        "isaac_env/evaluation_metrics.py",
        "scripts/train_isaac.py",
        "scripts/evaluate_isaac.py",
    ]:
        with open(path, "r") as f:
            src = f.read()
        ast.parse(src)


def test_yaml_configs():
    for path in ["configs/sac_config.yaml"]:
        if os.path.exists(path):
            with open(path, "r") as f:
                yaml.safe_load(f)


def test_isaac_import_graceful():
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        pass  # Expected on CI without Isaac Sim


def test_target_tracker_accepts_and_smooths_measurements():
    tracker = TargetTracker(smoothing_alpha=0.5)
    first = tracker.update(
        TargetEstimate((0.30, 0.00, 0.85), timestamp_s=1.0, confidence=0.9),
        now_s=1.1,
    )
    second = tracker.update(
        TargetEstimate((0.34, 0.02, 0.87), timestamp_s=1.2, confidence=0.9),
        now_s=1.25,
    )
    assert first.valid and first.position_xyz == (0.30, 0.00, 0.85)
    assert second.valid and second.reason == "accepted"
    assert second.position_xyz == (0.32, 0.01, 0.86)


def test_target_tracker_holds_then_expires_after_bad_detection():
    tracker = TargetTracker(min_confidence=0.7, max_age_s=0.3)
    tracker.update(
        TargetEstimate((0.30, 0.00, 0.85), timestamp_s=2.0, confidence=0.9),
        now_s=2.0,
    )
    held = tracker.update(
        TargetEstimate((0.31, 0.00, 0.85), timestamp_s=2.1, confidence=0.2),
        now_s=2.1,
    )
    expired = tracker.update(None, now_s=2.31)
    assert held.valid and held.reason == "held_after_low_confidence"
    assert not expired.valid and expired.reason == "missing_detection"


def test_target_tracker_rejects_workspace_and_jump_outliers():
    tracker = TargetTracker(max_jump_m=0.05)
    outside = tracker.update(
        TargetEstimate((0.90, 0.00, 0.85), timestamp_s=3.0, confidence=0.9),
        now_s=3.0,
    )
    tracker.update(
        TargetEstimate((0.30, 0.00, 0.85), timestamp_s=3.1, confidence=0.9),
        now_s=3.1,
    )
    jump = tracker.update(
        TargetEstimate((0.40, 0.00, 0.85), timestamp_s=3.2, confidence=0.9),
        now_s=3.2,
    )
    assert not outside.valid and outside.reason == "outside_workspace"
    assert jump.valid and jump.reason == "held_after_implausible_jump"


def test_episode_end_attribution_groups_stage_attempts_and_outcomes():
    attribution = EpisodeEndAttribution()
    attribution.record(
        ("invalid_arm", "invalid_gripper"),
        stage=3,
        attempts_completed=1,
        successful=False,
    )
    attribution.record(
        ("invalid_gripper",),
        stage=3,
        attempts_completed=1,
        successful=True,
    )
    attribution.record(
        ("time_out",),
        stage=4,
        attempts_completed=1,
        successful=False,
    )

    rows = attribution.rows(("invalid_arm", "invalid_gripper"))
    assert [(row.termination, row.stage, row.total) for row in rows] == [
        ("invalid_arm", "retract", 1),
        ("invalid_gripper", "retract", 2),
    ]
    assert rows[0].successful == 0 and rows[0].failed == 1
    assert rows[1].successful == 1 and rows[1].failed == 1


def test_grasp_stage_label_survives_unknown_diagnostic_value():
    assert grasp_stage_label(1) == "descend"
    assert grasp_stage_label(99) == "unknown(99)"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
