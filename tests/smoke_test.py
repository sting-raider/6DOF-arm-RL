"""Smoke tests that run without Isaac Sim installed."""
import os
import ast
import yaml


def test_repo_structure():
    required = [
        "isaac_env/env_cfg.py",
        "isaac_env/mdp.py",
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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
