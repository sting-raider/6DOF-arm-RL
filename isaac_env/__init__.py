# Copyright (c) 2026, 6DOF-arm-RL Project
"""UR10e pick-and-place environment for Isaac Lab."""

# Keep package import hardware-independent. Isaac Sim is launched explicitly by
# the training/evaluation entry points before they import ``env_cfg`` or ``mdp``.
# Eager imports here would bootstrap Kit during ordinary unit tests and tooling.
