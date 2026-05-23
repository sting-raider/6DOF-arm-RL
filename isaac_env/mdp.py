# Copyright (c) 2026, 6DOF-arm-RL Project
# MDP functions for UR10e pick-and-place with positive reward shaping.
# Complete set — actions, observations, rewards, terminations, events.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# =============================================================================
# ACTION TERMS
# =============================================================================

def joint_pos_delta_action(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    scale: float = 0.1,
) -> torch.Tensor:
    """Apply delta joint position commands (5 arm joints).

    Maps actions in [-1, 1] to position deltas scaled by `scale` radians.
    The 6th action dimension (gripper) is handled by gripper_action.

    Returns the clipped target joint positions (used by the action manager).
    """
    asset = env.scene[asset_name]
    # Current joint positions (first 5 arm joints)
    current_pos = asset.data.joint_pos[:, :5]
    # Action from policy (first 5 dims)
    action = env.action_manager.action[:, :5]
    # Target = current + scaled delta
    target_pos = current_pos + action * scale
    # Clip to joint limits
    limits = asset.data.joint_limits[:, :5, :]
    target_pos = torch.clamp(target_pos, limits[..., 0], limits[..., 1])
    return target_pos


def gripper_action(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    velocity: float = 0.1,
) -> torch.Tensor:
    """Binary gripper action from the 6th action dimension.

    Positive action (>0) → close gripper to `closed_pos`.
    Negative action (≤0) → open gripper to `open_pos`.
    Uses direct position control with a fixed velocity.
    """
    asset = env.scene[asset_name]
    action = env.action_manager.action[:, 5:6]  # (num_envs, 1)

    # Target position: binary open/close based on action sign
    target = torch.where(action > 0, closed_pos, open_pos)

    # Get current gripper joint position (last DOF)
    current = asset.data.joint_pos[:, -1:]

    # Move toward target at fixed velocity per step
    step_max = velocity * env.step_dt
    delta = torch.clamp(target - current, -step_max, step_max)

    return current + delta


# =============================================================================
# OBSERVATION TERMS
# =============================================================================

def ee_position(
    env: ManagerBasedRLEnv,
    link_name: str = "wrist_3_link",
) -> torch.Tensor:
    """End-effector position (3D) in world frame, relative to env origin."""
    ee_pos_w = env.scene["robot"].data.body_pos_w[
        :, env.scene["robot"].data.body_names.index(link_name)
    ]
    return ee_pos_w - env.scene.env_origins


def object_position(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
) -> torch.Tensor:
    """Object centroid position (3D). root_pos_w is already local (relative to env origin)."""
    obj = env.scene[object_name]
    return obj.data.root_pos_w  # already local — no env_origins subtraction needed


def gripper_state(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Gripper open/close state as a single float per env [0=open, 0.04=closed]."""
    return env.scene[asset_name].data.joint_pos[:, -1:]  # last DOF


# =============================================================================
# TERMINATION TERMS
# =============================================================================

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if episode has reached max length. Handled automatically by Isaac Lab."""
    return env.episode_length_buf >= env.max_episode_length


# =============================================================================
# EVENT TERMS (domain randomization / reset)
# =============================================================================

def reset_joints_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str = "robot",
    position_range: tuple = (0.0, 0.0),
    velocity_range: tuple = (0.0, 0.0),
) -> None:
    """Reset robot joint positions with optional uniform random offsets."""
    asset = env.scene[asset_name]
    num_joints = asset.num_joints

    # Base positions (default joint state)
    default_pos = asset.data.default_joint_pos[env_ids]

    # Add random offset
    pos_offset = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(
        *position_range
    )
    vel_offset = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(
        *velocity_range
    )

    joint_pos = default_pos + pos_offset
    joint_vel = vel_offset

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    velocity_range: dict,
) -> None:
    """Reset a rigid object's root state with uniform random pose + velocity.

    pose_range values are treated as ABSOLUTE world coordinates (not offsets).
    Default quaternion and zero velocities are used.
    """
    asset = env.scene[asset_cfg.name]

    # Build position tensor from scratch using absolute ranges
    pos = torch.zeros(len(env_ids), 3, device=env.device)
    if "x" in pose_range:
        pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["x"])
    else:
        pos[:, 0] = asset.data.root_pos_w[env_ids, 0]
    if "y" in pose_range:
        pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["y"])
    else:
        pos[:, 1] = asset.data.root_pos_w[env_ids, 1]
    if "z" in pose_range:
        pos[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["z"])
    else:
        pos[:, 2] = asset.data.root_pos_w[env_ids, 2]

    # Orientation — use default identity quaternion
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)

    # Velocity — zero or randomized
    vel = torch.zeros(len(env_ids), 6, device=env.device)
    if "x" in velocity_range:
        vel[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*velocity_range["x"])
    if "y" in velocity_range:
        vel[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*velocity_range["y"])
    if "z" in velocity_range:
        vel[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*velocity_range["z"])

    root_state = torch.cat([pos, quat, vel], dim=-1)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)


# =============================================================================
# REWARD TERMS
# =============================================================================

def action_penalty_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize large actions using L2 norm (smooth, non-zero gradients everywhere).

    Reward = -sum(action²)  → weighted by the reward term's weight (e.g. -0.01).
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def reach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    ee_link: str = "wrist_3_link",
) -> torch.Tensor:
    """Phase-aware reach/grasp/place reward.

    Reads ``env.cfg.curriculum_phase`` to dispatch:
      Phase 0 (REACH):  exp(-||ee_xyz - object_xyz|| / std)
      Phase 1 (GRASP):  Phase 0 reward + grasp bonus when gripper is closed near object
      Phase 2 (PLACE):  exp(-||object_xyz - basket_xyz|| / std)

    Returns a per-environment scalar in [0, 1] (plus grasp bonus in [0, 0.5]).
    """
    # End-effector position
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(ee_link)
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins

    # Object position
    obj_pos = object_position(env)

    phase = env.cfg.curriculum_phase

    # DEBUG: print first env's values on first call
    if not hasattr(reach_reward, '_debug_printed'):
        reach_reward._debug_printed = True

    # Distances
    ee_to_obj = torch.norm(ee_pos - obj_pos, dim=1)

    # Basket centre (from env_cfg: basket at (0.6, 0.0, 0.80))
    basket_center = torch.tensor([0.6, 0.0, 0.80], device=env.device, dtype=torch.float32)
    obj_to_basket = torch.norm(obj_pos - basket_center, dim=1)

    if phase == 0:
        # ── REACH: distance from EE to object ──
        return torch.exp(-ee_to_obj / std)

    elif phase == 1:
        # ── GRASP: reach + gripper-closed bonus ──
        reach = torch.exp(-ee_to_obj / std)
        gripper_joint_pos = robot.data.joint_pos[:, -1]  # last DOF
        # Gripper is "closed" near object → bonus scales with how closed
        closedness = torch.clamp(gripper_joint_pos / 0.04, 0.0, 1.0)
        # Bonus only when EE is close to object (within ~3× std)
        near_mask = torch.exp(-ee_to_obj / (std * 3.0))
        grasp_bonus = 0.5 * closedness * near_mask
        return reach + grasp_bonus

    elif phase == 2:
        # ── PLACE: object near basket ──
        return torch.exp(-obj_to_basket / std)

    else:
        return torch.zeros(env.num_envs, device=env.device)


# =============================================================================
# TERMINATION TERMS
# =============================================================================

def object_fell(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
) -> torch.Tensor:
    """Terminate when the object's z-position drops below ``minimum_height``.

    The object starts at z ≈ 0.85 (on the table). If it falls below ~0.5 it
    has likely dropped off the table.
    """
    obj_z = env.scene["object"].data.root_pos_w[:, 2]
    return obj_z < minimum_height
