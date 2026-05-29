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
    """Object centroid position (3D) in local env frame (relative to env origin)."""
    obj = env.scene[object_name]
    # root_pos_w is world-frame; subtract env_origins to get local frame
    return obj.data.root_pos_w - env.scene.env_origins


def relative_position(
    env: ManagerBasedRLEnv,
    ee_link: str = "wrist_3_link",
    object_name: str = "object",
) -> torch.Tensor:
    """Vector from end-effector to object in local env frame."""
    return object_position(env, object_name) - ee_position(env, ee_link)



def gripper_state(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Gripper open/close state as a single float per env [0=open, 0.785=closed]."""
    robot = env.scene[asset_name]
    finger_idx = robot.data.joint_names.index("finger_joint")
    return robot.data.joint_pos[:, finger_idx : finger_idx + 1]


# =============================================================================
# SCALED OBSERVATIONS (for use when obs_normalization is disabled)
# =============================================================================

def ee_position_scaled(env, link_name="wrist_3_link"):
    """EE position scaled to ~[-1, 1]."""
    return ee_position(env, link_name) / 1.5

def object_position_scaled(env, object_name="object"):
    """Object position scaled to ~[0, 1]."""
    return object_position(env, object_name) / 1.0

def gripper_state_scaled(env, asset_name="robot"):
    """Gripper state scaled to [0, 1]."""
    return gripper_state(env, asset_name) / 0.785398163


def ee_to_object_distance(
    env: ManagerBasedRLEnv,
    ee_link: str = "wrist_3_link",
) -> torch.Tensor:
    """Scalar distance from end-effector to object centroid."""
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(ee_link)
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    obj_pos = object_position(env)
    return torch.norm(ee_pos - obj_pos, dim=1, keepdim=True)


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

    pose_range values are LOCAL offsets relative to each env's origin.
    write_root_state_to_sim expects world-frame positions, so we add env_origins.
    """
    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]  # (N, 3) world-frame env origins

    # Build local position, then convert to world frame by adding env_origins
    local_pos = torch.zeros(len(env_ids), 3, device=env.device)
    if "x" in pose_range:
        local_pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["x"])
    else:
        # Keep current local offset: world_pos - origin
        local_pos[:, 0] = asset.data.root_pos_w[env_ids, 0] - origins[:, 0]
    if "y" in pose_range:
        local_pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["y"])
    else:
        local_pos[:, 1] = asset.data.root_pos_w[env_ids, 1] - origins[:, 1]
    if "z" in pose_range:
        local_pos[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["z"])
    else:
        local_pos[:, 2] = asset.data.root_pos_w[env_ids, 2] - origins[:, 2]

    # Convert local → world frame
    pos = local_pos + origins

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

    # Distances
    ee_to_obj = torch.norm(ee_pos - obj_pos, dim=1)

    # Basket centre in local env frame (env_cfg: basket init pos = (0.6, 0.0, 0.85))
    # obj_pos is already in local frame, so basket_center must also be local (same for all envs)
    basket_center = torch.tensor([0.6, 0.0, 0.85], device=env.device, dtype=torch.float32)
    obj_to_basket = torch.norm(obj_pos - basket_center, dim=1)

    # Unified finger joint index and closedness
    finger_idx = robot.data.joint_names.index("finger_joint")
    gripper_joint_pos = robot.data.joint_pos[:, finger_idx]
    closedness = torch.clamp(gripper_joint_pos / 0.785398163, 0.0, 1.0)

    if phase == 0:
        # ── REACH: dense shaping + sparse success bonuses ──
        reach = torch.exp(-ee_to_obj / 0.10)
        bonus_8cm = 2.0 * (ee_to_obj < 0.08).float()
        bonus_5cm = 1.0 * (ee_to_obj < 0.05).float()
        return reach + bonus_8cm + bonus_5cm

    elif phase == 1:
        # ── GRASP: reach + close bonus + grasp + lift ──
        # Reach: strong base
        reach = torch.exp(-ee_to_obj / 0.05) * 0.5
        
        # Close bonus: reward actual gripper closure when near object
        # Uses joint position (closedness) — works in reward context
        is_near = (ee_to_obj < 0.08).float()
        close_bonus = 1.5 * is_near * closedness  # 0=open, 1=closed
        
        # Hover penalty: small cost for being near but not closing
        hover_penalty = -0.05 * is_near * (1.0 - closedness)
        
        # Grasp detection
        is_grasping = (closedness > 0.5) & (ee_to_obj < 0.03)
        grasp_reward = is_grasping.float() * 1.0
        
        # Lift reward
        obj_z = obj_pos[:, 2]
        height_above = torch.clamp(obj_z - 0.80 - 0.02, 0.0, 0.10)
        lift_shaping = (height_above / 0.08) * 2.0
        
        lift_bonus = 2.0 * (height_above > 0.05).float() * is_grasping.float()
        
        return reach + close_bonus + hover_penalty + grasp_reward + lift_shaping + lift_bonus

    elif phase == 2:
        # ── PLACE: Phase 1 reach+grasp + basket bonus ──
        reach = torch.exp(-ee_to_obj / std)
        near_mask = torch.exp(-ee_to_obj / (std * 3.0))
        grasp_bonus = 0.5 * closedness * near_mask
        # Basket bonus: reward object being close to basket
        basket_bonus = torch.exp(-obj_to_basket / std)
        return reach + grasp_bonus + basket_bonus

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
