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


# Shared task geometry in each environment's local frame.  Keeping these values
# here prevents training rewards and evaluation metrics from silently disagreeing.
OBJECT_REST_HEIGHT = 0.83
LIFT_SUCCESS_DELTA = 0.05
BASKET_CENTER = (-0.35, 0.0, 0.85)
GRASP_TARGET_OFFSET = (0.0, 0.0, 0.19)
DESIRED_WRIST_QUAT = (0.0, 2**-0.5, 2**-0.5, 0.0)


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


def ee_orientation(
    env: ManagerBasedRLEnv,
    link_name: str = "wrist_3_link",
) -> torch.Tensor:
    """End-effector orientation as a world-frame quaternion (w, x, y, z)."""
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(link_name)
    return robot.data.body_quat_w[:, ee_idx]


def object_position(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
) -> torch.Tensor:
    """Object centroid position (3D) in local env frame (relative to env origin)."""
    obj = env.scene[object_name]
    # root_pos_w is world-frame; subtract env_origins to get local frame
    return obj.data.root_pos_w - env.scene.env_origins


def policy_target_position(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
) -> torch.Tensor:
    """Target supplied to the policy, defaulting to simulator truth.

    A camera adapter may set ``env._policy_target_position`` to an ``(N, 3)``
    tensor. Rewards and success metrics continue to call :func:`object_position`
    and therefore remain grounded in physical simulator truth.
    """
    override = getattr(env, "_policy_target_position", None)
    if override is None:
        return object_position(env, object_name)
    if override.shape != (env.num_envs, 3):
        raise ValueError(
            "env._policy_target_position must have shape "
            f"({env.num_envs}, 3), got {tuple(override.shape)}"
        )
    return override


def grasp_target_position(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
) -> torch.Tensor:
    """Top-down wrist target that places the 2F-85 fingers around the object."""
    offset = torch.tensor(
        GRASP_TARGET_OFFSET, device=env.device, dtype=torch.float32
    )
    return object_position(env, object_name) + offset


def policy_grasp_target_position(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
) -> torch.Tensor:
    """Policy target plus the fixed wrist pre-grasp offset."""
    offset = torch.tensor(
        GRASP_TARGET_OFFSET, device=env.device, dtype=torch.float32
    )
    return policy_target_position(env, object_name) + offset


def relative_position(
    env: ManagerBasedRLEnv,
    ee_link: str = "wrist_3_link",
    object_name: str = "object",
) -> torch.Tensor:
    """Vector from wrist to the top-down pre-grasp target in local frame."""
    return policy_grasp_target_position(env, object_name) - ee_position(env, ee_link)



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


def policy_target_position_scaled(env, object_name="object"):
    """Camera/simulator target channel, kept separate from reward truth."""
    return policy_target_position(env, object_name) / 1.0

def gripper_state_scaled(env, asset_name="robot"):
    """Gripper state scaled to [0, 1]."""
    return gripper_state(env, asset_name) / 0.785398163


def ee_to_object_distance(
    env: ManagerBasedRLEnv,
    ee_link: str = "wrist_3_link",
) -> torch.Tensor:
    """Scalar distance from wrist to the top-down pre-grasp target."""
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(ee_link)
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    target_pos = policy_grasp_target_position(env)
    return torch.norm(ee_pos - target_pos, dim=1, keepdim=True)


def wrist_orientation_error(
    env: ManagerBasedRLEnv,
    ee_link: str = "wrist_3_link",
) -> torch.Tensor:
    """Shortest angular error to the fixed top-down grasp orientation, in radians."""
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(ee_link)
    wrist_quat = robot.data.body_quat_w[:, ee_idx]
    desired = torch.tensor(
        DESIRED_WRIST_QUAT, device=env.device, dtype=wrist_quat.dtype
    ).expand_as(wrist_quat)
    similarity = torch.sum(wrist_quat * desired, dim=1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(similarity)


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
      Phase 0 (PRE-GRASP): approach the pose above the object, then align the wrist
      Phase 1 (GRASP): Phase 0 reward + grasp/lift bonus
      Phase 2 (PLACE):  exp(-||object_xyz - basket_xyz|| / std)

    Phase 0 couples position and orientation: a broad distance term guides the
    initial approach, while a fine orientation-weighted term becomes important
    only near the object.  This prevents either objective from being collected
    in isolation.
    """
    # End-effector position
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(ee_link)
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins

    # Object position
    obj_pos = object_position(env)

    phase = env.cfg.curriculum_phase

    # Distance and orientation error for the physically valid top-down pre-grasp.
    grasp_target = grasp_target_position(env)
    ee_to_obj = torch.norm(ee_pos - grasp_target, dim=1)
    orientation_error = wrist_orientation_error(env, ee_link)
    orientation_score = 1.0 - orientation_error / torch.pi
    aligned = (orientation_error < 0.785398163).float()

    # Smooth approach-then-align curriculum shared by Phases 0 and 1.
    coarse_reach = torch.exp(-ee_to_obj / 0.25)
    fine_reach = torch.exp(-ee_to_obj / 0.08)
    coupled_pregrasp = coarse_reach + 3.0 * fine_reach * (
        0.25 + torch.square(orientation_score)
    )

    # Basket centre in local env frame.
    # obj_pos is already in local frame, so basket_center must also be local (same for all envs)
    basket_center = torch.tensor(BASKET_CENTER, device=env.device, dtype=torch.float32)
    obj_to_basket = torch.norm(obj_pos - basket_center, dim=1)

    # Unified finger joint index and closedness
    finger_idx = robot.data.joint_names.index("finger_joint")
    gripper_joint_pos = robot.data.joint_pos[:, finger_idx]
    closedness = torch.clamp(gripper_joint_pos / 0.785398163, 0.0, 1.0)

    if phase == 0:
        # ── PRE-GRASP: position + top-down wrist alignment ──
        bonus_8cm = 2.0 * (ee_to_obj < 0.08).float() * aligned
        bonus_5cm = 1.0 * (ee_to_obj < 0.05).float() * aligned
        grasp_ready = (
            (ee_to_obj < 0.06) & (orientation_error < 0.436332313)
        ).float()
        return coupled_pregrasp + bonus_8cm + bonus_5cm + 6.0 * grasp_ready

    elif phase == 1:
        # ── GRASP: reach + lift reward gated on grasp ──
        # Reach: same as Phase 0 — maintains approach behavior
        grasp_ready = (
            (ee_to_obj < 0.06) & (orientation_error < 0.436332313)
        ).float()
        reach = (
            coupled_pregrasp
            + 2.0 * (ee_to_obj < 0.08).float() * aligned
            + 1.0 * (ee_to_obj < 0.05).float() * aligned
            + 6.0 * grasp_ready
        )

        # A grasp is only credited when the physically actuated finger has moved,
        # the wrist remains near the cube, and the cube actually rises.  The old
        # z>0.84 check was already true at the 0.85 m reset pose.
        grasp_contact_proxy = (closedness > 0.10) & (ee_to_obj < 0.08)
        obj_z = obj_pos[:, 2]
        lift_height = torch.clamp(obj_z - OBJECT_REST_HEIGHT, 0.0, 0.15)
        lifted = lift_height > LIFT_SUCCESS_DELTA
        lift_shaping = 10.0 * lift_height * grasp_contact_proxy.float()
        lift_bonus = 4.0 * lifted.float() * grasp_contact_proxy.float()

        return reach + lift_shaping + lift_bonus

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


def invalid_arm_state(
    env: ManagerBasedRLEnv,
    max_arm_position: float = 12.0,
    max_arm_velocity: float = 100.0,
) -> torch.Tensor:
    """Detect unstable active-arm states before they reach observations."""
    robot = env.scene["robot"]
    if not hasattr(env, "_integrity_arm_joint_ids"):
        arm_joint_names = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        env._integrity_arm_joint_ids = [
            robot.data.joint_names.index(name) for name in arm_joint_names
        ]

    arm_pos = robot.data.joint_pos[:, env._integrity_arm_joint_ids]
    arm_vel = robot.data.joint_vel[:, env._integrity_arm_joint_ids]
    return (
        (~torch.isfinite(arm_pos)).any(dim=1)
        | (~torch.isfinite(arm_vel)).any(dim=1)
        | (arm_pos.abs() > max_arm_position).any(dim=1)
        | (arm_vel.abs() > max_arm_velocity).any(dim=1)
    )


def invalid_gripper_state(
    env: ManagerBasedRLEnv,
    max_finger_position: float = 1.5,
) -> torch.Tensor:
    """Detect a non-finite or mechanically impossible drive-joint state."""
    robot = env.scene["robot"]
    if not hasattr(env, "_integrity_finger_joint_id"):
        env._integrity_finger_joint_id = robot.data.joint_names.index("finger_joint")
    finger_pos = robot.data.joint_pos[:, env._integrity_finger_joint_id]
    return (~torch.isfinite(finger_pos)) | (finger_pos.abs() > max_finger_position)


def invalid_object_state(
    env: ManagerBasedRLEnv,
    max_object_xy: float = 2.0,
    max_object_height: float = 2.0,
) -> torch.Tensor:
    """Detect a non-finite object pose or an object launched out of the scene."""
    obj_pos = object_position(env)
    return (
        (~torch.isfinite(obj_pos)).any(dim=1)
        | (obj_pos[:, :2].abs() > max_object_xy).any(dim=1)
        | (obj_pos[:, 2] > max_object_height)
    )


def invalid_simulation_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Backward-compatible combined integrity predicate used by spike scripts."""
    return invalid_arm_state(env) | invalid_gripper_state(env) | invalid_object_state(env)
