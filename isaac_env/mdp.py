# Copyright (c) 2026, 6DOF-arm-RL Project
# MDP functions for UR10e pick-and-place with positive reward shaping.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_reward(
    env: ManagerBasedRLEnv,
    ee_link_name: str = "robotiq_85_base_link",
    object_name: str = "object",
    reach_threshold: float = 0.05,
    reach_bonus: float = 10.0,
    distance_scale: float = 1.0,
) -> torch.Tensor:
    """Positive potential-based reach reward.

    reward = (1.0 - dist / max_dist) * distance_scale + shaping * delta_dist
    where shaping = 20.0 (strong improvement signal).
    Avoids the negative-only pitfall.
    """
    # End-effector position
    ee_pos = env.scene["robot"].data.body_pos_w[:, 
        env.scene["robot"].data.body_names.index(ee_link_name)] - env.scene.env_origins

    # Object position  
    obj_pos = env.scene[object_name].data.root_pos_w - env.scene.env_origins

    # Distance
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    max_dist = 1.0  # workspace radius in meters

    # Positive baseline: 1.0 at dist=0, ~0.0 at dist=max_dist
    baseline = (1.0 - dist / max_dist) * distance_scale

    # Potential-based shaping
    shaping = 20.0 * (env._prev_dist - dist)

    reward = baseline + shaping

    # Reach success bonus
    reached = dist < reach_threshold
    reward = torch.where(reached, reward + reach_bonus, reward)

    return reward


def grasp_reward(
    env: ManagerBasedRLEnv,
    ee_link_name: str = "robotiq_85_base_link",
    object_name: str = "object",
    grasp_bonus: float = 5.0,
    lift_bonus: float = 2.0,
) -> torch.Tensor:
    """Grasp phase reward: reach + grasp bonus + lift bonus."""
    # Base reach reward
    base = reach_reward(env, ee_link_name, object_name, reach_bonus=0.0)

    # Check if object is grasped (lifted above table)
    obj_pos = env.scene[object_name].data.root_pos_w
    table_height = 0.8  # roughly where table surface is
    is_lifted = obj_pos[:, 2] > table_height + 0.05

    reward = base
    reward = torch.where(is_lifted, reward + grasp_bonus, reward)
    reward = torch.where(is_lifted & (obj_pos[:, 2] > table_height + 0.10),
                         reward + lift_bonus, reward)

    return reward


def place_reward(
    env: ManagerBasedRLEnv,
    ee_link_name: str = "robotiq_85_base_link",
    object_name: str = "object",
    basket_name: str = "basket",
    place_bonus: float = 50.0,
) -> torch.Tensor:
    """Place phase reward: grasp reward + place bonus."""
    # Base grasp reward
    base = grasp_reward(env, ee_link_name, object_name, grasp_bonus=0.0)

    # Check if object is in basket
    obj_pos = env.scene[object_name].data.root_pos_w
    basket_pos = env.scene[basket_name].data.root_pos_w

    dist_to_basket = torch.norm(obj_pos - basket_pos, dim=-1)
    in_basket = dist_to_basket < 0.15  # basket radius

    reward = base
    reward = torch.where(in_basket, reward + place_bonus, reward)

    return reward


def action_penalty_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on actions to encourage smooth movements."""
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def object_fell(env: ManagerBasedRLEnv, object_name: str = "object",
                min_height: float = 0.1) -> torch.Tensor:
    """Check if object fell off table."""
    obj_pos = env.scene[object_name].data.root_pos_w
    return obj_pos[:, 2] < min_height


def object_in_basket(env: ManagerBasedRLEnv, object_name: str = "object",
                     basket_name: str = "basket", radius: float = 0.15) -> torch.Tensor:
    """Check if object is in the basket."""
    obj_pos = env.scene[object_name].data.root_pos_w
    basket_pos = env.scene[basket_name].data.root_pos_w
    dist = torch.norm(obj_pos[:, :2] - basket_pos[:, :2], dim=-1)
    return (dist < radius) & (obj_pos[:, 2] < basket_pos[:, 2] + 0.1)
