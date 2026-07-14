"""Custom action terms for stable UR10e control."""

import torch

from isaaclab.envs.mdp import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass


class BoundedRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply joint deltas without allowing their targets to leave soft limits."""

    def apply_actions(self):
        targets = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        targets = torch.clamp(targets, min=limits[..., 0], max=limits[..., 1])
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class BoundedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for bounded relative joint-position actions."""

    class_type: type = BoundedRelativeJointPositionAction


class SlewLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Move a binary joint target gradually instead of applying a target step."""

    cfg: "SlewLimitedBinaryJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._commanded_positions = self._open_command.repeat(self.num_envs, 1)

    def apply_actions(self):
        delta = torch.clamp(
            self._processed_actions - self._commanded_positions,
            min=-self.cfg.max_delta,
            max=self.cfg.max_delta,
        )
        self._commanded_positions.add_(delta)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._commanded_positions = torch.clamp(
            self._commanded_positions, min=limits[..., 0], max=limits[..., 1]
        )
        self._asset.set_joint_position_target(
            self._commanded_positions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._commanded_positions[env_ids] = self._open_command


@configclass
class SlewLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a stable binary position command with a target slew limit."""

    class_type: type = SlewLimitedBinaryJointPositionAction
    max_delta: float = 0.01
