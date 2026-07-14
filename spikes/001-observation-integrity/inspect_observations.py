#!/usr/bin/env python3
"""Measure raw Isaac observations before they enter RSL-RL normalization."""

import argparse
import os
import sys

import torch

if os.name == "nt":
    import h5py  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--action_mode",
    choices=[
        "zero",
        "random",
        "random_arm_open_gripper",
        "toggle_gripper",
        "close_gripper",
    ],
    default="zero",
)
parser.add_argument(
    "--action_clip",
    type=float,
    default=None,
    help="Symmetric raw-action clip applied before stepping the environment",
)
parser.add_argument(
    "--drive_only_finger_joint",
    action="store_true",
    help="Command only the Robotiq drive joint and leave mimic/passive joints alone",
)
parser.add_argument(
    "--bounded_absolute_arm",
    action="store_true",
    help="Use EMA-smoothed absolute targets constrained to the robot joint limits",
)
parser.add_argument(
    "--bounded_relative_arm",
    action="store_true",
    help="Clamp the existing relative joint targets to the robot's soft limits",
)
parser.add_argument(
    "--reset_invalid_states",
    action="store_true",
    help="Reset an environment before invalid robot/object states reach observations",
)
parser.add_argument("--print_metadata", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import EMAJointPositionToLimitsActionCfg, RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils import configclass

from isaac_env.env_cfg import PickPlaceEnvCfg


class BoundedRelativeJointPositionAction(RelativeJointPositionAction):
    """Disposable variant that clamps relative targets to soft joint limits."""

    def apply_actions(self):
        targets = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        targets = torch.clamp(targets, min=limits[..., 0], max=limits[..., 1])
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class BoundedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type: type = BoundedRelativeJointPositionAction


def invalid_simulation_state(env):
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints(
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        preserve_order=True,
    )
    finger_id = robot.data.joint_names.index("finger_joint")
    arm_pos = robot.data.joint_pos[:, arm_ids]
    arm_vel = robot.data.joint_vel[:, arm_ids]
    finger_pos = robot.data.joint_pos[:, finger_id]
    obj_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    return (
        (~torch.isfinite(arm_pos)).any(dim=1)
        | (~torch.isfinite(arm_vel)).any(dim=1)
        | (~torch.isfinite(finger_pos))
        | (~torch.isfinite(obj_pos)).any(dim=1)
        | (arm_pos.abs() > 12.0).any(dim=1)
        | (arm_vel.abs() > 100.0).any(dim=1)
        | (finger_pos.abs() > 1.5)
        | (obj_pos[:, :2].abs() > 2.0).any(dim=1)
        | (obj_pos[:, 2] > 2.0)
    )


TERM_SLICES = {
    "joint_pos": slice(0, 6),
    "joint_vel": slice(6, 12),
    "ee_pos_scaled": slice(12, 15),
    "ee_quat": slice(15, 19),
    "gripper_state": slice(19, 20),
    "object_pos": slice(20, 23),
    "relative_pos": slice(23, 26),
    "distance": slice(26, 27),
    "actions": slice(27, 34),
}

PLAUSIBLE_ABS_MAX = {
    "joint_pos": 20.0,
    "joint_vel": 100.0,
    "ee_pos_scaled": 5.0,
    "ee_quat": 2.0,
    "gripper_state": 2.0,
    "object_pos": 5.0,
    "relative_pos": 5.0,
    "distance": 5.0,
    "actions": 20.0,
}


def main():
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    if args.bounded_absolute_arm:
        cfg.actions.arm_action = EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=1.0,
            rescale_to_limits=True,
            alpha=0.1,
        )
    elif args.bounded_relative_arm:
        cfg.actions.arm_action = BoundedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.07,
            use_zero_offset=True,
        )
    if args.drive_only_finger_joint:
        cfg.actions.gripper_action.joint_names = ["finger_joint"]
        cfg.actions.gripper_action.open_command_expr = {"finger_joint": 0.0}
        cfg.actions.gripper_action.close_command_expr = {"finger_joint": 0.785398163}
    if not args.reset_invalid_states:
        # Let the spike observe the first bad state instead of allowing the
        # production integrity terms to reset it before observations are built.
        cfg.terminations.invalid_arm = None
        cfg.terminations.invalid_gripper = None
        cfg.terminations.invalid_object = None
    raw_env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(raw_env)
    obs = env.get_observations()
    if args.print_metadata:
        obs, _, _, _ = env.step(torch.zeros((args.num_envs, 7), device=raw_env.device))
        robot = raw_env.scene["robot"]
        print("\nRobot body names:")
        print(robot.data.body_names)
        print("\nRobot joint names:")
        print(robot.data.joint_names)
        print("\nSoft joint position limits (env 0):")
        print(robot.data.soft_joint_pos_limits[0].detach().cpu())
        print("\nKey body positions in env-local coordinates (env 0):")
        for body_name in ("wrist_3_link", "base_link_0", "left_inner_finger", "right_inner_finger"):
            body_id = robot.data.body_names.index(body_name)
            local_pos = robot.data.body_pos_w[0, body_id] - raw_env.scene.env_origins[0]
            print(body_name, local_pos.detach().cpu())
    minima = torch.full((34,), torch.inf, device=raw_env.device)
    maxima = torch.full((34,), -torch.inf, device=raw_env.device)
    first_bad = None
    resets = 0

    for step in range(args.steps + 1):
        policy_obs = obs["policy"]
        minima = torch.minimum(minima, policy_obs.amin(dim=0))
        maxima = torch.maximum(maxima, policy_obs.amax(dim=0))

        if first_bad is None:
            for name, obs_slice in TERM_SLICES.items():
                values = policy_obs[:, obs_slice]
                bad = (~torch.isfinite(values)) | (values.abs() > PLAUSIBLE_ABS_MAX[name])
                if bad.any():
                    env_id, local_feature = torch.nonzero(bad, as_tuple=False)[0].tolist()
                    first_bad = {
                        "step": step,
                        "env": env_id,
                        "term": name,
                        "feature": obs_slice.start + local_feature,
                        "value": policy_obs[env_id, obs_slice.start + local_feature].item(),
                    }

        if step == args.steps:
            break
        if args.action_mode == "zero":
            actions = torch.zeros((args.num_envs, 7), device=raw_env.device)
        elif args.action_mode == "close_gripper":
            actions = torch.zeros((args.num_envs, 7), device=raw_env.device)
            actions[:, 6] = -1.0
        else:
            actions = torch.randn((args.num_envs, 7), device=raw_env.device)
            if args.action_mode == "random_arm_open_gripper":
                actions[:, 6] = 1.0
            elif args.action_mode == "toggle_gripper":
                actions[:, :6] = 0.0
        if args.action_clip is not None:
            actions = torch.clamp(actions, -args.action_clip, args.action_clip)
        obs, _, dones, _ = env.step(actions)
        resets += int(dones.sum().item())

    print("\nRaw observation ranges")
    print("=" * 72)
    for name, obs_slice in TERM_SLICES.items():
        term_min = minima[obs_slice].detach().cpu().numpy()
        term_max = maxima[obs_slice].detach().cpu().numpy()
        abs_max = max(abs(float(term_min.min())), abs(float(term_max.max())))
        print(f"{name:17s} abs_max={abs_max:12.4g} min={term_min} max={term_max}")
    print("=" * 72)
    print(f"First implausible value: {first_bad}")
    print(f"Total resets (including timeouts): {resets}")
    print("Final gripper linkage positions (env 0):")
    gripper_joint_names = robot.data.joint_names[6:]
    gripper_joint_pos = robot.data.joint_pos[0, 6:].detach().cpu().numpy()
    for name, position in zip(gripper_joint_names, gripper_joint_pos):
        print(f"  {name:38s} {position: .6f}")

    env.close()
    if first_bad is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
