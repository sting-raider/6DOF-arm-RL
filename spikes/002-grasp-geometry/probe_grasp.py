#!/usr/bin/env python3
"""Sweep wrist heights and test close-then-retract grasp physics."""

import argparse
import os
import sys

import torch

if os.name == "nt":
    import h5py  # noqa: F401 -- preload the matching HDF5 DLL for Isaac Sim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--offsets",
    type=float,
    nargs="+",
    default=[0.15, 0.19, 0.23],
    help="Wrist heights above the cube center, in metres (one environment each)",
)
parser.add_argument("--approach_steps", type=int, default=60)
parser.add_argument("--close_steps", type=int, default=70)
parser.add_argument("--lift_steps", type=int, default=60)
parser.add_argument("--lift_distance", type=float, default=0.12)
parser.add_argument(
    "--copies_per_offset",
    type=int,
    default=16,
    help="Parallel randomized trials for each height (larger batches run faster on GPU PhysX)",
)
parser.add_argument(
    "--ik_interval",
    type=int,
    default=5,
    help="Recompute the Jacobian solution every N simulation steps",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab.envs import ManagerBasedRLEnv

from isaac_env.env_cfg import PickPlaceEnvCfg
from isaac_env.mdp import DESIRED_WRIST_QUAT


def quaternion_rotation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Shortest world-frame rotation vector from current to desired (wxyz quaternions)."""
    current_conjugate = current.clone()
    current_conjugate[:, 1:] *= -1.0
    aw, ax, ay, az = desired.unbind(dim=-1)
    bw, bx, by, bz = current_conjugate.unbind(dim=-1)
    error = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
    error = torch.where(error[:, :1] < 0.0, -error, error)
    vector_norm = torch.norm(error[:, 1:], dim=-1, keepdim=True).clamp_min(1.0e-8)
    angle = 2.0 * torch.atan2(vector_norm, error[:, :1].clamp_min(1.0e-8))
    return error[:, 1:] / vector_norm * angle


def main():
    device = args.device or "cuda:0"
    unique_offsets = torch.tensor(args.offsets, device=device, dtype=torch.float32)
    offsets = unique_offsets.repeat_interleave(args.copies_per_offset)
    num_envs = offsets.numel()

    cfg = PickPlaceEnvCfg()
    cfg.curriculum_phase = 1
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = 2026
    cfg.episode_length_s = 40.0

    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee_idx = robot.data.body_names.index("wrist_3_link")
    ee_jacobian_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
    arm_joint_ids = list(range(6))
    finger_idx = robot.data.joint_names.index("finger_joint")

    desired_quat_w = torch.tensor(
        DESIRED_WRIST_QUAT, device=device, dtype=torch.float32
    ).repeat(num_envs, 1)
    actions = torch.zeros((num_envs, 7), device=device)

    # Let the objects settle before fixing the per-environment grasp targets.
    for _ in range(20):
        actions.zero_()
        actions[:, 6] = 1.0
        env.step(actions)

    object_start_w = obj.data.root_pos_w.clone()
    grasp_target_w = object_start_w.clone()
    grasp_target_w[:, 2] += offsets
    baseline_z = object_start_w[:, 2].clone()
    max_z = baseline_z.clone()
    reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)

    total_steps = args.approach_steps + args.close_steps + args.lift_steps
    joint_target = robot.data.joint_pos[:, arm_joint_ids].clone()
    lift_start = args.approach_steps + args.close_steps
    for step in range(total_steps):
        lifting = step >= lift_start
        target_pos_w = grasp_target_w.clone()
        if lifting:
            target_pos_w[:, 2] += args.lift_distance

        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        if step % args.ik_interval == 0 or step == lift_start:
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobian_idx, :, arm_joint_ids
            ]
            ee_pose_w = robot.data.body_pose_w[:, ee_idx]
            pose_error = torch.cat(
                (
                    target_pos_w - ee_pose_w[:, :3],
                    quaternion_rotation_error(desired_quat_w, ee_pose_w[:, 3:7]),
                ),
                dim=-1,
            )
            jacobian_t = jacobian.transpose(1, 2)
            damping = 0.05
            regularizer = torch.eye(6, device=device).unsqueeze(0) * damping**2
            delta_joint = jacobian_t @ torch.linalg.solve(
                jacobian @ jacobian_t + regularizer, pose_error.unsqueeze(-1)
            )
            joint_target = joint_pos + delta_joint.squeeze(-1)

        # Production arm actions are relative joint targets with a 0.07 rad scale.
        actions[:, :6] = torch.clamp((joint_target - joint_pos) / 0.07, -1.0, 1.0)
        actions[:, 6] = -1.0 if step >= args.approach_steps else 1.0
        _, _, dones, _ = env.step(actions)

        reset_seen |= dones.bool()
        max_z = torch.maximum(max_z, obj.data.root_pos_w[:, 2])
        if step % 20 == 0 or step == total_steps - 1:
            stage = "lift" if lifting else ("close" if step >= args.approach_steps else "approach")
            print(f"  step {step + 1:3d}/{total_steps} ({stage})", flush=True)

    ee_pose_w = robot.data.body_pose_w[:, ee_idx]
    final_target_w = grasp_target_w.clone()
    final_target_w[:, 2] += args.lift_distance
    final_pos_error = torch.norm(ee_pose_w[:, :3] - final_target_w, dim=-1)
    final_finger = robot.data.joint_pos[:, finger_idx]
    lift = max_z - baseline_z
    final_obj_z = obj.data.root_pos_w[:, 2] - baseline_z

    print("\nGrasp geometry sweep (median across randomized trials)")
    print("offset_m  wrist_err_m  finger_rad  max_lift_m  final_dz_m  reset_rate")
    for index, offset in enumerate(args.offsets):
        start = index * args.copies_per_offset
        end = start + args.copies_per_offset
        print(
            f"{offset:8.3f}  {final_pos_error[start:end].median().item():11.3f}  "
            f"{final_finger[start:end].median().item():10.3f}  "
            f"{lift[start:end].median().item():10.3f}  "
            f"{final_obj_z[start:end].median().item():10.3f}  "
            f"{reset_seen[start:end].float().mean().item():10.1%}"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
