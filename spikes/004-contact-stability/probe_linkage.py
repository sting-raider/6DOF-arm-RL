#!/usr/bin/env python3
"""Compare Robotiq closure contacts at one sample per PhysX step."""

import argparse
import os
import sys

import torch

if os.name == "nt":
    import h5py  # noqa: F401 -- preload Isaac Sim's matching HDF5 DLL

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--close_steps", type=int, default=120)
parser.add_argument("--close_target", type=float, default=0.78)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from isaac_env.env_cfg import PickPlaceEnvCfg
from isaac_env.mdp import DESIRED_WRIST_QUAT


SCENARIOS = ("free", "object", "table")


def main() -> None:
    device = args.device or "cuda:0"
    num_envs = len(SCENARIOS)
    cfg = PickPlaceEnvCfg()
    cfg.curriculum_phase = 1
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = 2026
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.episode_length_s = 20.0
    cfg.scene.object.spawn.size = (0.04, 0.04, 0.10)
    cfg.scene.object.spawn.rigid_props.disable_gravity = True
    cfg.scene.contact_plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ContactPlate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 1.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    cfg.actions.gripper_action.close_command_expr = {
        "finger_joint": args.close_target
    }

    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    contact_plate = env.scene["contact_plate"]
    wrist_id = robot.data.body_names.index("wrist_3_link")
    gripper_joint_ids = list(range(6, len(robot.data.joint_names)))
    gripper_joint_names = [robot.data.joint_names[i] for i in gripper_joint_ids]
    finger_idx = robot.data.joint_names.index("finger_joint")
    actions = torch.zeros((num_envs, 7), device=device)
    arm_limits = robot.data.joint_pos_limits[0, :6].detach().cpu()
    arm_soft_limits = robot.data.soft_joint_pos_limits[0, :6].detach().cpu()
    gripper_limits = robot.data.joint_pos_limits[0, gripper_joint_ids].detach().cpu()

    def physics_step(step_actions: torch.Tensor) -> None:
        """Advance exactly one PhysX step without RL reward/observation overhead."""
        env.action_manager.process_action(step_actions)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

    for _ in range(20):
        actions.zero_()
        actions[:, 6] = 1.0
        physics_step(actions)

    wrist_positions = robot.data.body_pos_w[:, wrist_id]
    wrist_orientations = robot.data.body_quat_w[:, wrist_id]
    desired_orientation = torch.tensor(
        DESIRED_WRIST_QUAT, device=device, dtype=wrist_orientations.dtype
    ).repeat(num_envs, 1)
    fixture_orientations = quat_mul(
        wrist_orientations, quat_inv(desired_orientation)
    )

    # Reproduce the production relative geometry in the current wrist frame.
    # During a nominal close, the tall object center is 11 cm below the wrist;
    # the tabletop center is 17 cm below it (1 cm below its top surface).
    object_offset_world = torch.tensor(
        [0.0, 0.0, -0.11], device=device, dtype=wrist_positions.dtype
    ).repeat(num_envs, 1)
    plate_offset_world = torch.tensor(
        [0.0, 0.0, -0.17], device=device, dtype=wrist_positions.dtype
    ).repeat(num_envs, 1)
    object_offset_local = quat_apply_inverse(
        desired_orientation, object_offset_world
    )
    plate_offset_local = quat_apply_inverse(
        desired_orientation, plate_offset_world
    )

    # Free closure: move the object away. Object contact: use the production
    # wrist/object transform. Table contact: use an isolated kinematic plate at
    # the production wrist/table transform while keeping the object away.
    object_state = obj.data.root_state_w.clone()
    object_state[0, 0] += 0.30
    object_state[1, :3] = wrist_positions[1] + quat_apply(
        wrist_orientations[1:2], object_offset_local[1:2]
    )[0]
    object_state[1, 3:7] = fixture_orientations[1]
    object_state[2, 0] += 0.30
    object_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(object_state)

    plate_state = contact_plate.data.root_state_w.clone()
    plate_state[2, :3] = wrist_positions[2] + quat_apply(
        wrist_orientations[2:3], plate_offset_local[2:3]
    )[0]
    plate_state[2, 3:7] = fixture_orientations[2]
    plate_state[:, 7:] = 0.0
    contact_plate.write_root_state_to_sim(plate_state)

    for _ in range(10):
        actions.zero_()
        actions[:, 6] = 1.0
        physics_step(actions)

    peak_positions = torch.zeros(
        (num_envs, len(gripper_joint_ids)), device=device
    )
    peak_velocities = torch.zeros_like(peak_positions)
    invalid_positions = [None] * num_envs
    invalid_velocities = [None] * num_envs
    invalid_steps = [None] * num_envs
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for step in range(args.close_steps):
        actions.zero_()
        actions[:, 6] = -1.0
        actions[finished, 6] = 1.0
        physics_step(actions)

        joint_positions = robot.data.joint_pos[:, gripper_joint_ids]
        joint_velocities = robot.data.joint_vel[:, gripper_joint_ids]
        peak_positions = torch.maximum(peak_positions, joint_positions.abs())
        peak_velocities = torch.maximum(peak_velocities, joint_velocities.abs())
        drive_position = robot.data.joint_pos[:, finger_idx]
        invalid_gripper = (~torch.isfinite(joint_positions)).any(dim=1) | (
            drive_position.abs() > 1.5
        )
        for env_index in range(num_envs):
            if finished[env_index] or not bool(invalid_gripper[env_index].item()):
                continue
            invalid_steps[env_index] = step
            invalid_positions[env_index] = joint_positions[env_index].detach().cpu()
            invalid_velocities[env_index] = joint_velocities[env_index].detach().cpu()
            finished[env_index] = True

        if (step + 1) % 30 == 0 or step == args.close_steps - 1:
            print(f"close_step={step + 1}/{args.close_steps}", flush=True)

    print(f"physics_dt={cfg.sim.dt:.6f}s decimation={cfg.decimation}")
    print("arm_joint_limits:")
    for joint_index, name in enumerate(robot.data.joint_names[:6]):
        print(
            f"  {name}=hard[{arm_limits[joint_index, 0].item():.6f},"
            f" {arm_limits[joint_index, 1].item():.6f}] "
            f"soft[{arm_soft_limits[joint_index, 0].item():.6f},"
            f" {arm_soft_limits[joint_index, 1].item():.6f}]"
        )
    print("gripper_joint_limits:")
    for joint_index, name in enumerate(gripper_joint_names):
        print(
            f"  {name}=[{gripper_limits[joint_index, 0].item():.6f},"
            f" {gripper_limits[joint_index, 1].item():.6f}]"
        )
    for env_index, scenario in enumerate(SCENARIOS):
        print(f"\nscenario={scenario}")
        print(f"invalid_gripper_step={invalid_steps[env_index]}")
        print(
            "joint,peak_abs_position,peak_abs_velocity,"
            "invalid_position,invalid_velocity"
        )
        for joint_index, name in enumerate(gripper_joint_names):
            invalid_position = (
                float("nan")
                if invalid_positions[env_index] is None
                else invalid_positions[env_index][joint_index].item()
            )
            invalid_velocity = (
                float("nan")
                if invalid_velocities[env_index] is None
                else invalid_velocities[env_index][joint_index].item()
            )
            print(
                f"{name},{peak_positions[env_index, joint_index].item():.6f},"
                f"{peak_velocities[env_index, joint_index].item():.6f},"
                f"{invalid_position:.6f},{invalid_velocity:.6f}"
            )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
