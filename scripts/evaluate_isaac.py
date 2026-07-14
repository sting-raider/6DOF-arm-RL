#!/usr/bin/env python3
"""Isaac Lab evaluation script — runs N episodes and reports success metrics.

Uses obs_normalization=True for eval, loading both trained policy weights and
running observation normalizer statistics.
"""
import argparse, os, sys, time
import numpy as np
import torch

# The full Isaac Sim GUI loads an HDF5 DLL before Isaac Lab imports its optional
# dataset helpers on Windows.  Preloading h5py here ensures its bundled, matching
# HDF5 DLL is registered first and avoids a missing-procedure error in _errors.pyd.
if os.name == "nt":
    import h5py  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

argparser = argparse.ArgumentParser()
argparser.add_argument("--phase", type=int, required=True, choices=[0,1,2])
argparser.add_argument("--model", type=str, required=True, help="Path to model.pt")
argparser.add_argument("--num_envs", type=int, default=16)
argparser.add_argument("--episodes", type=int, default=20)
argparser.add_argument("--seed", type=int, default=42,
                       help="Environment random seed for reproducible benchmarks")
argparser.add_argument("--demo_layout", action="store_true",
                       help="Place the first four objects at distinct fixed locations")
argparser.add_argument("--realtime", action="store_true",
                       help="Pace policy steps to the configured environment step time")
argparser.add_argument(
    "--grasp_offset_z",
    type=float,
    default=None,
    help="Override the wrist height above the object center for geometry ablations",
)
argparser.add_argument(
    "--target_obs_mode",
    choices=["truth", "shuffled", "center", "noisy"],
    default="truth",
    help=(
        "Target values supplied to the policy: simulator truth, another env's "
        "target, or the workspace center. Success is always measured against truth."
    ),
)
argparser.add_argument(
    "--target_noise_std",
    type=float,
    default=0.0,
    help="Per-axis Gaussian target noise used by --target_obs_mode noisy",
)
argparser.add_argument(
    "--target_bias_xyz",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Fixed XYZ target bias used by --target_obs_mode noisy",
)
argparser.add_argument(
    "--scripted_retract",
    action="store_true",
    help=(
        "After the Phase 1 scripted close reaches 0.45 rad, ask the reach policy "
        "to move the wrist upward. This isolates physical grasp feasibility from "
        "whether PPO has learned the retract sequence."
    ),
)
argparser.add_argument(
    "--retract_delta_z",
    type=float,
    default=0.08,
    help="Upward wrist displacement requested by --scripted_retract (default: 0.08 m)",
)
argparser.add_argument(
    "--scripted_grasp_cycle",
    action="store_true",
    help=(
        "Run a deterministic Phase 1 reach-descend-close-retract state machine "
        "around the learned reach policy to test contact geometry."
    ),
)
argparser.add_argument(
    "--hybrid_phase1",
    action="store_true",
    help=(
        "Use the validated laptop-friendly Phase 1 controller: the learned "
        "reach policy plus deterministic pose-servo descend/close/retract. "
        "This also enables the tested starter-object contact preset."
    ),
)
argparser.add_argument(
    "--descent_delta_z",
    type=float,
    default=None,
    help="Downward displacement during --scripted_grasp_cycle (default: 0.08 m)",
)
argparser.add_argument(
    "--object_size",
    type=float,
    default=None,
    help="Cube edge length in meters for grasp-geometry ablations (default: 0.04)",
)
argparser.add_argument(
    "--object_height",
    type=float,
    default=None,
    help="Optional object height for rectangular-prism grasp controls",
)
argparser.add_argument(
    "--close_target",
    type=float,
    default=None,
    help="Robotiq drive-joint close target in radians (default: 0.65)",
)
argparser.add_argument(
    "--gripper_max_delta",
    type=float,
    default=0.01,
    help="Maximum gripper target change per control tick (default: 0.01 rad)",
)
argparser.add_argument(
    "--gripper_drive_damping",
    type=float,
    default=1.0,
    help="Drive-joint actuator damping used for contact ablations (default: 1.0)",
)
argparser.add_argument(
    "--gripper_finger_damping",
    type=float,
    default=0.05,
    help="Passive inner-finger actuator damping (default: 0.05)",
)
argparser.add_argument(
    "--gripper_effort_limit",
    type=float,
    default=10.0,
    help="Drive and inner-finger simulated effort limit (default: 10.0)",
)
argparser.add_argument(
    "--close_position_tolerance",
    type=float,
    default=None,
    help="Cartesian tolerance before scripted closure (default: 0.01 m)",
)
argparser.add_argument(
    "--close_lateral_tolerance",
    type=float,
    default=None,
    help=(
        "Optional XY-only close tolerance. When set, it replaces the spherical "
        "close gate together with --close_vertical_tolerance."
    ),
)
argparser.add_argument(
    "--close_vertical_tolerance",
    type=float,
    default=None,
    help="Z tolerance paired with --close_lateral_tolerance",
)
argparser.add_argument(
    "--object_friction",
    type=float,
    default=None,
    help="Static/dynamic object friction for contact ablations (default: 0.5)",
)
argparser.add_argument(
    "--retract_finger_threshold",
    type=float,
    default=None,
    help=(
        "Finger position required before scripted retract "
        "(default: 0.45 rad; --hybrid_phase1: 0.45 rad)"
    ),
)
argparser.add_argument(
    "--retract_step_limit",
    type=float,
    default=None,
    help=(
        "Maximum Cartesian position step per control tick while retracting "
        "(default: 0.01 m; --hybrid_phase1: 0.02 m)"
    ),
)
argparser.add_argument(
    "--max_grasp_attempts",
    type=int,
    default=None,
    help="Maximum close/lift attempts per episode (default: 1; --hybrid_phase1: 2)",
)
argparser.add_argument(
    "--retry_wait_steps",
    type=int,
    default=30,
    help="Control steps to assess a failed retract before retrying (default: 30)",
)
argparser.add_argument(
    "--grasp_target_bias_xy",
    type=float,
    nargs=2,
    default=None,
    metavar=("X", "Y"),
    help="Fixed XY wrist-target correction for contact alignment",
)
AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args()
if args_cli.target_noise_std < 0.0:
    argparser.error("--target_noise_std must be non-negative")
if args_cli.gripper_max_delta <= 0.0:
    argparser.error("--gripper_max_delta must be positive")
if args_cli.gripper_drive_damping < 0.0 or args_cli.gripper_finger_damping < 0.0:
    argparser.error("gripper damping values must be non-negative")
if args_cli.gripper_effort_limit <= 0.0:
    argparser.error("--gripper_effort_limit must be positive")

# Apply the physical-grasp preset before Isaac Sim starts. Explicit geometry
# flags remain available for ablations, while the one-switch path is stable and
# reproducible for normal laptop demos.
if args_cli.hybrid_phase1:
    args_cli.scripted_grasp_cycle = True
args_cli.descent_delta_z = (
    0.08 if args_cli.descent_delta_z is None else args_cli.descent_delta_z
)
args_cli.object_size = 0.04 if args_cli.object_size is None else args_cli.object_size
if args_cli.object_height is None and args_cli.hybrid_phase1:
    args_cli.object_height = 0.10
args_cli.close_target = (
    (0.78 if args_cli.hybrid_phase1 else 0.65)
    if args_cli.close_target is None
    else args_cli.close_target
)
args_cli.close_position_tolerance = (
    (0.035 if args_cli.hybrid_phase1 else 0.01)
    if args_cli.close_position_tolerance is None
    else args_cli.close_position_tolerance
)
if (args_cli.close_lateral_tolerance is None) != (
    args_cli.close_vertical_tolerance is None
):
    argparser.error(
        "--close_lateral_tolerance and --close_vertical_tolerance must be used together"
    )
args_cli.object_friction = (
    (1.0 if args_cli.hybrid_phase1 else 0.5)
    if args_cli.object_friction is None
    else args_cli.object_friction
)
args_cli.retract_finger_threshold = (
    0.45
    if args_cli.retract_finger_threshold is None
    else args_cli.retract_finger_threshold
)
args_cli.retract_step_limit = (
    (0.02 if args_cli.hybrid_phase1 else 0.01)
    if args_cli.retract_step_limit is None
    else args_cli.retract_step_limit
)
args_cli.max_grasp_attempts = (
    (2 if args_cli.hybrid_phase1 else 1)
    if args_cli.max_grasp_attempts is None
    else args_cli.max_grasp_attempts
)
args_cli.grasp_target_bias_xy = (
    ((0.0, 0.007) if args_cli.hybrid_phase1 else (0.0, 0.0))
    if args_cli.grasp_target_bias_xy is None
    else tuple(args_cli.grasp_target_bias_xy)
)
if args_cli.max_grasp_attempts < 1:
    argparser.error("--max_grasp_attempts must be at least 1")
if args_cli.retry_wait_steps < 1:
    argparser.error("--retry_wait_steps must be at least 1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.utils.math import compute_pose_error
from isaac_env.env_cfg import PickPlaceEnvCfg
import isaac_env.mdp as pick_mdp
from isaac_env.mdp import (
    BASKET_CENTER,
    DESIRED_WRIST_QUAT,
    LIFT_SUCCESS_DELTA,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

if args_cli.grasp_offset_z is not None:
    pick_mdp.GRASP_TARGET_OFFSET = (0.0, 0.0, args_cli.grasp_offset_z)
GRASP_TARGET_OFFSET = pick_mdp.GRASP_TARGET_OFFSET


OBS_SLICES = {
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


def target_conditioned_observations(
    obs, true_object_pos, mode, noise_std=0.0, bias_xyz=(0.0, 0.0, 0.0)
):
    """Return a cloned TensorDict with a controlled target supplied to the policy.

    Rewards and evaluation metrics continue to use simulator truth. Replacing all
    three object-dependent terms avoids leaking the real target through relative
    position or distance during ablation tests.
    """
    if mode == "truth":
        return obs

    policy_obs = obs.clone()
    policy_tensor = policy_obs["policy"]
    if mode == "shuffled":
        if true_object_pos.shape[0] < 2:
            raise ValueError("--target_obs_mode shuffled requires --num_envs >= 2")
        observed_object_pos = torch.roll(true_object_pos, shifts=1, dims=0)
    elif mode == "center":
        observed_object_pos = torch.tensor(
            [0.35, 0.0, 0.85], device=policy_tensor.device, dtype=policy_tensor.dtype
        ).expand_as(true_object_pos)
    elif mode == "noisy":
        bias = torch.tensor(
            bias_xyz, device=true_object_pos.device, dtype=true_object_pos.dtype
        )
        observed_object_pos = true_object_pos + bias
        if noise_std > 0.0:
            observed_object_pos = observed_object_pos + noise_std * torch.randn_like(
                true_object_pos
            )
    else:
        raise ValueError(f"Unsupported target observation mode: {mode}")

    ee_pos = policy_tensor[:, OBS_SLICES["ee_pos_scaled"]] * 1.5
    target_offset = torch.tensor(
        GRASP_TARGET_OFFSET, device=policy_tensor.device, dtype=policy_tensor.dtype
    )
    relative_pos = observed_object_pos + target_offset - ee_pos
    policy_tensor[:, OBS_SLICES["object_pos"]] = observed_object_pos
    policy_tensor[:, OBS_SLICES["relative_pos"]] = relative_pos
    policy_tensor[:, OBS_SLICES["distance"]] = torch.norm(
        relative_pos, dim=1, keepdim=True
    )
    return policy_obs


def retract_conditioned_observations(obs, active_mask, delta_z):
    """Request an upward wrist move through the policy's learned relative target.

    The absolute object observation remains physical truth. Only the relative
    target and its distance are changed, which keeps the fixed-height object
    channel in-distribution while reusing Phase 0's learned reaching behavior.
    """
    if not torch.any(active_mask):
        return obs

    policy_obs = obs.clone()
    policy_tensor = policy_obs["policy"]
    relative_pos = policy_tensor[:, OBS_SLICES["relative_pos"]]
    relative_pos[active_mask, 2] += delta_z
    policy_tensor[:, OBS_SLICES["distance"]] = torch.norm(
        relative_pos, dim=1, keepdim=True
    )
    return policy_obs


def main():
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"\n{'='*60}")
    print(f"  Evaluating Phase {args_cli.phase}: {phase_names[args_cli.phase]}")
    print(f"  Model: {args_cli.model}")
    print(f"  Envs: {args_cli.num_envs} | Episodes: {args_cli.episodes}")
    print(f"  Policy target observations: {args_cli.target_obs_mode}")
    if args_cli.target_obs_mode == "noisy":
        print(
            "  Target perturbation: "
            f"noise std={args_cli.target_noise_std:.3f} m, "
            f"bias={tuple(args_cli.target_bias_xyz)} m"
        )
    print(f"  Wrist grasp offset Z: {GRASP_TARGET_OFFSET[2]:.3f} m")
    object_height = (
        args_cli.object_height
        if args_cli.object_height is not None
        else args_cli.object_size
    )
    print(
        "  Object dimensions: "
        f"{args_cli.object_size:.3f} x {args_cli.object_size:.3f} x "
        f"{object_height:.3f} m"
    )
    print(f"  Gripper close target: {args_cli.close_target:.3f} rad")
    print(f"  Object contact friction: {args_cli.object_friction:.2f}")
    if args_cli.scripted_retract:
        if args_cli.phase != 1:
            raise ValueError("--scripted_retract is only valid with --phase 1")
        print(f"  Scripted retract: +{args_cli.retract_delta_z:.3f} m after closure")
    if args_cli.scripted_grasp_cycle:
        if args_cli.phase != 1:
            raise ValueError("--scripted_grasp_cycle is only valid with --phase 1")
        if args_cli.scripted_retract:
            raise ValueError(
                "Use either --scripted_grasp_cycle or --scripted_retract, not both"
            )
        print(
            "  Scripted grasp cycle: reach -> "
            f"descend {args_cli.descent_delta_z:.3f} m -> close -> retract"
        )
        print(
            "  Scripted close position tolerance: "
            f"{args_cli.close_position_tolerance:.3f} m"
        )
        if args_cli.close_lateral_tolerance is not None:
            print(
                "  Axis close gate: "
                f"XY<{args_cli.close_lateral_tolerance:.3f} m, "
                f"|Z|<{args_cli.close_vertical_tolerance:.3f} m"
            )
        print(
            "  Retract finger threshold: "
            f"{args_cli.retract_finger_threshold:.3f} rad"
        )
        print(f"  Retract position step limit: {args_cli.retract_step_limit:.3f} m")
        print(f"  Maximum grasp attempts: {args_cli.max_grasp_attempts}")
        if any(args_cli.grasp_target_bias_xy):
            print(
                "  Grasp target XY correction: "
                f"{tuple(args_cli.grasp_target_bias_xy)} m"
            )
        if args_cli.hybrid_phase1:
            print("  Hybrid Phase 1 preset: enabled")
    print(f"{'='*60}\n")

    env_cfg = PickPlaceEnvCfg()
    env_cfg.curriculum_phase = args_cli.phase
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.scene.object.spawn.size = (
        args_cli.object_size,
        args_cli.object_size,
        object_height,
    )
    env_cfg.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=args_cli.object_friction,
        dynamic_friction=args_cli.object_friction,
        restitution=0.0,
    )
    env_cfg.actions.gripper_action.close_command_expr = {
        "finger_joint": args_cli.close_target
    }
    env_cfg.actions.gripper_action.max_delta = args_cli.gripper_max_delta
    env_cfg.scene.robot.actuators["gripper_drive"].damping = (
        args_cli.gripper_drive_damping
    )
    env_cfg.scene.robot.actuators["gripper_finger"].damping = (
        args_cli.gripper_finger_damping
    )
    env_cfg.scene.robot.actuators["gripper_drive"].effort_limit_sim = (
        args_cli.gripper_effort_limit
    )
    env_cfg.scene.robot.actuators["gripper_finger"].effort_limit_sim = (
        args_cli.gripper_effort_limit
    )
    # Table top is z=0.81 m; use the actual resting cube centroid in metrics.
    object_rest_height = 0.81 + 0.5 * object_height
    reset_height = object_rest_height + 0.02
    env_cfg.events.reset_object.params["pose_range"]["z"] = (
        reset_height,
        reset_height,
    )
    if args_cli.scripted_grasp_cycle:
        # The diagnostic includes a deliberately slow close and needs more time
        # than the training episode to complete all four stages.
        env_cfg.episode_length_s = 20.0 + 10.0 * (
            args_cli.max_grasp_attempts - 1
        )
    if args_cli.demo_layout and args_cli.num_envs == 4:
        # Frame the 2x2 environment grid instead of focusing only env_0.
        env_cfg.viewer.eye = (7.5, 7.5, 6.0)
        env_cfg.viewer.lookat = (1.5, 1.5, 0.8)
        env_cfg.viewer.origin_type = "world"
    elif args_cli.num_envs == 1:
        # Frame env_0's tabletop workspace for physical-grasp inspection.
        env_cfg.viewer.eye = (1.35, 1.15, 1.30)
        env_cfg.viewer.lookat = (0.32, 0.0, 0.84)
        env_cfg.viewer.origin_type = "world"
    device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.device = device
    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)

    # Eval PPO config — architecture must match training, hyperparams are irrelevant
    ppo_cfg = {
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 8, "num_mini_batches": 8,
                      "learning_rate": 3e-4, "gamma": 0.995, "lam": 0.95,
                      "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.001,
                      "max_grad_norm": 1.0, "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "schedule": "adaptive"},
        "runner": {"class_name": "OnPolicyRunner", "max_iterations": 1,
                    "experiment_name": "eval", "log_dir": "/tmp/eval_log", "resume": False},
        "num_steps_per_env": 24, "save_interval": 9999,
        "multi_gpu": {"enabled": False},
        "policy": {"class_name": "ActorCritic", "init_noise_std": 1.0,
                   "noise_std_type": "scalar", "actor_obs_normalization": True,
                   "critic_obs_normalization": True,
                   "actor_hidden_dims": [256, 128, 64],
                   "critic_hidden_dims": [256, 128, 64], "activation": "elu"},
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }
    runner = OnPolicyRunner(env, ppo_cfg, log_dir="/tmp/eval_log", device=device)

    print(f"  Loading model from {args_cli.model}...")
    runner.load(args_cli.model)  # loads training normalizer stats
    print("  Model loaded.")
    inference_policy = runner.get_inference_policy(device=device)

    actor_normalizer = runner.alg.policy.actor_obs_normalizer
    normalizer_std = actor_normalizer._std[0].detach().cpu().numpy()
    print(
        "  Normalizer object std (x/y/z): "
        f"{normalizer_std[20]:.3g}, {normalizer_std[21]:.3g}, {normalizer_std[22]:.3g}"
    )
    if np.max(normalizer_std[:27]) > 100.0:
        print("  [WARNING] Checkpoint normalizer contains implausibly large observation scales.")

    # Cache scene references for real (unnormalized) position queries
    robot = raw_env.scene["robot"]
    ee_idx = robot.data.body_names.index("wrist_3_link")
    finger_idx = robot.data.joint_names.index("finger_joint")
    inner_finger_body_ids = (
        robot.data.body_names.index("left_inner_finger"),
        robot.data.body_names.index("right_inner_finger"),
    )
    obj = raw_env.scene["object"]

    if args_cli.demo_layout:
        if args_cli.num_envs != 4:
            raise ValueError("--demo_layout requires --num_envs 4")
        env_ids = torch.arange(4, device=device, dtype=torch.long)
        local_positions = torch.tensor(
            [
                [0.27, -0.13, reset_height],
                [0.43, -0.10, reset_height],
                [0.29, 0.13, reset_height],
                [0.44, 0.11, reset_height],
            ],
            device=device,
            dtype=torch.float32,
        )
        root_state = obj.data.root_state_w[env_ids].clone()
        root_state[:, :3] = local_positions + raw_env.scene.env_origins[env_ids]
        root_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )
        root_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
        print("  Four-object demo layout applied.")

    # ── Auto-close gripper for Phase 1+ eval ────────────────────────────
    if args_cli.phase in (0, 1):
        gripper_closed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        grasp_stage = torch.zeros(
            args_cli.num_envs, dtype=torch.long, device=device
        )
        grasp_stage_reached = torch.zeros(
            (args_cli.num_envs, 5), dtype=torch.bool, device=device
        )
        grasp_stage_reached[:, 0] = True
        grasp_stage_steps = torch.zeros(
            args_cli.num_envs, dtype=torch.long, device=device
        )
        grasp_attempts = torch.zeros(
            args_cli.num_envs, dtype=torch.long, device=device
        )
        grasp_target_correction = torch.zeros(
            (args_cli.num_envs, 3), dtype=torch.float32, device=device
        )
        grasp_target_correction[:, :2] = torch.tensor(
            args_cli.grasp_target_bias_xy,
            dtype=torch.float32,
            device=device,
        )
        initial_grasp_target_correction = grasp_target_correction.clone()
        retract_targets = torch.zeros(
            (args_cli.num_envs, 3), dtype=torch.float32, device=device
        )
        finger_midpoint_error_at_close = torch.full(
            (args_cli.num_envs, 3), torch.nan, dtype=torch.float32, device=device
        )
        _orig_step = env.step

        def _apply_cartesian_pose_servo(
            safe_actions, target_positions, mask, position_step_limit=0.01
        ):
            """Override arm actions with a small damped-least-squares pose step."""
            if not torch.any(mask):
                return
            # Fixed-base PhysX Jacobians omit the root body, hence ee_idx - 1.
            pose_jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_idx - 1, :6, :6
            ]
            current_positions = (
                robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
            )
            current_orientations = robot.data.body_quat_w[:, ee_idx]
            target_orientations = torch.tensor(
                DESIRED_WRIST_QUAT,
                device=device,
                dtype=current_orientations.dtype,
            ).expand_as(current_orientations)
            position_error, orientation_error = compute_pose_error(
                current_positions,
                current_orientations,
                target_positions,
                target_orientations,
                rot_error_type="axis_angle",
            )
            # Limit each Cartesian request so contact-rich retracts can move more
            # slowly than free-space descent.
            error_norm = torch.norm(position_error, dim=1, keepdim=True).clamp_min(1e-6)
            position_step = position_error * torch.clamp(
                position_step_limit / error_norm, max=1.0
            )
            orientation_norm = torch.norm(
                orientation_error, dim=1, keepdim=True
            ).clamp_min(1e-6)
            orientation_step = orientation_error * torch.clamp(
                0.05 / orientation_norm, max=1.0
            )
            pose_step = torch.cat((position_step, orientation_step), dim=1)
            jacobian_t = pose_jacobian.transpose(1, 2)
            damping = 0.03
            regularizer = (
                damping * damping
                * torch.eye(6, device=device, dtype=pose_jacobian.dtype)
                .unsqueeze(0)
                .expand(args_cli.num_envs, -1, -1)
            )
            joint_delta = torch.bmm(
                jacobian_t,
                torch.linalg.solve(
                    torch.bmm(pose_jacobian, jacobian_t) + regularizer,
                    pose_step.unsqueeze(-1),
                ),
            ).squeeze(-1)
            # arm_action has a 0.07-rad relative-joint scale in the environment.
            safe_actions[mask, :6] = torch.clamp(
                joint_delta[mask] / 0.07, min=-1.0, max=1.0
            )

        def _step_with_scripted_gripper(actions):
            safe_actions = actions.clone()
            if args_cli.phase == 0:
                safe_actions[:, 6] = 1.0
            else:
                ee_p = robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
                ee_q = robot.data.body_quat_w[:, ee_idx]
                obj_p = obj.data.root_pos_w - raw_env.scene.env_origins
                target_offset = torch.tensor(
                    GRASP_TARGET_OFFSET, device=device, dtype=ee_p.dtype
                )
                desired_quat = torch.tensor(
                    DESIRED_WRIST_QUAT, device=device, dtype=ee_q.dtype
                )
                position_error = torch.norm(ee_p - (obj_p + target_offset), dim=1)
                orientation_error = 2.0 * torch.acos(
                    torch.sum(ee_q * desired_quat, dim=1).abs().clamp(max=1.0)
                )
                orientation_ready = orientation_error < 0.436332313
                if args_cli.scripted_grasp_cycle:
                    stage_at_start = grasp_stage.clone()
                    begin_descent = (
                        (grasp_stage == 0)
                        & (position_error < 0.05)
                        & orientation_ready
                    )
                    grasp_stage[begin_descent] = 1
                    lowered_target = (
                        obj_p + target_offset + grasp_target_correction
                    )
                    lowered_target[:, 2] -= args_cli.descent_delta_z
                    lowered_error = torch.norm(
                        ee_p - lowered_target, dim=1
                    )
                    if args_cli.close_lateral_tolerance is None:
                        close_position_ready = (
                            lowered_error < args_cli.close_position_tolerance
                        )
                    else:
                        lowered_vector_error = ee_p - lowered_target
                        close_position_ready = (
                            torch.norm(lowered_vector_error[:, :2], dim=1)
                            < args_cli.close_lateral_tolerance
                        ) & (
                            torch.abs(lowered_vector_error[:, 2])
                            < args_cli.close_vertical_tolerance
                        )
                    begin_close = (
                        (grasp_stage == 1)
                        & close_position_ready
                        & (orientation_error < 0.261799388)
                    )
                    finger_midpoint = 0.5 * (
                        robot.data.body_pos_w[:, inner_finger_body_ids[0]]
                        + robot.data.body_pos_w[:, inner_finger_body_ids[1]]
                    ) - raw_env.scene.env_origins
                    finger_midpoint_error_at_close[begin_close] = (
                        finger_midpoint[begin_close] - obj_p[begin_close]
                    )
                    grasp_stage[begin_close] = 2
                    grasp_attempts[begin_close] += 1
                    begin_retract = (
                        (grasp_stage == 2)
                        & (
                            robot.data.joint_pos[:, finger_idx]
                            > args_cli.retract_finger_threshold
                        )
                    )
                    retract_targets[begin_retract] = ee_p[begin_retract]
                    retract_targets[begin_retract, 2] += args_cli.descent_delta_z
                    grasp_stage[begin_retract] = 3
                    failed_retract = (
                        (grasp_stage == 3)
                        & ~begin_retract
                        & (grasp_stage_steps >= args_cli.retry_wait_steps)
                        & (obj_p[:, 2] < object_rest_height + 0.03)
                        & (grasp_attempts < args_cli.max_grasp_attempts)
                    )
                    midpoint_correction = torch.clamp(
                        obj_p - finger_midpoint, min=-0.02, max=0.02
                    )
                    midpoint_correction[:, 2] = 0.0
                    grasp_target_correction[failed_retract] = (
                        midpoint_correction[failed_retract]
                    )
                    grasp_stage[failed_retract] = 4
                    corrected_pregrasp_target = (
                        obj_p + target_offset + grasp_target_correction
                    )
                    pregrasp_error = torch.norm(
                        ee_p - corrected_pregrasp_target, dim=1
                    )
                    begin_retry = (
                        (grasp_stage == 4)
                        & (pregrasp_error < 0.04)
                        & orientation_ready
                        & (robot.data.joint_pos[:, finger_idx] < 0.10)
                    )
                    grasp_stage[begin_retry] = 1
                    changed_stage = grasp_stage != stage_at_start
                    grasp_stage_steps[changed_stage] = 0
                    gripper_closed.copy_(
                        (grasp_stage == 2) | (grasp_stage == 3)
                    )
                    for stage_index in range(5):
                        grasp_stage_reached[:, stage_index].logical_or_(
                            grasp_stage >= stage_index
                        )
                else:
                    gripper_closed.logical_or_(
                        (position_error < 0.06) & orientation_ready
                    )
                safe_actions[:, 6] = torch.where(gripper_closed, -1.0, 1.0)
                if args_cli.scripted_grasp_cycle:
                    servo_targets = lowered_target.clone()
                    retracting = grasp_stage == 3
                    recovering = grasp_stage == 4
                    servo_targets[retracting] = retract_targets[retracting]
                    servo_targets[recovering] = corrected_pregrasp_target[recovering]
                    _apply_cartesian_pose_servo(
                        safe_actions,
                        servo_targets,
                        (grasp_stage == 1) | (grasp_stage == 2),
                    )
                    _apply_cartesian_pose_servo(
                        safe_actions,
                        servo_targets,
                        retracting,
                        position_step_limit=args_cli.retract_step_limit,
                    )
                    _apply_cartesian_pose_servo(
                        safe_actions,
                        servo_targets,
                        recovering,
                    )
            obs, rew, dones, info = _orig_step(safe_actions)
            grasp_stage_steps.add_(1)
            gripper_closed[dones.bool()] = False
            grasp_stage[dones.bool()] = 0
            grasp_stage_steps[dones.bool()] = 0
            grasp_attempts[dones.bool()] = 0
            grasp_target_correction[dones.bool()] = (
                initial_grasp_target_correction[dones.bool()]
            )
            return obs, rew, dones, info

        env.step = _step_with_scripted_gripper
        if args_cli.phase == 1:
            if args_cli.scripted_grasp_cycle:
                print("  Scripted physical grasp cycle enabled for eval\n")
            else:
                print("  Scripted gripper enabled for eval (actuator close at <6cm)\n")

    basket_center = torch.tensor(BASKET_CENTER, device=device, dtype=torch.float32)

    # Debug: print initial EE-to-object distance
    ee_pos0 = robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
    obj_pos0 = obj.data.root_pos_w[:, :3] - raw_env.scene.env_origins
    target_offset0 = torch.tensor(
        GRASP_TARGET_OFFSET, device=device, dtype=ee_pos0.dtype
    )
    ee_to_obj0 = torch.norm(ee_pos0 - (obj_pos0 + target_offset0), dim=1)
    print(f"  [DEBUG] Initial wrist-to-pregrasp distance: mean={ee_to_obj0.mean().item():.3f}m, "
          f"min={ee_to_obj0.min().item():.3f}m, max={ee_to_obj0.max().item():.3f}m\n")

    reach_thresholds = (0.08, 0.06, 0.05)
    reach_successes = {threshold: 0 for threshold in reach_thresholds}
    successes = {"grasp": 0, "place": 0, "total_eps": 0}
    rewards = []
    ep_lengths = []
    min_distances = []
    min_orientation_errors = []
    grasp_ready_episodes = 0
    max_lift_heights = []
    max_gripper_positions = []
    num_envs = args_cli.num_envs
    retract_activated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    gripper_joint_ids = list(range(6, len(robot.data.joint_names)))
    gripper_joint_names = [robot.data.joint_names[i] for i in gripper_joint_ids]
    linkage_peak_samples = []
    finger_separation_min_samples = []
    finger_separation_max_samples = []
    termination_names = tuple(raw_env.termination_manager.active_terms)
    termination_counts = {name: 0 for name in termination_names}

    episodes_to_run = args_cli.episodes
    episodes_done = 0
    obs = env.get_observations()

    ep_reward = torch.zeros(num_envs, device=device)
    ep_steps = torch.zeros(num_envs, device=device)
    ep_min_distance = torch.full((num_envs,), torch.inf, device=device)
    ep_min_orientation_error = torch.full((num_envs,), torch.inf, device=device)
    ep_grasp_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_max_lift_height = torch.full((num_envs,), -torch.inf, device=device)
    ep_max_gripper_position = torch.full((num_envs,), -torch.inf, device=device)
    ep_linkage_peak = torch.zeros(
        (num_envs, len(gripper_joint_ids)), device=device
    )
    ep_min_finger_separation = torch.full(
        (num_envs,), torch.inf, device=device
    )
    ep_max_finger_separation = torch.full(
        (num_envs,), -torch.inf, device=device
    )
    ep_grasp_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_place_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    print(f"  Running {episodes_to_run} episodes across {num_envs} envs...")
    start = time.time()

    while episodes_done < episodes_to_run:
        step_started = time.perf_counter()
        true_object_pos_for_policy = obj.data.root_pos_w[:, :3] - raw_env.scene.env_origins
        policy_obs = target_conditioned_observations(
            obs,
            true_object_pos_for_policy,
            args_cli.target_obs_mode,
            args_cli.target_noise_std,
            args_cli.target_bias_xyz,
        )
        if args_cli.scripted_retract:
            retract_now = gripper_closed & (robot.data.joint_pos[:, finger_idx] > 0.45)
            retract_activated.logical_or_(retract_now)
            policy_obs = retract_conditioned_observations(
                policy_obs, retract_now, args_cli.retract_delta_z
            )
        with torch.no_grad():
            actions = inference_policy(policy_obs)
        obs, rew, dones, infos = env.step(actions)
        done_now = dones > 0.5
        if torch.any(done_now):
            for term_name in termination_names:
                term_values = raw_env.termination_manager.get_term(term_name)
                termination_counts[term_name] += int(
                    (term_values & done_now).sum().item()
                )
        if args_cli.realtime:
            step_budget = raw_env.step_dt
            time.sleep(max(0.0, step_budget - (time.perf_counter() - step_started)))

        ep_reward += rew
        ep_steps += 1

        # Get REAL (unnormalized) positions from the Isaac Lab scene
        ee_pos = robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
        obj_pos = obj.data.root_pos_w[:, :3] - raw_env.scene.env_origins
        
        # Unify gripper state to finger_joint position
        grip_state = robot.data.joint_pos[:, finger_idx]

        obj_height = obj_pos[:, 2]
        lift_height = obj_height - object_rest_height
        obj_to_basket = torch.norm(obj_pos - basket_center, dim=1)
        target_offset = torch.tensor(
            GRASP_TARGET_OFFSET, device=device, dtype=ee_pos.dtype
        )
        grasp_target = obj_pos + target_offset
        ee_to_obj = torch.norm(ee_pos - grasp_target, dim=1)
        ee_quat = robot.data.body_quat_w[:, ee_idx]
        desired_quat = torch.tensor(
            DESIRED_WRIST_QUAT, device=device, dtype=ee_quat.dtype
        )
        orientation_error = 2.0 * torch.acos(
            torch.sum(ee_quat * desired_quat, dim=1).abs().clamp(max=1.0)
        )

        ep_min_distance = torch.minimum(ep_min_distance, ee_to_obj)
        ep_min_orientation_error = torch.minimum(
            ep_min_orientation_error, orientation_error
        )
        ep_grasp_ready = ep_grasp_ready | (
            (ee_to_obj < 0.06) & (orientation_error < 0.436332313)
        )
        ep_max_lift_height = torch.maximum(ep_max_lift_height, lift_height)
        ep_max_gripper_position = torch.maximum(ep_max_gripper_position, grip_state)
        ep_linkage_peak = torch.maximum(
            ep_linkage_peak,
            robot.data.joint_pos[:, gripper_joint_ids].abs(),
        )
        finger_separation = torch.norm(
            robot.data.body_pos_w[:, inner_finger_body_ids[0]]
            - robot.data.body_pos_w[:, inner_finger_body_ids[1]],
            dim=1,
        )
        ep_min_finger_separation = torch.minimum(
            ep_min_finger_separation, finger_separation
        )
        ep_max_finger_separation = torch.maximum(
            ep_max_finger_separation, finger_separation
        )
        ep_grasp_success = ep_grasp_success | (
            (grip_state > 0.10)
            & (ee_to_obj < 0.08)
            & (lift_height > LIFT_SUCCESS_DELTA)
        )
        ep_place_success = ep_place_success | (obj_to_basket < 0.10)

        done_mask = dones > 0.5
        for i in range(num_envs):
            if done_mask[i]:
                rewards.append(ep_reward[i].item())
                ep_lengths.append(ep_steps[i].item())
                episode_min_distance = ep_min_distance[i].item()
                min_distances.append(episode_min_distance)
                min_orientation_errors.append(ep_min_orientation_error[i].item())
                if ep_grasp_ready[i]:
                    grasp_ready_episodes += 1
                max_lift_heights.append(ep_max_lift_height[i].item())
                max_gripper_positions.append(ep_max_gripper_position[i].item())
                linkage_peak_samples.append(ep_linkage_peak[i].detach().cpu().numpy())
                finger_separation_min_samples.append(
                    ep_min_finger_separation[i].item()
                )
                finger_separation_max_samples.append(
                    ep_max_finger_separation[i].item()
                )
                successes["total_eps"] += 1
                for threshold in reach_thresholds:
                    if episode_min_distance < threshold:
                        reach_successes[threshold] += 1
                if ep_grasp_success[i]: successes["grasp"] += 1
                if ep_place_success[i]: successes["place"] += 1
                episodes_done += 1

                ep_reward[i] = 0
                ep_steps[i] = 0
                ep_min_distance[i] = torch.inf
                ep_min_orientation_error[i] = torch.inf
                ep_grasp_ready[i] = False
                ep_max_lift_height[i] = -torch.inf
                ep_max_gripper_position[i] = -torch.inf
                ep_linkage_peak[i] = 0.0
                ep_min_finger_separation[i] = torch.inf
                ep_max_finger_separation[i] = -torch.inf
                ep_grasp_success[i] = False
                ep_place_success[i] = False

                if episodes_done % 10 == 0 or episodes_done == episodes_to_run:
                    elapsed = time.time() - start
                    avg_r = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                    pct = 100 * reach_successes[0.08] / max(1, successes['total_eps'])
                    print(f"  [{elapsed:5.1f}s] Ep {episodes_done:3d}/{episodes_to_run} | "
                          f"AvgR: {avg_r:7.1f} | Reach@8cm: "
                          f"{reach_successes[0.08]}/{successes['total_eps']} ({pct:.0f}%)")

                if episodes_done >= episodes_to_run:
                    break

    total_time = time.time() - start

    phase = args_cli.phase
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — Phase {phase}: {phase_names[phase]}")
    print(f"  Model: {args_cli.model}")
    print(f"  Duration: {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"  Episodes:          {successes['total_eps']}")
    print(f"  Mean reward:       {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Episode length:     {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f}")
    for threshold in reach_thresholds:
        label = int(threshold * 100)
        count = reach_successes[threshold]
        print(f"  Reach success @{label}cm: {count}/{successes['total_eps']} "
              f"({100*count/max(1,successes['total_eps']):.0f}%)")
    p10, p50, p90 = np.percentile(min_distances, [10, 50, 90])
    print(f"  Min distance p10/50/90: {p10:.3f} / {p50:.3f} / {p90:.3f} m")
    a10, a50, a90 = np.degrees(np.percentile(min_orientation_errors, [10, 50, 90]))
    print(f"  Min orientation error p10/50/90: {a10:.1f} / {a50:.1f} / {a90:.1f} deg")
    print(
        f"  Grasp-ready pose:   {grasp_ready_episodes}/{successes['total_eps']} "
        f"({100*grasp_ready_episodes/max(1,successes['total_eps']):.0f}%)"
    )
    print(
        "  Episode endings:    "
        + ", ".join(
            f"{name}={count}" for name, count in termination_counts.items()
        )
    )
    if phase >= 1:
        print(f"  Grasp success:      {successes['grasp']}/{successes['total_eps']} "
              f"({100*successes['grasp']/max(1,successes['total_eps']):.0f}%)")
        lift_p10, lift_p50, lift_p90 = np.percentile(max_lift_heights, [10, 50, 90])
        print(
            "  Max lift p10/50/90: "
            f"{lift_p10:.3f} / {lift_p50:.3f} / {lift_p90:.3f} m"
        )
        grip_p10, grip_p50, grip_p90 = np.percentile(max_gripper_positions, [10, 50, 90])
        print(
            "  Max finger position p10/50/90: "
            f"{grip_p10:.3f} / {grip_p50:.3f} / {grip_p90:.3f} rad"
        )
        linkage_medians = np.median(np.asarray(linkage_peak_samples), axis=0)
        print("  Linkage peak |position| medians:")
        for joint_name, peak in zip(gripper_joint_names, linkage_medians):
            print(f"    {joint_name:38s} {peak:.3f} rad")
        print(
            "  Inner-finger body separation median min/max: "
            f"{np.median(finger_separation_min_samples):.3f} / "
            f"{np.median(finger_separation_max_samples):.3f} m"
        )
        if args_cli.scripted_retract:
            print(
                "  Retract activated:  "
                f"{int(retract_activated.sum().item())}/{num_envs} active env slots"
            )
        if args_cli.scripted_grasp_cycle:
            stage_labels = ("approach", "descend", "close", "retract", "recover")
            stage_counts = grasp_stage_reached.sum(dim=0).detach().cpu().tolist()
            print(
                "  Grasp-cycle stages: "
                + ", ".join(
                    f"{label}={count}/{num_envs}"
                    for label, count in zip(stage_labels, stage_counts)
                )
            )
            measured = torch.isfinite(finger_midpoint_error_at_close).all(dim=1)
            if torch.any(measured):
                midpoint_error_median = (
                    finger_midpoint_error_at_close[measured]
                    .median(dim=0)
                    .values.detach()
                    .cpu()
                    .numpy()
                )
                print(
                    "  Finger-midpoint minus cube at close (median): "
                    f"dx={midpoint_error_median[0]:+.3f}, "
                    f"dy={midpoint_error_median[1]:+.3f}, "
                    f"dz={midpoint_error_median[2]:+.3f} m"
                )
    if phase >= 2:
        print(f"  Place success:      {successes['place']}/{successes['total_eps']} "
              f"({100*successes['place']/max(1,successes['total_eps']):.0f}%)")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
