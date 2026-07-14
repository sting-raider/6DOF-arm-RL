"""
Isaac Lab training script for UR10e pick-and-place using RSL-RL (PPO).

Usage:
    # Train Phase 0 (REACH) - 128 envs headless:
    python scripts/train_isaac.py --phase 0 --num_envs 128 --headless

    # Train Phase 2 (PLACE) - 128 envs:
    python scripts/train_isaac.py --phase 2 --num_envs 128 --headless

    # Train with viewer (requires display):
    python scripts/train_isaac.py --phase 0 --num_envs 32 --enable_cameras

    # Watch live progress in another terminal:
    tail -f logs/isaac/training_live.log
"""

import argparse
import os
import sys
import time

# Insert project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

argparser = argparse.ArgumentParser()
argparser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                       help="Curriculum phase: 0=REACH, 1=GRASP, 2=PLACE")
argparser.add_argument("--num_envs", type=int, default=128,
                       help="Number of parallel environments")
argparser.add_argument("--max_iterations", type=int, default=1000,
                       help="Maximum training iterations")
argparser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
argparser.add_argument(
    "--warm_start",
    type=str,
    default=None,
    help=(
        "Initialize only the actor and its observation normalizer from another "
        "phase; critic, optimizer, and iteration count start fresh"
    ),
)
argparser.add_argument(
    "--output_model",
    type=str,
    default=None,
    help="Final checkpoint path (defaults to models/isaac/phase_N/model.pt)",
)
argparser.add_argument("--seed", type=int, default=42, help="Environment random seed")

AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args()
if args_cli.checkpoint and args_cli.warm_start:
    argparser.error("--checkpoint and --warm_start are mutually exclusive")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

from isaac_env.env_cfg import PickPlaceEnvCfg
from isaac_env.mdp import DESIRED_WRIST_QUAT, GRASP_TARGET_OFFSET

# Config
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def main():
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"\n{'='*60}")
    print(f"  Isaac Lab — UR10e Pick-and-Place")
    print(f"  Phase {args_cli.phase}: {phase_names[args_cli.phase]}")
    print(f"  Envs: {args_cli.num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Device: {args_cli.device}")
    print(f"{'='*60}\n")

    # ── Create environment ───────────────────────────────────────────────
    env_cfg = PickPlaceEnvCfg()
    env_cfg.curriculum_phase = args_cli.phase
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.seed = args_cli.seed

    raw_env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── Wrap for RSL-RL ──────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)

    # ── PPO config ───────────────────────────────────────────────────────
    log_root = os.path.join("logs", "isaac", f"phase_{args_cli.phase}")
    os.makedirs(log_root, exist_ok=True)
    # Numbered run dirs — highest number = latest
    existing = [d for d in os.listdir(log_root) if d.startswith("run_") and os.path.isdir(os.path.join(log_root, d))]
    run_num = max([int(d.split("_")[1]) for d in existing] + [0]) + 1
    log_dir = os.path.join(log_root, f"run_{run_num}")
    os.makedirs(log_dir, exist_ok=True)

    ppo_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 2e-4,   # balanced for stable progress
            "gamma": 0.99,           # moderate horizon for shaping + terminal
            "lam": 0.95,
            "clip_param": 0.2,       # standard for stable updates at scale
            "value_loss_coef": 1.0,
            "desired_kl": 0.01,
            "entropy_coef": 0.003,   # moderate — enough exploration for tight threshold
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "rnd_cfg": None,
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "max_iterations": args_cli.max_iterations,
            "experiment_name": f"ur10e_pick_place_phase_{args_cli.phase}",
            "log_dir": log_dir,
            "resume": args_cli.checkpoint is not None,
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "multi_gpu": {
            "enabled": False,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },
    }

    device = args_cli.device if args_cli.device else "cuda:0"
    print(f"  PPO config:")
    print_dict(ppo_cfg)
    print()

    # ── Create runner ────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=device)

    # ── Scaled-down/near-zero critic initialization ──────────────────────
    # PyTorch default init gives critic output ≈ ±10. With bootstrapping
    # (time_out=True), this amplifies immediately: V≈10 → returns≈10×150 →
    # value_loss starts at 10M. Near-zero initialization makes V(s)≈0 so returns=reward≈0.1
    runner.alg.policy.critic[-1].weight.data.mul_(0.01)
    runner.alg.policy.critic[-1].bias.data.zero_()
    print("  Near-zero initialized critic output layer (prevents bootstrap amplification)")



    if args_cli.checkpoint:
        print(f"  Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    elif args_cli.warm_start:
        print(f"  Warm-starting actor from: {args_cli.warm_start}")
        checkpoint = torch.load(args_cli.warm_start, map_location=device, weights_only=False)
        source_state = checkpoint["model_state_dict"]
        transferable = {
            key: value
            for key, value in source_state.items()
            if key.startswith("actor.") or key.startswith("actor_obs_normalizer.")
        }
        unexpected = sorted(set(transferable) - set(runner.alg.policy.state_dict()))
        if unexpected:
            raise RuntimeError(f"Unexpected warm-start parameters: {unexpected}")
        # RSL-RL's ActorCritic overrides PyTorch's return type with a boolean,
        # so validate keys above and then perform the partial load.
        runner.alg.policy.load_state_dict(transferable, strict=False)
        if args_cli.phase > 0:
            # Phase 0 keeps the gripper open, so these two dimensions have an
            # epsilon-sized variance in its normalizer.  A first close command
            # would otherwise become a ~200-sigma input and destroy the reach
            # behavior we are trying to transfer.
            normalizer = runner.alg.policy.actor_obs_normalizer
            normalizer._mean[0, 19] = 0.5   # scaled finger position: [0, 1]
            normalizer._var[0, 19] = 0.25
            normalizer._std[0, 19] = 0.5
            normalizer._mean[0, 33] = 0.0   # previous gripper action: [-1, 1]
            normalizer._var[0, 33] = 1.0
            normalizer._std[0, 33] = 1.0
            runner.alg.policy.actor[0].weight.data[:, [19, 33]] = 0.0
        print(
            f"  Transferred {len(transferable)} actor/normalizer tensors; "
            "critic and optimizer remain freshly initialized"
        )

    # ── Auto-close gripper for Phase 1 ───────────────────────────────────
    if args_cli.phase in (0, 1):
        robot = raw_env.scene["robot"]
        ee_idx = robot.data.body_names.index("wrist_3_link")
        obj = raw_env.scene["object"]
        gripper_closed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        _orig_step = env.step

        def _step_with_scripted_gripper(actions):
            safe_actions = actions.clone()
            if args_cli.phase == 0:
                # Gripper exploration is irrelevant to reaching and was the main
                # source of articulation instability in the old Phase 0 run.
                safe_actions[:, 6] = 1.0
            else:
                ee_pos = robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
                ee_quat = robot.data.body_quat_w[:, ee_idx]
                obj_pos = obj.data.root_pos_w - raw_env.scene.env_origins
                target_offset = torch.tensor(
                    GRASP_TARGET_OFFSET, device=device, dtype=ee_pos.dtype
                )
                desired_quat = torch.tensor(
                    DESIRED_WRIST_QUAT, device=device, dtype=ee_quat.dtype
                )
                position_ready = torch.norm(
                    ee_pos - (obj_pos + target_offset), dim=1
                ) < 0.06
                orientation_error = 2.0 * torch.acos(
                    torch.sum(ee_quat * desired_quat, dim=1).abs().clamp(max=1.0)
                )
                gripper_closed.logical_or_(
                    position_ready & (orientation_error < 0.436332313)
                )
                safe_actions[:, 6] = torch.where(gripper_closed, -1.0, 1.0)
            obs, rew, dones, info = _orig_step(safe_actions)
            gripper_closed[dones.bool()] = False
            return obs, rew, dones, info

        env.step = _step_with_scripted_gripper
        if args_cli.phase == 0:
            print("  Scripted gripper: held open during reach training")
        else:
            print("  Scripted gripper: closes through actuator dynamics at <6cm")

    # ── Curriculum: start with closer objects, expand over training ─────
    # ── Train ────────────────────────────────────────────────────────────
    print("  Starting training...")
    start_time = time.time()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    total_time = time.time() - start_time
    print(f"\n  Training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = args_cli.output_model or os.path.join(
        "models", "isaac", f"phase_{args_cli.phase}", "model.pt"
    )
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    runner.save(model_path)
    print(f"  Model saved to: {model_path}")

    env.close()
    print("  Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
