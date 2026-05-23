"""
Isaac Lab training script for UR10e pick-and-place using RSL-RL (PPO).

Usage:
    # Train Phase 0 (REACH) - 128 envs headless:
    python scripts/train_isaac.py --phase 0 --num_envs 128 --headless

    # Train Phase 2 (PLACE) - 128 envs:
    python scripts/train_isaac.py --phase 2 --num_envs 128 --headless

    # Train with viewer (requires display):
    python scripts/train_isaac.py --phase 0 --num_envs 32 --enable_cameras
"""

import argparse
import os
import time
from datetime import datetime

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

AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

from isaac_env.env_cfg import PickPlaceEnvCfg

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

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── Wrap for RSL-RL ──────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)

    # ── PPO config ───────────────────────────────────────────────────────
    log_root = os.path.join("logs", "isaac", f"phase_{args_cli.phase}")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    ppo_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "num_minibatches": 4,
            "update_per_rollout": 8,
            "batch_size": args_cli.num_envs * 24,  # num_steps_per_env = 24
            "minibatch_size": args_cli.num_envs * 24 // 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "policy_class_name": "ActorCritic",
            "num_steps_per_env": 24,
            "max_iterations": args_cli.max_iterations,
            "experiment_name": f"ur10e_pick_place_phase_{args_cli.phase}",
            "log_dir": log_dir,
            "save_interval": 50,
            "resume": args_cli.checkpoint is not None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
        },
    }

    device = args_cli.device if args_cli.device else "cuda:0"
    print(f"  PPO config:")
    print_dict(ppo_cfg)
    print()

    # ── Create runner ────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=device)

    if args_cli.checkpoint:
        print(f"  Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    # ── Train ────────────────────────────────────────────────────────────
    print("  Starting training...")
    start_time = time.time()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    total_time = time.time() - start_time
    print(f"\n  Training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save model ────────────────────────────────────────────────────────
    save_dir = os.path.join("models", "isaac", f"phase_{args_cli.phase}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(runner.alg.actor_critic.state_dict(),
               os.path.join(save_dir, "model.pt"))
    print(f"  Model: {save_dir}/model.pt")

    env.close()
    print("  Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
