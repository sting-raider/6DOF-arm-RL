#!/usr/bin/env python3
"""Isaac Lab evaluation script — runs N episodes and reports success metrics.

Usage:
    python scripts/evaluate_isaac.py --phase 0 --model models/isaac/phase_0/model.pt
    python scripts/evaluate_isaac.py --phase 2 --model models/isaac/phase_2/model.pt --episodes 20
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

argparser = argparse.ArgumentParser()
argparser.add_argument("--phase", type=int, required=True, choices=[0,1,2])
argparser.add_argument("--model", type=str, required=True, help="Path to model.pt")
argparser.add_argument("--num_envs", type=int, default=1)
argparser.add_argument("--episodes", type=int, default=10)
AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Must happen after AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from isaac_env.env_cfg import PickPlaceEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"\n{'='*60}")
    print(f"  Evaluating Phase {args_cli.phase}: {phase_names[args_cli.phase]}")
    print(f"  Model: {args_cli.model}")
    print(f"  Episodes: {args_cli.episodes}")
    print(f"{'='*60}\n")

    # Create env
    env_cfg = PickPlaceEnvCfg()
    env_cfg.curriculum_phase = args_cli.phase
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load model
    ppo_cfg = {
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 8, "num_mini_batches": 4,
                      "learning_rate": 3e-4, "gamma": 0.99, "lam": 0.95,
                      "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.01,
                      "max_grad_norm": 1.0},
        "runner": {"class_name": "OnPolicyRunner", "max_iterations": 1,
                    "experiment_name": "eval", "log_dir": "/tmp/eval_log", "resume": False},
        "num_steps_per_env": 24, "save_interval": 9999,
        "multi_gpu": {"enabled": False},
        "actor": {"class_name": "rsl_rl.models.mlp_model.MLPModel",
                  "hidden_dims": [256, 128, 64], "activation": "elu",
                  "obs_normalization": False,
                  "distribution_cfg": {"class_name": "rsl_rl.modules.distribution.GaussianDistribution"}},
        "critic": {"class_name": "rsl_rl.models.mlp_model.MLPModel",
                   "hidden_dims": [256, 128, 64], "activation": "elu",
                   "obs_normalization": False},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    }
    device = args_cli.device if args_cli.device else "cuda:0"
    runner = OnPolicyRunner(env, ppo_cfg, log_dir="/tmp/eval_log", device=device)

    # Load the trained model
    print(f"  Loading model from {args_cli.model}...")
    runner.load(args_cli.model)
    print("  Model loaded.\n")

    # Evaluation loop
    successes = {"reach": 0, "grasp": 0, "place": 0}
    rewards = []
    ep_lengths = []
    obs = env.get_observations()

    print(f"  Running {args_cli.episodes} episodes...")
    for ep in range(1, args_cli.episodes + 1):
        ep_reward = 0.0
        ep_steps = 0
        ep_reach = False
        ep_grasp = False
        ep_place = False

        for step in range(500):  # max episode length
            actions = runner.alg.actor_critic.act_inference(obs)
            obs, rew, dones, infos = env.step(actions)
            ep_reward += rew.item()
            ep_steps += 1

            # Check success flags from info
            if "episode" in infos or dones.item():
                # Extract success flags if available
                pass

            if dones.item():
                break

        # Check success at end of episode
        ee_pos = obs[0, 0:3].cpu().numpy()
        obj_pos = obs[0, 3:6].cpu().numpy()
        dist = np.linalg.norm(ee_pos - obj_pos)
        if dist < 0.05:
            ep_reach = True
            successes["reach"] += 1

        rewards.append(ep_reward)
        ep_lengths.append(ep_steps)

        if ep % 5 == 0 or ep == args_cli.episodes:
            avg_r = np.mean(rewards[-5:]) if rewards else 0
            print(f"  Ep {ep:3d}/{args_cli.episodes} | "
                  f"Reward: {ep_reward:8.1f} (avg5: {avg_r:8.1f}) | "
                  f"Steps: {ep_steps} | Reach: {'✓' if ep_reach else '✗'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — Phase {args_cli.phase}: {phase_names[args_cli.phase]}")
    print(f"{'='*60}")
    print(f"  Mean reward:      {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Episode length:    {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f}")
    print(f"  Reach success:     {successes['reach']}/{args_cli.episodes} ({100*successes['reach']/args_cli.episodes:.0f}%)")
    print(f"{'='*60}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
