#!/usr/bin/env python3
"""Isaac Lab evaluation script — runs N episodes and reports success metrics.

Uses obs_normalization=False for eval (loads trained weights with strict=False
to skip normalizer stats). Training uses normalization; eval uses raw observations.
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

argparser = argparse.ArgumentParser()
argparser.add_argument("--phase", type=int, required=True, choices=[0,1,2])
argparser.add_argument("--model", type=str, required=True, help="Path to model.pt")
argparser.add_argument("--num_envs", type=int, default=16)
argparser.add_argument("--episodes", type=int, default=20)
AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaac_env.env_cfg import PickPlaceEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"\n{'='*60}")
    print(f"  Evaluating Phase {args_cli.phase}: {phase_names[args_cli.phase]}")
    print(f"  Model: {args_cli.model}")
    print(f"  Envs: {args_cli.num_envs} | Episodes: {args_cli.episodes}")
    print(f"{'='*60}\n")

    env_cfg = PickPlaceEnvCfg()
    env_cfg.curriculum_phase = args_cli.phase
    env_cfg.scene.num_envs = args_cli.num_envs
    device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.device = device
    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)

    # Eval with obs_normalization=False — load weights with strict=False
    # to skip the normalizer state. Policy MLP weights are identical.
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
    runner = OnPolicyRunner(env, ppo_cfg, log_dir="/tmp/eval_log", device=device)

    print(f"  Loading model from {args_cli.model}...")
    runner.load(args_cli.model)  # configs match (same obs_normalization)
    print("  Model loaded.\n")

    # Cache scene references for real (unnormalized) position queries
    robot = raw_env.scene["robot"]
    ee_idx = robot.data.body_names.index("wrist_3_link")
    obj = raw_env.scene["object"]
    basket_center = torch.tensor([0.6, 0.0, 0.80], device=device, dtype=torch.float32)

    successes = {"reach": 0, "grasp": 0, "place": 0, "total_eps": 0}
    rewards = []
    ep_lengths = []

    episodes_to_run = args_cli.episodes
    episodes_done = 0
    obs = env.get_observations()

    num_envs = args_cli.num_envs
    ep_reward = torch.zeros(num_envs, device=device)
    ep_steps = torch.zeros(num_envs, device=device)
    ep_reach_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_grasp_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_place_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    print(f"  Running {episodes_to_run} episodes across {num_envs} envs...")
    start = time.time()

    while episodes_done < episodes_to_run:
        actions = runner.alg.actor(obs, stochastic_output=False)
        obs, rew, dones, infos = env.step(actions)

        ep_reward += rew
        ep_steps += 1

        # Get REAL (unnormalized) positions from the Isaac Lab scene
        ee_pos = robot.data.body_pos_w[:, ee_idx] - raw_env.scene.env_origins
        obj_pos = obj.data.root_pos_w
        grip_state = robot.data.joint_pos[:, -1]

        ee_to_obj = torch.norm(ee_pos - obj_pos, dim=1)
        obj_height = obj_pos[:, 2]
        obj_to_basket = torch.norm(obj_pos - basket_center, dim=1)

        ep_reach_success = ep_reach_success | (ee_to_obj < 0.05)
        ep_grasp_success = ep_grasp_success | ((grip_state > 0.02) & (obj_height > 0.88))
        ep_place_success = ep_place_success | (obj_to_basket < 0.10)

        done_mask = dones > 0.5
        for i in range(num_envs):
            if done_mask[i]:
                rewards.append(ep_reward[i].item())
                ep_lengths.append(ep_steps[i].item())
                successes["total_eps"] += 1
                if ep_reach_success[i]: successes["reach"] += 1
                if ep_grasp_success[i]: successes["grasp"] += 1
                if ep_place_success[i]: successes["place"] += 1
                episodes_done += 1

                ep_reward[i] = 0
                ep_steps[i] = 0
                ep_reach_success[i] = False
                ep_grasp_success[i] = False
                ep_place_success[i] = False

                if episodes_done % 10 == 0 or episodes_done == episodes_to_run:
                    elapsed = time.time() - start
                    avg_r = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                    pct = 100 * successes['reach'] / max(1, successes['total_eps'])
                    print(f"  [{elapsed:5.1f}s] Ep {episodes_done:3d}/{episodes_to_run} | "
                          f"AvgR: {avg_r:7.1f} | Reach: {successes['reach']}/{successes['total_eps']} ({pct:.0f}%)")

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
    print(f"  Reach success:      {successes['reach']}/{successes['total_eps']} "
          f"({100*successes['reach']/max(1,successes['total_eps']):.0f}%)")
    if phase >= 1:
        print(f"  Grasp success:      {successes['grasp']}/{successes['total_eps']} "
              f"({100*successes['grasp']/max(1,successes['total_eps']):.0f}%)")
    if phase >= 2:
        print(f"  Place success:      {successes['place']}/{successes['total_eps']} "
              f"({100*successes['place']/max(1,successes['total_eps']):.0f}%)")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
