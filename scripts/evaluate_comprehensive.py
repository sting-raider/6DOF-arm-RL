"""
Comprehensive evaluation script for trained pick-and-place policies.

Features:
  - Multiple evaluation episodes with detailed per-episode metrics
  - Correct VecNormalize loading per phase model
  - Success rate tracking (reach / grasp / place)
  - Reward breakdown analysis
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.pick_and_place_env import PickAndPlaceEnv


def find_vec_normalize(model_path: str, phase: int) -> str | None:
    """
    Search for vec_normalize.pkl in several candidate locations:
    1. Same directory as model file
    2. models/phase_<N>/
    3. models/phase_<N>/best_model/ (if model is best_model.zip)
    """
    candidates = [
        os.path.join(os.path.dirname(model_path), "vec_normalize.pkl"),
        os.path.join("models", f"phase_{phase}", "vec_normalize.pkl"),
        os.path.join(os.path.dirname(os.path.dirname(model_path)), "vec_normalize.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def evaluate_comprehensive(model_path: str, phase: int, n_episodes: int = 20):
    """
    Run comprehensive evaluation of a trained SAC policy.

    Args:
        model_path: Path to saved SAC model (.zip, with or without extension)
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        n_episodes: Number of evaluation episodes
    """
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"\n=== COMPREHENSIVE EVALUATION: Phase {phase} ({phase_names[phase]}) ===")
    print(f"Episodes: {n_episodes}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "scenes", "pick_and_place_scene.xml")

    # Build a single-env DummyVecEnv (evaluation always single-threaded)
    def make_env():
        env = PickAndPlaceEnv(xml_path=xml_path, curriculum_phase=phase)
        return Monitor(env, info_keywords=(
            "reach_success", "grasp_success", "place_success", "dist_to_obj",
        ))

    env = DummyVecEnv([make_env])

    # Load VecNormalize stats — critical for correct behaviour
    norm_path = find_vec_normalize(model_path, phase)
    if norm_path:
        print(f"Loading VecNormalize from: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = False   # freeze running stats
        env.norm_reward = False  # don't normalise rewards at eval time
    else:
        print("WARNING: vec_normalize.pkl not found — observations will NOT be normalised.")
        print("         Evaluation results may be unreliable.")

    # Load model
    model = SAC.load(model_path, env=env, device="auto")
    print(f"Model loaded from: {model_path}\n")

    # ── Evaluation loop ──────────────────────────────────────────────────────
    rewards = []
    episode_lengths = []
    success_counts = {"reach": 0, "grasp": 0, "place": 0}

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        # Track whether success was achieved at ANY point during the episode
        ep_reach = False
        ep_grasp = False
        ep_place = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            # reward / dones are arrays of length 1 in DummyVecEnv
            ep_reward += float(reward[0])
            ep_steps += 1

            info = infos[0]
            ep_reach = ep_reach or bool(info.get("reach_success", False))
            ep_grasp = ep_grasp or bool(info.get("grasp_success", False))
            ep_place = ep_place or bool(info.get("place_success", False))

            if dones[0]:
                break

        rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        if ep_reach:
            success_counts["reach"] += 1
        if ep_grasp:
            success_counts["grasp"] += 1
        if ep_place:
            success_counts["place"] += 1

        print(f"  Episode {ep+1:2d}: reward={ep_reward:8.2f}, "
              f"steps={ep_steps:3d}, "
              f"reach={'✓' if ep_reach else '✗'}, "
              f"grasp={'✓' if ep_grasp else '✗'}, "
              f"place={'✓' if ep_place else '✗'}")

    env.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Mean reward:    {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
    print(f"Median reward:  {np.median(rewards):8.2f}")
    print(f"Min / Max:      {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"Episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Reach  success: {success_counts['reach']}/{n_episodes} "
          f"({100*success_counts['reach']/n_episodes:.0f}%)")
    print(f"Grasp  success: {success_counts['grasp']}/{n_episodes} "
          f"({100*success_counts['grasp']/n_episodes:.0f}%)")
    print(f"Place  success: {success_counts['place']}/{n_episodes} "
          f"({100*success_counts['place']/n_episodes:.0f}%)")

    # Save text results
    os.makedirs("videos", exist_ok=True)
    results_path = os.path.join("videos", f"phase_{phase}_evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Phase {phase} ({phase_names[phase]}) — {n_episodes} episodes\n")
        f.write(f"Mean reward:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"Reach success: {success_counts['reach']}/{n_episodes}\n")
        f.write(f"Grasp success: {success_counts['grasp']}/{n_episodes}\n")
        f.write(f"Place success: {success_counts['place']}/{n_episodes}\n")
    print(f"\nResults saved to {results_path}")

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "reach_success_rate": success_counts["reach"] / n_episodes,
        "grasp_success_rate": success_counts["grasp"] / n_episodes,
        "place_success_rate": success_counts["place"] / n_episodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of trained policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SAC model (e.g. models/phase_2/best_model)")
    parser.add_argument("--phase", type=int, default=2, choices=[0, 1, 2],
                        help="Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate_comprehensive(
        model_path=args.model,
        phase=args.phase,
        n_episodes=args.episodes,
    )
