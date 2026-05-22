"""
Final evaluation script: comprehensive analysis of trained policy.

Features:
  - Multiple evaluation episodes with detailed metrics
  - Video recording for visualization
  - Success rate tracking by phase
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


def evaluate_comprehensive(model_path: str, phase: int, n_episodes: int = 20,
                           record: bool = True, video_dir: str = "videos"):
    """
    Comprehensive evaluation across all curriculum phases.

    Args:
        model_path: Path to saved SAC model
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        n_episodes: Number of evaluation episodes
        record: Whether to record videos
        video_dir: Directory to save videos
    """
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"=== COMPREHENSIVE EVALUATION: Phase {phase} ({phase_names[phase]}) ===")
    print(f"Episodes: {n_episodes}")

    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "scenes", "pick_and_place_scene.xml")

    def make_env():
        env = PickAndPlaceEnv(xml_path=xml_path, curriculum_phase=phase)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # Load normalization stats if available
    norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = SAC.load(model_path, env=env)

    # Run evaluation
    rewards = []
    success_rates = {"reach": 0, "grasp": 0, "place": 0}
    total_steps = 0
    episode_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        frames = [] if record else None
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0] if hasattr(reward, '__iter__') else reward
            ep_steps += 1

            if record:
                frames.append(env.render(mode="rgb_array"))

        # Append results after episode ends (only once)
        rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        # Track successes
        if info[0].get("reach_success", False):
            success_rates["reach"] += 1
        if info[0].get("grasp_success", False):
            success_rates["grasp"] += 1
        if info[0].get("place_success", False):
            success_rates["place"] += 1

        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, "
              f"steps={ep_steps}, "
              f"reach={info[0].get('reach_success', False)}, "
              f"grasp={info[0].get('grasp_success', False)}, "
              f"place={info[0].get('place_success', False)}")

    # Summary
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Median reward: {np.median(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Reach success: {success_rates['reach']}/{n_episodes} "
          f"({100*success_rates['reach']/n_episodes:.0f}%)")
    print(f"Grasp success: {success_rates['grasp']}/{n_episodes} "
          f"({100*success_rates['grasp']/n_episodes:.0f}%)")
    print(f"Place success: {success_rates['place']}/{n_episodes} "
          f"({100*success_rates['place']/n_episodes:.0f}%)")

    # Detailed reward breakdown
    print(f"\n=== REWARD BREAKDOWN ===")
    reach_bonus = 10.0
    grasp_bonus = 5.0
    place_bonus = 50.0
    distance_penalty = -1.0
    per_step_penalty = -0.01
    
    avg_reach_bonus = sum(1 for r in rewards if r >= reach_bonus) / len(rewards)
    avg_grasp_bonus = sum(1 for r in rewards if r >= reach_bonus + grasp_bonus) / len(rewards)
    avg_place_bonus = sum(1 for r in rewards if r >= reach_bonus + grasp_bonus + place_bonus) / len(rewards)
    
    print(f"Average reach bonus achieved: {avg_reach_bonus:.1%}")
    print(f"Average grasp bonus achieved: {avg_grasp_bonus:.1%}")
    print(f"Average place bonus achieved: {avg_place_bonus:.1%}")

    if record and frames:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"phase_{phase}_comprehensive_eval.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")

    # Save results to file
    results_path = os.path.join(video_dir, f"phase_{phase}_evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Comprehensive Evaluation Results - Phase {phase} ({phase_names[phase]})\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"Success rates: reach={success_rates['reach']}/{n_episodes}, grasp={success_rates['grasp']}/{n_episodes}, place={success_rates['place']}/{n_episodes}\n")
        f.write(f"Reward breakdown: reach={avg_reach_bonus:.1%}, grasp={avg_grasp_bonus:.1%}, place={avg_place_bonus:.1%}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of trained policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SAC model")
    parser.add_argument("--phase", type=int, default=2, choices=[0, 1, 2],
                        help="Curriculum phase")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true",
                        help="Record video of evaluation")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")

    args = parser.parse_args()
    evaluate_comprehensive(model_path=args.model, phase=args.phase, n_episodes=args.episodes,
                           record=args.record, video_dir=args.video_dir)
