"""
Evaluation script for trained SAC policy.

Usage:
    python scripts/evaluate.py --phase 0 --model models/phase_0/final_model
    python scripts/evaluate.py --phase 2 --model models/phase_2/final_model --record
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.pick_and_place_env import PickAndPlaceEnv


def evaluate(model_path: str, phase: int, n_episodes: int = 10,
             record: bool = False, video_dir: str = "videos"):
    """
    Evaluate a trained policy.

    Args:
        model_path: Path to saved SAC model
        phase: Curriculum phase
        n_episodes: Number of evaluation episodes
        record: Whether to record videos
        video_dir: Directory to save videos
    """
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    print(f"=== Evaluating Phase {phase}: {phase_names[phase]} ===")

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
    successes = {"reach": 0, "grasp": 0, "place": 0}

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        frames = [] if record else None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0] if hasattr(reward, '__iter__') else reward

            if record:
                frames.append(env.render(mode="rgb_array"))

        rewards.append(ep_reward)

        # Track successes
        if info[0].get("reach_success", False):
            successes["reach"] += 1
        if info[0].get("grasp_success", False):
            successes["grasp"] += 1
        if info[0].get("place_success", False):
            successes["place"] += 1

        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, "
              f"reach={info[0].get('reach_success', False)}, "
              f"grasp={info[0].get('grasp_success', False)}, "
              f"place={info[0].get('place_success', False)}")

    # Summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Reach success: {successes['reach']}/{n_episodes} "
          f"({100*successes['reach']/n_episodes:.0f}%)")
    print(f"Grasp success: {successes['grasp']}/{n_episodes} "
          f"({100*successes['grasp']/n_episodes:.0f}%)")
    print(f"Place success: {successes['place']}/{n_episodes} "
          f"({100*successes['place']/n_episodes:.0f}%)")

    if record and frames:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"phase_{phase}_eval.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SAC policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SAC model")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="Curriculum phase")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true",
                        help="Record video of evaluation")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")

    args = parser.parse_args()
    evaluate(model_path=args.model, phase=args.phase, n_episodes=args.episodes,
             record=args.record, video_dir=args.video_dir)
