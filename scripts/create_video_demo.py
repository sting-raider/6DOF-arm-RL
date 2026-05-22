#!/usr/bin/env python3
"""
Create video demonstrations of the trained model.
Generates MP4 videos showing the robot performing pick-and-place tasks.
"""

import sys
import os
import numpy as np
import imageio
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from envs.pick_and_place_env import PickAndPlaceEnv


def create_video_demo(model_path: str, phase: int = 2, num_episodes: int = 3,
                     video_dir: str = "videos", fps: int = 30):
    """
    Create video demonstrations of the trained model.
    
    Args:
        model_path: Path to saved model
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        num_episodes: Number of episodes to record
        video_dir: Directory to save videos
        fps: Frames per second for output video
    """
    print(f"Creating video demos for model: {model_path}")
    print(f"Phase: {phase}, Episodes: {num_episodes}")
    
    # Create video directory
    Path(video_dir).mkdir(exist_ok=True)
    
    # Create environment
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "scenes", "pick_and_place_scene.xml")
    env = PickAndPlaceEnv(xml_path, curriculum_phase=phase, use_vision=True)
    
    # Load trained model
    model = SAC.load(model_path)
    
    for episode in range(num_episodes):
        print(f"\n--- Recording Episode {episode + 1} ---")
        
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        frames = []
        
        # Record initial frame
        img = env.render(mode="rgb_array")
        if img is not None:
            frames.append(img)
        
        while not done and step_count < 100:  # Max 100 steps
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Record frame every 2 steps for smoother video
            if step_count % 2 == 0:
                img = env.render(mode="rgb_array")
                if img is not None:
                    frames.append(img)
            
            if done:
                print(f"Episode finished after {step_count} steps")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Success info: {info}")
                break
        
        # Save video
        if frames:
            video_path = os.path.join(video_dir, f"phase_{phase}_episode_{episode+1}.mp4")
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved: {video_path}")
        else:
            print("Warning: No frames recorded")
    
    env.close()
    print(f"\nVideo creation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create video demonstrations of trained model")
    parser.add_argument("--model", type=str, 
                       default="models/phase_2/final_model",
                       help="Path to trained model")
    parser.add_argument("--phase", type=int, default=2,
                       choices=[0, 1, 2],
                       help="Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to record")
    parser.add_argument("--video-dir", type=str, default="videos",
                       help="Directory to save videos")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output video")
    
    args = parser.parse_args()
    
    create_video_demo(args.model, args.phase, args.episodes, 
                      args.video_dir, args.fps)
