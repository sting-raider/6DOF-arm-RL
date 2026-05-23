#!/usr/bin/env python3
"""
Record 3D video of the trained model performing pick-and-place.
Uses MuJoCo's offscreen renderer (works without display).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio

from stable_baselines3 import SAC
from envs.pick_and_place_env import PickAndPlaceEnv


def record_video(model_path="models/phase_2/final_model", 
                 phase=2, num_episodes=3, fps=30,
                 output_path="videos/robot_demo.mp4"):
    """Record a 3D video of the robot performing pick-and-place."""
    
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "scenes", "pick_and_place_scene.xml")
    
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    
    all_frames = []
    
    for ep in range(num_episodes):
        print(f"\n🎬 Recording Episode {ep+1}...")
        
        env = PickAndPlaceEnv(xml_path, curriculum_phase=phase, use_vision=False)
        obs, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0
        
        while not done and step < 150:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            
            # Render every frame
            frame = env.render(mode="rgb_array")
            if frame is not None and frame.size > 0:
                all_frames.append(frame)
            
            status = ""
            if info.get("reach_success"): status += "[REACHED] "
            if info.get("grasp_success"): status += "[GRASPED] "
            if info.get("place_success"): status += "[PLACED!] "
            
            if step % 30 == 0:
                print(f"  Step {step:3d} | reward={ep_reward:.1f} | {status}")
        
        print(f"  ✅ Episode {ep+1} done: {step} steps, reward={ep_reward:.1f}")
        env.close()
    
    # Save video
    print(f"\n💾 Saving {len(all_frames)} frames to {output_path}...")
    imageio.mimsave(output_path, all_frames, fps=fps)
    print(f"✅ Video saved: {output_path}")
    
    # Also save a GIF preview
    gif_path = output_path.replace('.mp4', '.gif')
    # Downsample for GIF
    gif_frames = all_frames[::3][:200]
    print(f"💾 Saving GIF preview ({len(gif_frames)} frames)...")
    imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
    print(f"✅ GIF saved: {gif_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phase_2/final_model")
    parser.add_argument("--phase", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="videos/robot_demo.mp4")
    args = parser.parse_args()
    
    record_video(args.model, args.phase, args.episodes, args.fps, args.output)
