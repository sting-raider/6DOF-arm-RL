#!/usr/bin/env python3
"""
Visualize the trained model by running it in the environment with rendering.
Shows the robot performing the pick-and-place task.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from envs.pick_and_place_env import PickAndPlaceEnv


def visualize_model(model_path: str, phase: int = 2, num_episodes: int = 3):
    """
    Visualize the trained model by running episodes and rendering.
    
    Args:
        model_path: Path to saved model
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        num_episodes: Number of episodes to visualize
    """
    print(f"Visualizing model: {model_path}")
    print(f"Phase: {phase}")
    print(f"Episodes: {num_episodes}")
    
    # Create environment
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "scenes", "pick_and_place_scene.xml")
    env = PickAndPlaceEnv(xml_path, curriculum_phase=phase, use_vision=True)
    
    # Load trained model
    model = SAC.load(model_path)
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        # Get initial image
        img = env.render(mode="rgb_array")
        
        # Show initial state
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Episode {episode + 1}, Step 0 - Initial State")
        plt.axis('off')
        plt.show()
        
        while not done and step_count < 100:  # Max 100 steps
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Render every 10 steps to avoid too many plots
            if step_count % 10 == 0 or done:
                img = env.render(mode="rgb_array")
                
                if img is not None:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.title(f"Episode {episode + 1}, Step {step_count}\n"
                             f"Reward: {reward:.2f}, Total: {total_reward:.2f}\n"
                             f"Info: {info}")
                    plt.axis('off')
                    plt.show()
            
            if done:
                print(f"Episode finished after {step_count} steps")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Success info: {info}")
                break
        
        print(f"Episode {episode + 1} completed: {step_count} steps, reward: {total_reward:.2f}")
    
    env.close()
    print(f"\nVisualization complete!")


def plot_training_curves():
    """
    Plot training curves from TensorBoard logs if available.
    """
    import os
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "phase_2")
    
    if not os.path.exists(log_dir):
        print("No training logs found for plotting")
        return
    
    # Find the latest log file
    log_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not log_files:
        print("No TensorBoard log files found")
        return
    
    latest_log = os.path.join(log_dir, sorted(log_files)[-1])
    event_acc = EventAccumulator(latest_log)
    event_acc.Reload()
    
    # Get scalar data
    if 'rollout/ep_rew_mean' in event_acc.scalars.Keys():
        rew_data = event_acc.scalars.Items('rollout/ep_rew_mean')
        steps = [s.step for s in rew_data]
        rewards = [s.value for s in rew_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rewards)
        plt.title('Episode Reward During Training (Phase 2)')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Reward')
        plt.grid(True)
        plt.show()
    else:
        print("No reward data found in logs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trained 6-DOF arm model")
    parser.add_argument("--model", type=str, 
                       default="models/phase_2/final_model",
                       help="Path to trained model")
    parser.add_argument("--phase", type=int, default=2,
                       choices=[0, 1, 2],
                       help="Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)")
    parser.add_argument("--episodes", type=int, default=2,
                       help="Number of episodes to visualize")
    parser.add_argument("--plot-curves", action="store_true",
                       help="Plot training curves from logs")
    
    args = parser.parse_args()
    
    if args.plot_curves:
        plot_training_curves()
    else:
        visualize_model(args.model, args.phase, args.episodes)
