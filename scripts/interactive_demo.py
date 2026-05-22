#!/usr/bin/env python3
"""
Interactive 6-DOF Arm Demo

Real-time interactive system where you can:
- Move the object with arrow keys
- Press 'g' to grasp
- Press 'p' to place in basket
- Press 'r' to reset
- Press 'q' to quit

Uses MuJoCo rendering for real-time visualization.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from envs.pick_and_place_env import PickAndPlaceEnv
from robots.kuka_iiwa import KukaRobot
from sensors.camera import OverheadCamera


def interactive_demo(model_path: str = "models/phase_2/final_model", 
                     xml_path: str = "scenes/pick_and_place_scene.xml"):
    """
    Interactive demo with keyboard controls.
    """
    print("=== INTERACTIVE 6-DOF ARM DEMO ===")
    print("Controls:")
    print("  Arrow Keys: Move object left/right/up/down")
    print("  g: Grasp object")
    print("  p: Place in basket")
    print("  r: Reset to initial position")
    print("  q: Quit")
    print()
    
    # Create environment
    env = PickAndPlaceEnv(xml_path, curriculum_phase=2, use_vision=True)
    
    # Load trained model
    model = SAC.load(model_path)
    
    # Get camera
    camera = OverheadCamera(env.robot)
    
    # Initial reset
    obs, info = env.reset()
    
    # Main loop
    while True:
        # Render current state
        img = env.render(mode="rgb_array")
        if img is not None:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add UI text
            cv2.putText(img_bgr, "INTERACTIVE DEMO - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Object pos: {env.robot.get_object_pos()}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_bgr, f"Grasped: {env.robot.is_object_grasped()}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('6-DOF Arm Interactive Demo', img_bgr)
        
        # Wait for key press (30ms)
        key = cv2.waitKey(30) & 0xFF
        
        # Handle key presses
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('r'):
            print("Resetting...")
            obs, info = env.reset()
        elif key == ord('g'):
            print("Grasping...")
            # Send grasp action
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            obs, reward, terminated, truncated, info = env.step(action)
        elif key == ord('p'):
            print("Placing in basket...")
            # Send place action (move to basket position)
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, terminated, truncated, info = env.step(action)
        elif key == 27:  # ESC
            print("Quitting...")
            break
        elif key == 81:  # Left arrow
            print("Moving object left...")
            # Get current object position
            obj_pos = env.robot.get_object_pos()
            obj_pos[0] -= 0.02  # Move left
            env.robot.set_object_position(obj_pos)
        elif key == 83:  # Right arrow
            print("Moving object right...")
            obj_pos = env.robot.get_object_pos()
            obj_pos[0] += 0.02  # Move right
            env.robot.set_object_position(obj_pos)
        elif key == 82:  # Up arrow
            print("Moving object up...")
            obj_pos = env.robot.get_object_pos()
            obj_pos[1] += 0.02  # Move up
            env.robot.set_object_position(obj_pos)
        elif key == 84:  # Down arrow
            print("Moving object down...")
            obj_pos = env.robot.get_object_pos()
            obj_pos[1] -= 0.02  # Move down
            env.robot.set_object_position(obj_pos)
    
    cv2.destroyAllWindows()
    env.close()
    print("Demo finished.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive 6-DOF arm demo")
    parser.add_argument("--model", type=str, default="models/phase_2/final_model",
                       help="Path to trained model")
    parser.add_argument("--xml", type=str, default="scenes/pick_and_place_scene.xml",
                       help="Path to MuJoCo XML scene")
    
    args = parser.parse_args()
    
    interactive_demo(args.model, args.xml)
