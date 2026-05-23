#!/usr/bin/env python3
"""
Terminal-based interactive demo for the 6-DOF arm.
Works in any terminal environment.

Commands:
  g          - Grasp object
  r          - Release object
  p          - Try to place in basket
  a          - Auto mode (let the AI decide)
  left/right - Move object left/right (x)
  up/down    - Move object up/down (y)
  set X Y    - Set object position
  snap       - Save a screenshot
  reset      - Reset arm and object
  q          - Quit
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from robots.kuka_iiwa import KukaRobot
from utils.constants import TABLE_HEIGHT


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def main():
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    XML_PATH = os.path.join(PROJECT_DIR, "scenes", "pick_and_place_scene.xml")
    MODEL_PATH = os.path.join(PROJECT_DIR, "models", "phase_2", "final_model")
    SCREENSHOT_DIR = os.path.join(PROJECT_DIR, "screenshots")
    
    Path(SCREENSHOT_DIR).mkdir(exist_ok=True)
    
    # Initialize
    print("Loading scene...")
    robot = KukaRobot(XML_PATH)
    robot.reset(object_pos=np.array([0.2, 0.0, TABLE_HEIGHT + 0.021]))
    
    model = None
    if os.path.exists(MODEL_PATH):
        model = SAC.load(MODEL_PATH)
        print(f"Loaded trained model from {MODEL_PATH}")
    
    screenshot_count = 0
    steps = 0
    
    clear_screen()
    
    while True:
        # Get state
        obj_pos = robot.get_object_pos()
        ee_pos = robot.get_ee_pos()
        grasped = robot.is_object_grasped()
        in_basket = robot.is_object_in_basket()
        dist = np.linalg.norm(ee_pos - obj_pos)
        
        # Display status
        print("=" * 60)
        print("  🤖 6-DOF ARM INTERACTIVE DEMO")
        print("=" * 60)
        print()
        
        # Scene visualization (ASCII)
        print("  TABLE LAYOUT (top view):")
        print()
        
        # Simple ASCII grid showing positions
        grid_size = 20
        table_min_x, table_max_x = 0.0, 0.5
        table_min_y, table_max_y = -0.2, 0.2
        
        def to_grid(val, min_val, max_val, size):
            return int((val - min_val) / (max_val - min_val) * (size - 1))
        
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size // 2)]
        
        # Place objects on grid
        obj_gx = to_grid(obj_pos[0], table_min_x, table_max_x, grid_size)
        obj_gy = to_grid(obj_pos[1], table_min_y, table_max_y, grid_size // 2)
        ee_gx = to_grid(ee_pos[0], table_min_x, table_max_x, grid_size)
        ee_gy = to_grid(ee_pos[1], table_min_y, table_max_y, grid_size // 2)
        basket_gx = to_grid(0.4, table_min_x, table_max_x, grid_size)
        basket_gy = to_grid(0.0, table_min_y, table_max_y, grid_size // 2)
        
        # Mark positions
        for y in range(grid_size // 2):
            for x in range(grid_size):
                marker = '.'
                if (x, y) == (basket_gx, basket_gy):
                    marker = 'B'  # Basket
                if (x, y) == (obj_gx, obj_gy):
                    marker = 'O'  # Object
                if (x, y) == (ee_gx, ee_gy):
                    marker = 'E'  # End-effector
                grid[y][x] = marker
        
        # Print grid
        print("    " + "─" * grid_size)
        for row in grid:
            print("    │" + "".join(row) + "│")
        print("    " + "─" * grid_size)
        print("    Legend: O=Object  E=End-Effector  B=Basket")
        print()
        
        # Detailed state
        print(f"  📍 Object:    ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"  🦾 EE:        ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        print(f"  📏 Distance:  {dist:.4f}")
        print(f"  🔒 Grasped:   {'YES ✅' if grasped else 'no ❌'}")
        print(f"  📦 In Basket: {'YES 🎉' if in_basket else 'no'}")
        print(f"  🔢 Steps:     {steps}")
        print()
        print("─" * 60)
        print("  Commands:")
        print("    g - Grasp    r - Release    p - Place")
        print("    a - Auto AI  snap - Screenshot  reset")
        print("    ← → - Move X    ↑ ↓ - Move Y")
        print("    set X Y - Jump object to position")
        print("    q - Quit")
        print()
        
        cmd = input("  > ").strip().lower()
        
        if cmd == 'q':
            print("Goodbye! 👋")
            break
        
        elif cmd == 'g':
            robot.apply_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
            steps += 1
            print("  → Closing gripper...")
            
        elif cmd == 'r':
            robot.apply_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
            steps += 1
            print("  → Opening gripper...")
            
        elif cmd == 'p':
            if model is not None:
                target_pos = np.array([0.4, 0.0, TABLE_HEIGHT + 0.05])
                for _ in range(60):
                    obs = np.concatenate([
                        robot.get_ee_pos(),
                        target_pos,
                        np.array([robot.get_gripper_state()]),
                        np.array([1.0 if robot.is_object_grasped() else 0.0])
                    ]).astype(np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    robot.apply_action(action)
                    steps += 1
                robot.apply_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
                steps += 1
            else:
                print("  No trained model loaded!")
                
        elif cmd == 'a':
            if model is not None:
                for _ in range(100):
                    obs = np.concatenate([
                        robot.get_ee_pos(),
                        robot.get_object_pos(),
                        np.array([robot.get_gripper_state()]),
                        np.array([1.0 if robot.is_object_grasped() else 0.0])
                    ]).astype(np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    robot.apply_action(action)
                    steps += 1
                print("  🤖 Auto cycle complete!")
            else:
                print("  No trained model loaded!")
                
        elif cmd == 'snap':
            img = robot.render_image(width=640, height=480)
            from PIL import Image
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{screenshot_count:03d}.png")
            Image.fromarray(img).save(path)
            screenshot_count += 1
            print(f"  📸 Screenshot saved: {path}")
            
        elif cmd == 'reset':
            robot.reset(object_pos=np.array([0.2, 0.0, TABLE_HEIGHT + 0.021]))
            steps = 0
            print("  🔄 Reset complete!")
            
        elif cmd.startswith('set '):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    robot.set_object_position(np.array([x, y, TABLE_HEIGHT + 0.021]))
                    print(f"  → Moved object to ({x}, {y})")
                except ValueError:
                    print("  Invalid position!")
                    
        elif cmd == 'left' or cmd == '\x1b[D':
            obj = robot.get_object_pos()
            robot.set_object_position(np.array([obj[0] - 0.02, obj[1], TABLE_HEIGHT + 0.021]))
            print("  → Object moved left")
            
        elif cmd == 'right' or cmd == '\x1b[C':
            obj = robot.get_object_pos()
            robot.set_object_position(np.array([obj[0] + 0.02, obj[1], TABLE_HEIGHT + 0.021]))
            print("  → Object moved right")
            
        elif cmd == 'up' or cmd == '\x1b[A':
            obj = robot.get_object_pos()
            robot.set_object_position(np.array([obj[0], obj[1] + 0.02, TABLE_HEIGHT + 0.021]))
            print("  → Object moved up")
            
        elif cmd == 'down' or cmd == '\x1b[B':
            obj = robot.get_object_pos()
            robot.set_object_position(np.array([obj[0], obj[1] - 0.02, TABLE_HEIGHT + 0.021]))
            print("  → Object moved down")
            
        else:
            print(f"  Unknown command: {cmd}")
        
        input("  Press Enter to continue...")
        clear_screen()


if __name__ == "__main__":
    main()
