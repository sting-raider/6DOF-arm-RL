#!/usr/bin/env python3
"""
Record high-quality 3D video of the trained model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import imageio
from stable_baselines3 import SAC
from robots.kuka_iiwa import KukaRobot
from utils.constants import TABLE_HEIGHT

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_DIR, "scenes", "pick_and_place_scene.xml")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "phase_2", "final_model")

print("Loading model and scene...")
model = SAC.load(MODEL_PATH)
robot = KukaRobot(XML_PATH)

frames = []
for ep in range(3):
    print(f"\n🎬 Episode {ep+1}")
    # Random object position
    x = np.random.uniform(0.1, 0.35)
    y = np.random.uniform(-0.15, 0.15)
    robot.reset(object_pos=np.array([x, y, TABLE_HEIGHT + 0.021]))
    
    for step in range(150):
        obs = np.concatenate([
            robot.get_ee_pos(),
            robot.get_object_pos(),
            np.array([robot.get_gripper_state()]),
            np.array([1.0 if robot.is_object_grasped() else 0.0])
        ]).astype(np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        robot.apply_action(action)
        
        # Render HIGH RES frame
        img = robot.render_image(width=640, height=480)
        frames.append(img)
        
        if robot.is_object_in_basket():
            print(f"  🎉 PLACED at step {step+1}!")
            # Add more frames to show result
            for _ in range(30):
                frames.append(robot.render_image(width=640, height=480))
            break
        
        if step % 50 == 0:
            print(f"  Step {step}")

print(f"\n💾 Saving {len(frames)} frames...")
imageio.mimsave(os.path.join(PROJECT_DIR, "videos/robot_3d_demo.mp4"), frames, fps=30)
print("✅ Done: videos/robot_3d_demo.mp4")
