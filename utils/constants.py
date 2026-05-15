"""
Constants for the 6-DOF RL environment.
"""

import numpy as np

# Simulation
TIMESTEP = 0.002
SUBSTEPS = 5
FRAME_SKIP = 10  # Physics steps per RL step

# Robot
NUM_JOINTS = 6
# Joint limits (radians)
JOINT_LIMITS_LOW = np.array([-np.pi, -1.5708, -np.pi, -3.14159, -1.5708, -np.pi])
JOINT_LIMITS_HIGH = np.array([np.pi, 1.5708, np.pi, 3.14159, 1.5708, np.pi])
# Default joint deltas per step (max movement per action)
DELTA_MAX = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0])  # last is gripper

# Home position (radians)
HOME_ANGLES = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.0])

# Gripper
GRASP_DISTANCE_THRESHOLD = 0.05  # meters
WELD_BREAK_DISTANCE = 0.15  # if object pulled too far, break weld

# Scene
TABLE_HEIGHT = 0.8
TABLE_SIZE = (0.3, 0.2)  # half-widths
BASKET_POS = np.array([0.4, 0.0, TABLE_HEIGHT])
BASKET_SIZE = (0.075, 0.075, 0.05)

# Object spawn range (on table)
OBJECT_SPAWN_X_MIN = 0.05
OBJECT_SPAWN_X_MAX = 0.35
OBJECT_SPAWN_Y_MIN = -0.15
OBJECT_SPAWN_Y_MAX = 0.15

# Vision
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BACKGROUND_TABLE_COLOR = np.array([150, 150, 150])  # gray
OBJECT_COLOR_THRESHOLD = 30  # for HSV detection

# Environment
MAX_EPISODE_STEPS = 100
REWARD_REACH_SUCCESS = 10.0
REWARD_GRASP_BONUS = 5.0
REWARD_PLACE_SUCCESS = 50.0
REWARD_DISTANCE_SCALE = -1.0
REWARD_PER_STEP = -0.01

# Pixel to world scale (approximate, depends on camera FOV)
PIXEL_TO_WORLD_SCALE = 0.01
