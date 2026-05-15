"""
Constants for the 6-DOF arm pick-and-place environment.
All dimensions in meters, angles in radians, time in seconds.
"""

import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------
TABLE_HEIGHT = 0.8          # Table surface z
TABLE_SIZE = [0.6, 0.4, 0.02]  # (length, width, thickness)
TABLE_POS = [0.0, 0.0, TABLE_HEIGHT - TABLE_SIZE[2]/2]  # Center of table

BASKET_POS = [0.4, 0.0, TABLE_HEIGHT]  # Center of basket on table surface
BASKET_SIZE = [0.15, 0.15, 0.1]  # (length, width, height)
BASKET_WALL_THICKNESS = 0.01

OBJECT_SIZE = [0.04, 0.04, 0.04]  # Cube side length
OBJECT_SPAWN_CENTER = [0.0, 0.0, TABLE_HEIGHT + OBJECT_SIZE[2]/2]
OBJECT_SPAWN_RANGE_X = [-0.2, 0.2]  # Relative to center
OBJECT_SPAWN_RANGE_Y = [-0.15, 0.15]

# ---------------------------------------------------------------------------
# Robot (KUKA iiwa 6-DOF simplified)
# ---------------------------------------------------------------------------
NUM_JOINTS = 6
GRIPPER_DOF = 1
TOTAL_ACTUATORS = NUM_JOINTS + GRIPPER_DOF

# Joint limits (radians) — deliberately wide to allow exploration
JOINT_LIMITS_LOW = np.array([
    -np.pi,      # Joint 1: base rotation
    -np.pi/2,    # Joint 2: shoulder
    -np.pi,      # Joint 3: elbow
    -np.pi,      # Joint 4: wrist 1
    -np.pi/2,    # Joint 5: wrist 2
    -np.pi,      # Joint 6: wrist 3
])
JOINT_LIMITS_HIGH = np.array([
    np.pi,
    np.pi/2,
    np.pi,
    np.pi,
    np.pi/2,
    np.pi,
])

# Joint damping (Nm·s/rad)
JOINT_DAMPING = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])

# Max joint velocity (rad/s)
MAX_JOINT_VELOCITY = 1.0

# Action scaling: target += action * DELTA_MAX
DELTA_MAX = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])  # rad per step

# Home pose (all zeros = arm pointing straight up, slightly forward)
HOME_ANGLES = np.array([0.0, -np.pi/6, np.pi/3, 0.0, -np.pi/3, 0.0])

# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------
GRIPPER_OPEN = 0.0    # When gripper action < 0
GRIPPER_CLOSE = 1.0   # When gripper action > 0
GRIPPER_THRESHOLD = 0.0  # Threshold for open/close decision

# Magnetic weld parameters
GRASP_DISTANCE_THRESHOLD = 0.08  # m: EE to object distance for grasp
WELD_BREAK_DISTANCE = 0.15         # m: auto-release if object pulled too far

# ---------------------------------------------------------------------------
# Camera (overhead, 45° down)
# ---------------------------------------------------------------------------
CAMERA_RESOLUTION = [640, 480]
CAMERA_POS = np.array([0.0, 0.0, 2.0])
CAMERA_QUAT = np.array([0.9659258, 0.258819, 0.0, 0.0])  # w, x, y, z for 45° pitch down
CAMERA_FOV = 60.0  # degrees, vertical
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_SENSOR_SIZE_MM = 36.0  # full-frame sensor width
CAMERA_NEAR = 0.01
CAMERA_FAR = 5.0

# Derived camera intrinsics (approximate, pinhole model)
CAMERA_FX = CAMERA_FOCAL_LENGTH_MM * CAMERA_RESOLUTION[0] / CAMERA_SENSOR_SIZE_MM
CAMERA_FY = CAMERA_FOCAL_LENGTH_MM * CAMERA_RESOLUTION[1] / CAMERA_SENSOR_SIZE_MM
CAMERA_CX = CAMERA_RESOLUTION[0] / 2.0
CAMERA_CY = CAMERA_RESOLUTION[1] / 2.0

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
TIMESTEP = 0.002            # 2 ms physics step
SUBSTEPS = 4                  # Substeps per control step
PHYSICS_SOLVER_ITER = 100     # Solver iterations
GRAVITY = np.array([0.0, 0.0, -9.81])
MAX_CONTACT_PENETRATION = 0.001  # m

# Object physical properties
OBJECT_MASS_RANGE = [0.05, 0.2]  # kg
OBJECT_FRICTION_RANGE = [0.3, 1.0]
OBJECT_COLOR_DEFAULT = [0.8, 0.3, 0.2, 1.0]  # RGBA

# Table friction
TABLE_FRICTION = 0.8

# ---------------------------------------------------------------------------
# RL
# ---------------------------------------------------------------------------
OBS_DIM = 21       # 6 joint_pos + 6 joint_vel + 3 ee_pos + 3 obj_pos + 3 rel_vec
ACT_DIM = 6        # 5 joint_deltas + 1 gripper
MAX_EPISODE_STEPS = 500

# ---------------------------------------------------------------------------
# Reward weights (curriculum phases)
# ---------------------------------------------------------------------------
ALPHA_DIST = 1.0      # Distance reward multiplier
BETA_GRASP = 5.0      # Grasp bonus
GAMMA_TRANSPORT = 0.5 # Transport reward multiplier
DELTA_PLACE = 10.0    # Place bonus
EPS_ACTION = 0.01     # Action penalty
ETA_TIME = -0.01      # Time penalty per step
REACH_DISTANCE = 0.02 # m: threshold for "reached" object
LIFT_HEIGHT = 0.1     # m: threshold for "lifted" object above table
PLACE_STEPS = 10      # steps: object must be in basket for this many steps

# Phase step thresholds (training steps)
PHASE_REACH_STEPS = 1_000_000
PHASE_GRASP_STEPS = 3_000_000
PHASE_PLACE_STEPS = 5_000_000

# ---------------------------------------------------------------------------
# Training Hyperparameters (SAC)
# ---------------------------------------------------------------------------
SAC_CONFIG = {
    "total_timesteps": 10_000_000,
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_kwargs": {"net_arch": [64, 64, 64]},
    "verbose": 1,
}

# VecEnv settings
NUM_ENVS = 64  # Will use DummyVecEnv for MuJoCo (CPU-based)
EVAL_FREQ = 100_000
CHECKPOINT_FREQ = 500_000
VIDEO_FREQ = 1_000_000
N_EVAL_EPISODES = 50
