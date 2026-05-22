"""
Pick-and-Place Gymnasium Environment — v2 (Fixed Rewards).

Curriculum phases:
  Phase 0: REACH  - Move end-effector to object
  Phase 1: GRASP  - Reach + grasp + lift
  Phase 2: PLACE  - Full pick-and-place into basket

Observation (20D):
  ee_pos (3) + obj_pos (3) + relative_pos (3) + joint_pos (5) +
  joint_vel (5) + gripper_state (1)

Action (6D):
  delta_joint_0..4 (5) + gripper_action (1)

Key changes from v1:
  - Positive potential-based reward shaping (not flat negative)
  - prev_dist tracking for temporal shaping signal
  - Richer observations (relative pos, joint state)
  - Per-phase episode length limits
  - Reward component logging for diagnostics
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict

from robots.kuka_iiwa import KukaRobot
from sensors.camera import OverheadCamera
from utils.constants import (
    DELTA_MAX,
    OBJECT_SPAWN_X_MIN, OBJECT_SPAWN_X_MAX,
    OBJECT_SPAWN_Y_MIN, OBJECT_SPAWN_Y_MAX,
    TABLE_HEIGHT, BASKET_POS,
)

# Per-phase episode lengths (steps)
PHASE_MAX_STEPS = {0: 200, 1: 300, 2: 400}

# Workspace radius for reward normalization (meters)
MAX_WORKSPACE_DIST = 1.0


class PickAndPlaceEnv(gym.Env):
    """
    6-DOF arm pick-and-place environment with curriculum learning
    and positive potential-based reward shaping.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: str, curriculum_phase: int = 0,
                 use_vision: bool = False, render_mode: str = "rgb_array"):
        """
        Initialize the environment.

        Args:
            xml_path: Path to MuJoCo XML scene file.
            curriculum_phase: 0=REACH, 1=GRASP, 2=PLACE.
            use_vision: If True, include camera image in observation.
            render_mode: "rgb_array" or "human".
        """
        super().__init__()

        self.curriculum_phase = curriculum_phase
        self.use_vision = use_vision
        self.render_mode = render_mode

        # Initialize robot and camera
        self.robot = KukaRobot(xml_path)
        self.camera = OverheadCamera(self.robot) if use_vision else None

        # Observation space (20D):
        #   ee_pos (3) + obj_pos (3) + relative_pos (3) +
        #   joint_pos (5) + joint_vel (5) + gripper_state (1)
        obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 5 joint deltas + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Episode state
        self._episode_step = 0
        self._max_episode_steps = PHASE_MAX_STEPS.get(curriculum_phase, 200)
        self._reach_success = False
        self._grasp_success = False

        # For potential-based reward shaping
        self._prev_dist = 0.0
        self._prev_dist_to_basket = 0.0

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> tuple:
        """
        Reset the environment.

        Returns:
            observation, info dict
        """
        super().reset(seed=seed)

        # Randomize object position on table
        obj_x = self.np_random.uniform(OBJECT_SPAWN_X_MIN, OBJECT_SPAWN_X_MAX)
        obj_y = self.np_random.uniform(OBJECT_SPAWN_Y_MIN, OBJECT_SPAWN_Y_MAX)
        obj_pos = np.array([obj_x, obj_y, TABLE_HEIGHT + 0.021])  # on table

        self.robot.reset(object_pos=obj_pos)

        self._episode_step = 0
        self._reach_success = False
        self._grasp_success = False

        # Initialize shaping distances
        ee_pos = self.robot.get_ee_pos()
        self._prev_dist = float(np.linalg.norm(ee_pos - obj_pos))
        self._prev_dist_to_basket = float(np.linalg.norm(
            obj_pos - BASKET_POS
        ))

        obs = self._get_observation()
        info = {"reset": True}
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one environment step.

        Args:
            action: (6,) array of [delta_j0..delta_j4, gripper]

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0)
        self.robot.apply_action(action)
        self._episode_step += 1

        reward, reward_info = self._compute_reward()
        obs = self._get_observation()

        terminated = self._check_terminated()
        truncated = self._episode_step >= self._max_episode_steps

        info = {
            "reach_success": self._reach_success,
            "grasp_success": self._grasp_success,
            "place_success": self.robot.is_object_in_basket(),
            "episode_step": self._episode_step,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector (20D).

        Returns:
            [ee_pos(3), obj_pos(3), relative_pos(3),
             joint_pos(5), joint_vel(5), gripper_state(1)]
        """
        ee_pos = self.robot.get_ee_pos()
        obj_pos = self.robot.get_object_pos()
        relative_pos = obj_pos - ee_pos  # vector from EE to object

        # Joint state (first 5 actuated joints)
        joint_pos = self.robot.get_joint_positions()[:5]
        joint_vel = self.robot.get_joint_velocities()[:5]

        gripper = np.array([self.robot.get_gripper_state()])

        obs = np.concatenate([
            ee_pos,         # 3
            obj_pos,        # 3
            relative_pos,   # 3
            joint_pos,      # 5
            joint_vel,      # 5
            gripper,        # 1
        ])  # total = 20
        return obs.astype(np.float32)

    def _compute_reward(self) -> tuple:
        """
        Compute reward with positive potential-based shaping.

        Returns:
            (reward_float, reward_info_dict)
        """
        ee_pos = self.robot.get_ee_pos()
        obj_pos = self.robot.get_object_pos()
        dist = float(np.linalg.norm(ee_pos - obj_pos))

        # === Positive baseline: higher = closer to object ===
        # Maps [0, MAX_WORKSPACE_DIST] → [1.0, 0.0]
        baseline = max(0.0, 1.0 - dist / MAX_WORKSPACE_DIST)

        # === Temporal shaping: reward for getting closer ===
        shaping = 10.0 * (self._prev_dist - dist)
        self._prev_dist = dist

        reward = baseline + shaping

        # Track reward components for logging
        reward_info = {
            "r_baseline": baseline,
            "r_shaping": shaping,
            "r_reach_bonus": 0.0,
            "r_grasp_bonus": 0.0,
            "r_lift_bonus": 0.0,
            "r_place_shaping": 0.0,
            "r_place_bonus": 0.0,
            "dist_to_obj": dist,
        }

        # === Phase 0: REACH ===
        if dist < 0.05:
            reward += 5.0
            reward_info["r_reach_bonus"] = 5.0
            self._reach_success = True

        # === Phase 1+: GRASP ===
        if self.curriculum_phase >= 1:
            if self.robot.is_object_grasped():
                reward += 5.0
                reward_info["r_grasp_bonus"] = 5.0
                self._grasp_success = True

                # Lift bonus: reward for getting object above table
                lift_height = obj_pos[2] - (TABLE_HEIGHT + 0.02)
                if lift_height > 0.02:
                    lift_bonus = min(lift_height * 20.0, 3.0)  # up to 3.0
                    reward += lift_bonus
                    reward_info["r_lift_bonus"] = lift_bonus

        # === Phase 2: PLACE ===
        if self.curriculum_phase >= 2 and self.robot.is_object_grasped():
            dist_to_basket = float(np.linalg.norm(obj_pos - BASKET_POS))

            # Shaping toward basket
            basket_shaping = 5.0 * (self._prev_dist_to_basket - dist_to_basket)
            self._prev_dist_to_basket = dist_to_basket
            reward += basket_shaping
            reward_info["r_place_shaping"] = basket_shaping
            reward_info["dist_to_basket"] = dist_to_basket

            # Big bonus for successful placement
            if self.robot.is_object_in_basket():
                reward += 50.0
                reward_info["r_place_bonus"] = 50.0

        # Small action efficiency penalty (encourages shorter paths)
        reward -= 0.005

        return reward, reward_info

    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate early.

        Terminates if:
        - Object placed in basket (phase 2) — success
        - Object falls off table — failure
        """
        if self.curriculum_phase == 2 and self.robot.is_object_in_basket():
            return True

        obj_pos = self.robot.get_object_pos()
        if obj_pos[2] < TABLE_HEIGHT - 0.1:  # fell off table
            return True

        return False

    def render(self) -> Optional[np.ndarray]:
        """Render the current frame."""
        if self.render_mode == "rgb_array":
            if self.camera is not None:
                return self.camera.capture_rgb()
            return self.robot.render_image()
        return None

    def close(self):
        """Clean up MuJoCo resources."""
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot.close()
