"""
Pick-and-Place Gymnasium Environment.

Curriculum phases:
  Phase 0: REACH  - Only reward for getting EE close to object
  Phase 1: GRASP  - Reward for reaching + grasping + lifting
  Phase 2: PLACE  - Full task: reach, grasp, place in basket

Observation: [ee_pos (3), object_pos (3), gripper_state (1), is_grasping (1)] = 8D
Action: [delta_joint_0..4, gripper] = 6D (continuous)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

from robots.kuka_iiwa import KukaRobot
from sensors.camera import OverheadCamera
from utils.constants import (
    NUM_JOINTS, DELTA_MAX, MAX_EPISODE_STEPS,
    REWARD_REACH_SUCCESS, REWARD_GRASP_BONUS, REWARD_PLACE_SUCCESS,
    REWARD_DISTANCE_SCALE, REWARD_PER_STEP,
    OBJECT_SPAWN_X_MIN, OBJECT_SPAWN_X_MAX,
    OBJECT_SPAWN_Y_MIN, OBJECT_SPAWN_Y_MAX,
    TABLE_HEIGHT, BASKET_POS,
)


class PickAndPlaceEnv(gym.Env):
    """
    6-DOF arm pick-and-place environment with curriculum learning.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: str, curriculum_phase: int = 0, use_vision: bool = False):
        """
        Initialize the environment.

        Args:
            xml_path: Path to MuJoCo XML scene file.
            curriculum_phase: 0=REACH, 1=GRASP, 2=PLACE.
            use_vision: If True, include camera image in observation.
        """
        super().__init__()

        self.curriculum_phase = curriculum_phase
        self.use_vision = use_vision

        # Initialize robot and camera
        self.robot = KukaRobot(xml_path)
        self.camera = OverheadCamera(self.robot) if use_vision else None

        # Observation space
        # EE pos (3) + object pos (3) + gripper (1) + grasping flag (1) = 8D
        obs_dim = 8
        if use_vision:
            # Will add image observation separately if needed
            pass

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 5 joint deltas + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._episode_step = 0
        self._reach_success = False
        self._grasp_success = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple:
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

        obs = self._get_observation()
        info = {}
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

        reward = self._compute_reward()
        obs = self._get_observation()

        terminated = self._check_terminated()
        truncated = self._episode_step >= MAX_EPISODE_STEPS

        info = {
            "reach_success": self._reach_success,
            "grasp_success": self._grasp_success,
            "place_success": self.robot.is_object_in_basket(),
            "episode_step": self._episode_step,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector.

        Returns:
            Observation array (8,): ee_pos (3), obj_pos (3), gripper (1), grasping (1)
        """
        ee_pos = self.robot.get_ee_pos()
        obj_pos = self.robot.get_object_pos()
        gripper = np.array([self.robot.get_gripper_state()])
        grasping = np.array([1.0 if self.robot.is_object_grasped() else 0.0])

        obs = np.concatenate([ee_pos, obj_pos, gripper, grasping])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward based on current curriculum phase.

        Phase 0 (REACH): Reward for minimizing EE-object distance
        Phase 1 (GRASP): REACH + bonus for grasping
        Phase 2 (PLACE): Full reward: reach + grasp + lift + place
        """
        ee_pos = self.robot.get_ee_pos()
        obj_pos = self.robot.get_object_pos()
        dist = float(np.linalg.norm(ee_pos - obj_pos))

        # Base distance reward (always active)
        reward = REWARD_DISTANCE_SCALE * dist

        # Per-step penalty to encourage efficiency
        reward += REWARD_PER_STEP

        # Phase 0: REACH
        if self.curriculum_phase == 0:
            if dist < 0.05:
                reward += REWARD_REACH_SUCCESS
                self._reach_success = True

        # Phase 1: GRASP
        elif self.curriculum_phase == 1:
            if dist < 0.05:
                reward += REWARD_REACH_SUCCESS
                self._reach_success = True

            if self.robot.is_object_grasped():
                reward += REWARD_GRASP_BONUS
                self._grasp_success = True
                # Bonus for lifting object above table
                if obj_pos[2] > TABLE_HEIGHT + 0.05:
                    reward += 2.0

        # Phase 2: PLACE
        elif self.curriculum_phase == 2:
            if dist < 0.05:
                reward += REWARD_REACH_SUCCESS
                self._reach_success = True

            if self.robot.is_object_grasped():
                reward += REWARD_GRASP_BONUS
                self._grasp_success = True

            if self.robot.is_object_in_basket():
                reward += REWARD_PLACE_SUCCESS

        return reward

    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate early.

        Terminates if:
        - Object placed in basket (phase 2)
        - Object falls off table
        """
        if self.curriculum_phase == 2 and self.robot.is_object_in_basket():
            return True

        obj_pos = self.robot.get_object_pos()
        if obj_pos[2] < TABLE_HEIGHT - 0.1:  # fell off table
            return True

        return False

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the current frame."""
        if mode == "rgb_array":
            return self.camera.capture_rgb() if self.camera else self.robot.render_image()
        return None

    def close(self):
        """Clean up resources."""
        pass
