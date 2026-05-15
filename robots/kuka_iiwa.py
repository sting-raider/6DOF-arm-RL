"""
KUKA iiwa Robot Wrapper for MuJoCo.

Handles:
  - Loading the MuJoCo scene
  - PD position control for 6 arm joints
  - Gripper simulation (open/close)
  - Magnetic grasping via weld equality constraint
  - Forward kinematics (end-effector position)
  - Object manipulation
"""

import numpy as np
import mujoco
import os
from typing import Optional, Tuple

from utils.constants import (
    NUM_JOINTS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    HOME_ANGLES, DELTA_MAX, GRASP_DISTANCE_THRESHOLD, WELD_BREAK_DISTANCE,
    TIMESTEP, SUBSTEPS, TABLE_HEIGHT,
)


class KukaRobot:
    """
    MuJoCo robot wrapper for the 6-DOF KUKA iiwa arm with magnetic gripper.
    """

    # Actuator names (match XML actuator names)
    ARM_ACTUATOR_NAMES = [
        "actuator_joint1", "actuator_joint2", "actuator_joint3",
        "actuator_joint4", "actuator_joint5", "actuator_joint6"
    ]
    ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ]
    GRIPPER_JOINT_NAMES = ["gripper_left_joint", "gripper_right_joint"]

    def __init__(self, xml_path: str):
        """
        Load the MuJoCo model and initialize the robot.

        Args:
            xml_path: Path to the MuJoCo XML scene file.
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Build lookup tables for actuators, joints, bodies
        self._build_lookups()

        # Weld constraint (magnetic grasp)
        self._weld_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"
        )

        # Internal state (only first 5 arm joints are controlled by RL)
        self._target_joint_pos = HOME_ANGLES[:5].copy()
        self._gripper_action = 0.0  # -1=open, +1=close
        self._is_grasping = False
        self._object_grasped = False

        # Default control gains (PD)
        self._Kp = np.array([10.0] * NUM_JOINTS)
        self._Kd = np.array([2.0] * NUM_JOINTS)

        # Physics step parameters
        self._timestep = TIMESTEP
        self._n_substeps = SUBSTEPS

        # Initial reset to settle
        self.reset()

    def _build_lookups(self) -> None:
        """Cache IDs for actuators, joints, and bodies."""
        # Actuator IDs
        self._arm_act_ids = np.array([
            self.model.actuator(name).id
            for name in self.ARM_ACTUATOR_NAMES
        ])
        self._gripper_left_act_id = self.model.actuator("actuator_gripper_left").id
        self._gripper_right_act_id = self.model.actuator("actuator_gripper_right").id

        # Joint DOF addresses for position/velocity
        self._arm_jnt_ids = np.array([
            self.model.joint(name).id for name in self.ARM_JOINT_NAMES
        ])
        self._arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.joint(name).id]
            for name in self.ARM_JOINT_NAMES
        ])
        self._arm_qvel_adr = np.array([
            self.model.jnt_dofadr[self.model.joint(name).id]
            for name in self.ARM_JOINT_NAMES
        ])

        # Gripper joint IDs
        self._gripper_left_jnt_id = self.model.joint("gripper_left_joint").id
        self._gripper_right_jnt_id = self.model.joint("gripper_right_joint").id
        self._gripper_left_qpos_adr = self.model.jnt_qposadr[self._gripper_left_jnt_id]
        self._gripper_right_qpos_adr = self.model.jnt_qposadr[self._gripper_right_jnt_id]

        # Body IDs for FK
        self._ee_body_id = self.model.body("ee_mount").id
        self._object_body_id = self.model.body("object").id

        # Free joint address for object
        self._object_free_jnt_id = self.model.joint("object_free").id
        self._object_qpos_adr = self.model.jnt_qposadr[self._object_free_jnt_id]

    def reset(self, object_pos: Optional[np.ndarray] = None) -> None:
        """
        Reset the simulation to initial state.

        Args:
            object_pos: Optional 3D position to place the object.
                       If None, uses default spawn position.
        """
        # Re-initialize data (deep reset)
        self.data = mujoco.MjData(self.model)

        # Set arm to home position
        self._target_joint_pos = HOME_ANGLES[:5].copy()
        for i, qadr in enumerate(self._arm_qpos_adr):
            self.data.qpos[qadr] = HOME_ANGLES[i]
        for i, vadr in enumerate(self._arm_qvel_adr):
            self.data.qvel[vadr] = 0.0

        # Reset gripper to open
        self.data.qpos[self._gripper_left_qpos_adr] = 0.0
        self.data.qpos[self._gripper_right_qpos_adr] = 0.0
        gl_dof = self.model.jnt_dofadr[self._gripper_left_jnt_id]
        gr_dof = self.model.jnt_dofadr[self._gripper_right_jnt_id]
        self.data.qvel[gl_dof] = 0.0
        self.data.qvel[gr_dof] = 0.0
        self._gripper_action = 0.0
        self._is_grasping = False
        self._object_grasped = False

        # Disable weld (should already be false from XML, but ensure)
        if self._weld_eq_id >= 0:
            self.data.eq_active[self._weld_eq_id] = 0

        # Place object on table (random or given)
        if object_pos is None:
            # Default spawn position
            object_x = 0.2
            object_y = 0.0
            object_z = TABLE_HEIGHT + 0.02 + 0.001  # on table surface + half height
        else:
            object_x, object_y, object_z = object_pos

        obj_qpos_adr = self._object_qpos_adr
        self.data.qpos[obj_qpos_adr + 0] = object_x
        self.data.qpos[obj_qpos_adr + 1] = object_y
        self.data.qpos[obj_qpos_adr + 2] = object_z
        # Quaternion (identity): w=1, x=0, y=0, z=0
        self.data.qpos[obj_qpos_adr + 3] = 1.0
        self.data.qpos[obj_qpos_adr + 4] = 0.0
        self.data.qpos[obj_qpos_adr + 5] = 0.0
        self.data.qpos[obj_qpos_adr + 6] = 0.0

        # Step to settle physics
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def apply_action(self, action: np.ndarray) -> None:
        """
        Apply a 6D action and step the simulation.

        Args:
            action: Array of shape (6,)
                    action[0:5] = joint position deltas for 5 joints + 1 gripper
                    action[5] = gripper action (-1=open, +1=close)
        """
        if len(action) != 6:
            raise ValueError(f"Action must be 6D, got {len(action)}")

        joint_deltas = action[:5]
        gripper_act = float(np.clip(action[5], -1.0, 1.0))

        # Update target positions (delta control)
        self._target_joint_pos += joint_deltas * DELTA_MAX[:5]
        self._target_joint_pos = np.clip(
            self._target_joint_pos, JOINT_LIMITS_LOW[:5], JOINT_LIMITS_HIGH[:5]
        )

        # PD control for arm joints
        current_jpos = self.get_joint_positions()
        current_jvel = self.get_joint_velocities()

        for i in range(5):
            err = self._target_joint_pos[i] - current_jpos[i]
            ctrl = self._Kp[i] * err - self._Kd[i] * current_jvel[i]
            ctrl = float(np.clip(ctrl, -1.0, 1.0))
            self.data.ctrl[self._arm_act_ids[i]] = ctrl

        # Gripper control
        self._gripper_action = gripper_act
        gripper_ctrl = -gripper_act * 1.0  # negative: close, positive: open
        self.data.ctrl[self._gripper_left_act_id] = gripper_ctrl
        self.data.ctrl[self._gripper_right_act_id] = gripper_ctrl

        # Magnetic grasp logic
        self._update_grasp()

        # Physics step (with substeps)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

    def _update_grasp(self) -> None:
        """
        Check proximity and update the magnetic weld.
        If gripper is closed and EE is near object -> grasp
        If gripper is open -> release
        """
        if self._weld_eq_id < 0:
            return

        # Check gripper state
        gripper_closed = self._gripper_action > 0.0
        gripper_open = self._gripper_action < 0.0

        ee_pos = self.get_ee_pos()
        obj_pos = self.get_object_pos()
        dist = float(np.linalg.norm(ee_pos - obj_pos))

        if gripper_closed and dist < GRASP_DISTANCE_THRESHOLD:
            # Activate weld (grasp)
            if not self._is_grasping:
                self._is_grasping = True
                self._object_grasped = True
                self.data.eq_active[self._weld_eq_id] = 1
        elif gripper_open:
            # Deactivate weld (release)
            if self._is_grasping:
                self._is_grasping = False
                self.data.eq_active[self._weld_eq_id] = 0

    def get_joint_positions(self) -> np.ndarray:
        """Return current joint angles (radians), shape (6,)."""
        jpos = np.zeros(6)
        for i, qadr in enumerate(self._arm_qpos_adr):
            jpos[i] = self.data.qpos[qadr]
        return jpos

    def get_joint_velocities(self) -> np.ndarray:
        """Return current joint velocities (rad/s), shape (6,)."""
        jvel = np.zeros(6)
        for i, qveladr in enumerate(self._arm_qvel_adr):
            jvel[i] = self.data.qvel[qveladr]
        return jvel

    def get_ee_pos(self) -> np.ndarray:
        """Return end-effector world position, shape (3,)."""
        return self.data.xpos[self._ee_body_id].copy()

    def get_object_pos(self) -> np.ndarray:
        """Return object world position, shape (3,)."""
        return self.data.xpos[self._object_body_id].copy()

    def is_object_grasped(self) -> bool:
        """Return True if object is currently attached via weld."""
        return self._object_grasped and self._is_grasping

    def is_object_in_basket(self) -> bool:
        """
        Check if object center is inside the basket volume.
        Basket is at (0.4, 0, 0.8) with size (0.075, 0.075, 0.1).
        """
        obj_pos = self.get_object_pos()
        bx, by, bz = 0.4, 0.0, 0.8
        hw, hd, hh = 0.075, 0.075, 0.05  # use half-extents
        inside = (
            abs(obj_pos[0] - bx) < hw and
            abs(obj_pos[1] - by) < hd and
            bz - hh < obj_pos[2] < bz + hh * 2
        )
        return inside

    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Directly set the target joint positions (for reset/initialization)."""
        self._target_joint_pos[:5] = np.clip(
            targets[:5], JOINT_LIMITS_LOW[:5], JOINT_LIMITS_HIGH[:5]
        )

    def get_gripper_state(self) -> float:
        """
        Return gripper open/close state: -1=fully open, +1=fully closed.
        Based on average finger position.
        """
        left_qpos = self.data.qpos[self._gripper_left_qpos_adr]
        right_qpos = self.data.qpos[self._gripper_right_qpos_adr]
        avg = (left_qpos + right_qpos) / 2.0
        # Map to [-1, 1]: 0.02 (open limit) -> -1, -0.02 (close limit) -> +1
        return float(np.clip(-avg / 0.02, -1.0, 1.0))

    def render_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Render an RGB image from the overhead camera.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            RGB image as numpy array (H, W, 3), uint8.
        """
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera="overhead")
        rgb = renderer.render()
        renderer.close()
        return rgb