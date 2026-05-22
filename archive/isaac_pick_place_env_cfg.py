# Copyright (c) 2026, Isaac Lab 6-DOF Arm Pick-and-Place
# Based on Isaac Lab manipulation tasks framework

from dataclasses import MISSING
from typing import Any

import torch
import numpy as np

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners import UsdFileCfg, GroundPlaneCfg, DomeLightCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Scene definition
##

@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the 6-DOF arm pick-and-place scene."""
    
    num_envs: int = 512  # parallel environments for RTX 3060
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # table surface
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.0, 0.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, -0.35), 
            rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )
    
    # robot: will be set from robot config (KUKA IIWA or Franka)
    robot: ArticulationCfg = MISSING
    
    # object to pick (red cube)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.05)),
        spawn=UsdFileCfg(
            # Using a simple box prim instead of a USD file
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(0.05, 0.05, 0.05),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )
    
    # basket / target zone
    basket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.15, 0.05)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Containers/plastic_basket.usd",
            scale=(0.3, 0.3, 0.2),
        ),
    )
    
    # dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Joint position control for 5 DOF arm
    arm_action: ActionTermCfg = MISSING
    
    # Gripper action (open/close)
    gripper_action: ActionTermCfg = MISSING


##
# Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        
        # Joint states (5 joints + gripper)
        joint_pos = ObservationTermCfg(
            func=lambda env: env.robot.data.joint_pos,
        )
        joint_vel = ObservationTermCfg(
            func=lambda env: env.robot.data.joint_vel,
        )
        
        # End-effector pose
        ee_pos = ObservationTermCfg(
            func=lambda env: env.robot.data.body_pos_w[:, env.ee_idx, :],
        )
        
        # Object position relative to world
        object_pos = ObservationTermCfg(
            func=lambda env: env.object.data.root_pos_w[:, 0, :],
        )
        
        # Basket position
        basket_pos = ObservationTermCfg(
            func=lambda env: env.basket.data.root_pos_w[:, 0, :],
        )
        
        # Gripper state
        gripper_pos = ObservationTermCfg(
            func=lambda env: env.robot.data.joint_pos[:, -1:],
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


##
# Rewards (3-phase curriculum: REACH -> GRASP -> PLACE)
##

@configclass
class RewardsCfg:
    """Reward terms for the 3-phase curriculum."""
    
    # Phase 0: REACH - minimize distance to object
    reach_distance = RewardTermCfg(
        func=lambda env: -torch.norm(
            env.robot.data.body_pos_w[:, env.ee_idx, :] - 
            env.object.data.root_pos_w[:, 0, :], 
            dim=-1
        ),
        weight=1.0,
    )
    
    # Phase 1: GRASP - bonus for closing gripper near object
    grasp_bonus = RewardTermCfg(
        func=lambda env: torch.where(
            torch.norm(
                env.robot.data.body_pos_w[:, env.ee_idx, :] - 
                env.object.data.root_pos_w[:, 0, :], 
                dim=-1
            ) < 0.05,
            env.robot.data.joint_pos[:, -1] * 10.0,  # gripper closure bonus
            torch.zeros(env.num_envs, device=env.device)
        ),
        weight=5.0,
    )
    
    # Phase 2: PLACE - bonus for object near basket
    place_bonus = RewardTermCfg(
        func=lambda env: torch.where(
            torch.norm(
                env.object.data.root_pos_w[:, 0, :] - 
                env.basket.data.root_pos_w[:, 0, :], 
                dim=-1
            ) < 0.1,
            torch.ones(env.num_envs, device=env.device) * 50.0,
            torch.zeros(env.num_envs, device=env.device)
        ),
        weight=50.0,
    )
    
    # Action smoothness penalty
    action_rate = RewardTermCfg(
        func=lambda env: -torch.sum(torch.square(
            env.robot.data.joint_pos[:, :5] - env.prev_joint_pos
        ), dim=-1),
        weight=0.01,
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms."""
    
    time_out = TerminationTermCfg(
        func=lambda env: env.episode_length_buf >= env.max_episode_length,
        time_out=True,
    )
    
    object_fell = TerminationTermCfg(
        func=lambda env: env.object.data.root_pos_w[:, 0, 2] < 0.05,
    )
    
    success = TerminationTermCfg(
        func=lambda env: torch.norm(
            env.object.data.root_pos_w[:, 0, :] - 
            env.basket.data.root_pos_w[:, 0, :], 
            dim=-1
        ) < 0.05,
    )


##
# Curriculum (3-phase)
##

@configclass
class CurriculumCfg:
    """3-phase curriculum for pick-and-place."""
    
    # Phase 0: REACH (episodes 0-200)
    # Phase 1: GRASP (episodes 200-400) 
    # Phase 2: PLACE (episodes 400+)
    phase = CurriculumTermCfg(
        func=lambda env: torch.where(
            env.common_step_counter < 200 * env.max_episode_length,
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            torch.where(
                env.common_step_counter < 400 * env.max_episode_length,
                torch.ones(env.num_envs, dtype=torch.long, device=env.device),
                2 * torch.ones(env.num_envs, dtype=torch.long, device=env.device)
            )
        ),
    )


##
# Events (domain randomization)
##

@configclass
class EventsCfg:
    """Event terms for randomization."""
    
    reset_all = EventTermCfg(
        func=lambda env, env_ids: env._reset_idx(env_ids),
        mode="reset",
    )
    
    # Randomize object position on reset
    randomize_object = EventTermCfg(
        func=lambda env, env_ids: setattr(
            env.object.write_root_pose_to_sim()[env_ids], 
            'pos', 
            torch.tensor([
                env._random_uniform(0.3, 0.45, env_ids),
                env._random_uniform(-0.15, 0.15, env_ids),
                torch.ones_like(env_ids, dtype=torch.float) * 0.05
            ])
        ),
        mode="reset",
    )


##
# Main environment config
##

@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the 6-DOF arm pick-and-place environment."""
    
    # Scene
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg()
    
    # Basic settings
    decimation: int = 2
    episode_length_s: float = 10.0  # 10 seconds per episode
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventsCfg = EventsCfg()
    
    # Unused managers
    commands = None
    
    def __post_init__(self):
        """Post initialization configuration."""
        # Physics settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        
        # PhysX solver settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21
        
        # Viewer settings
        self.viewer.eye = [1.5, -1.0, 1.5]
        self.viewer.lookat = [0.5, 0.0, 0.0]
        
        # Compute max episode length
        self.max_episode_length = int(self.episode_length_s / (self.sim.dt * self.decimation))
