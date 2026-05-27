# Copyright (c) 2026, 6DOF-arm-RL Project
"""UR10e pick-and-place environment config for Isaac Lab.

Three-phase curriculum:
  Phase 0 (REACH): Move end-effector to object
  Phase 1 (GRASP): Reach, grasp, and lift object  
  Phase 2 (PLACE): Full pick-and-place into basket
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG

import isaac_env.mdp as mdp
import isaaclab.envs.mdp as isaac_mdp



@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: ground, table, UR10e+Robotiq, object, basket, lights."""

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Table — kinematic rigid body with collision
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # UR10e with Robotiq 2F-85 gripper
    robot: ArticulationCfg = UR10e_ROBOTIQ_2F_85_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Override default pose: face the table (+X), arm slightly lifted
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.0,
                "elbow_joint": 1.5,
                "wrist_1_joint": -1.0,
                "wrist_2_joint": -1.0,
                "wrist_3_joint": 0.0,
            },
        ),
    )

    # Object to manipulate (red cube)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.0, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
    )

    # Basket (open-top box for placing)
    basket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.80),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


from isaaclab.envs.mdp import RelativeJointPositionActionCfg, BinaryJointPositionActionCfg

@configclass
class ActionsCfg:
    """Joint position delta + gripper actions."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.05,
        use_zero_offset=True,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.04},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Joint states (6 active arm joints)
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ]
            )}
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ]
            )}
        )
        # End effector position and gripper state
        ee_pos = ObsTerm(func=mdp.ee_position_scaled, params={"link_name": "wrist_3_link"})
        gripper_state = ObsTerm(func=mdp.gripper_state_scaled)
        # Object position and relative distance vector
        object_pos = ObsTerm(func=mdp.object_position)
        relative_pos = ObsTerm(func=mdp.relative_position)
        # Last actions (7 control deltas)
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms — positive shaping throughout.

    ``reach_reward`` reads ``env.cfg.curriculum_phase`` at runtime and dispatches
    REACH / GRASP / PLACE rewards automatically — no phase-switching needed in config.
    """

    reach = RewTerm(func=mdp.reach_reward, weight=1.0)
    # NOTE: action_penalty omitted in Phase 0 — raw pre-scale network outputs
    # cause return explosions if the critic ever destabilizes temporarily.
    # Add back in Phase 1+: action_penalty = RewTerm(func=mdp.action_penalty_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    # time_out=True: standard bootstrapping for non-terminal timeouts.
    # Safe here because without action_penalty, returns stay bounded [0, ~7].
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # object_fell uses -1.5 threshold: object must fall well below table to trigger.
    # Higher thresholds cause false positives due to physics contact initialization.
    object_fell = DoneTerm(func=mdp.object_fell, params={"minimum_height": -1.5})


@configclass
class EventCfg:
    """Domain randomization events."""
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.25, 0.45), "y": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e pick-and-place environment."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
    curriculum_phase: int = 0

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.5)

        # ── Stable PhysX Contacts (Standard for Isaac Lab Manipulation Tasks) ──
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
