#!/usr/bin/env python3

"""
Simple USD-based environment configuration that loads the pre-created welding scene.
This eliminates all the constraint violation issues at startup.
"""

print(f"[BaseMoveEnvUSDCfg] Using USD-based base_move_env_usd_cfg.py from: {__file__}")

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import FrameTransformerCfg

# Commands (same style as PTP)
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
# Built-in MDP terms (action penalties, time_out, etc.)
import isaaclab.envs.mdp as isaac_mdp
# Pose-error rewards we can reuse (point them at link_base instead of link_eef)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

# Your robot preset
from grab_skywalker.xarm7 import XARM7_CFG
# Your helpers
from grab_skywalker.mdp import base_move_functions as bm

##
# Pre-defined configs (agents)
##
from grab_skywalker.config.xarm7 import agents  # noqa: F401, F403

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class BaseMoveSceneUSDCfg(InteractiveSceneCfg):
    """Scene: Uses the pre-created welding USD scene with robot, wall, and constraints."""

    # Load the complete scene from USD
    # This includes the robot, wall, anchor, and physics setup
    welding_scene = AssetBaseCfg(
        prim_path="/World",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/bruno/IsaacLab/scripts/SKYWALKER/welding_scene.usd",
        ),
    )

    # Still need to configure the robot for Isaac Lab control
    # This gets the robot that's already in the USD scene
    robot = XARM7_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",  # Robot is already positioned in USD
        spawn=XARM7_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=False,  # Base is free to move
            ),
        ),
    )

    # (Optional) EE frame for debugging
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.06], rot=[0, 0, 0, 1]),
            ),
        ],
    )

# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Command terms for the USD-based MDP."""
    
    # Same as before - target pose command
    target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_base",  # We track base pose, not EE
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.8),     # Move in +X toward wall
            pos_y=(-0.3, 0.3),    # Small lateral movement
            pos_z=(0.15, 0.35),   # Keep above ground
            roll=(0.0, 0.0),      # No roll
            pitch=(0.0, 0.0),     # No pitch
            yaw=(-0.5, 0.5),      # Small yaw variations
        ),
    )

@configclass
class ActionsCfg:
    """Action specification for the USD-based MDP."""
    
    # Same as before - joint position control
    arm_action: isaac_mdp.JointPositionActionCfg = isaac_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specification for the USD-based MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Joint state
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        
        # Base state in world frame
        base_pos_w = ObsTerm(func=isaac_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_quat_w = ObsTerm(func=isaac_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # Target command
        target_pose_command = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target_pose"})
        
        # Actions
        actions = ObsTerm(func=isaac_mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
    """Event specification for the USD-based MDP."""
    
    # Simple reset to a default pose - no anchor manipulation needed since it's in the USD
    init_grounded_pose = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, -0.5),
                "y": (0.0, 0.0), 
                "z": (0.25, 0.25),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

@configclass
class RewardsCfg:
    """Reward specification for the USD-based MDP."""
    
    # Same rewards as before
    position_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"), "command_name": "target_pose"},
    )
    position_tracking_fine = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"), "command_name": "target_pose", "std": 0.1},
    )
    orientation_tracking = RewTerm(
        func=reach_mdp.orientation_command_error,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"), "command_name": "target_pose"},
    )
    
    # Base movement rewards
    base_motion = RewTerm(
        func=bm.base_linear_motion_reward, 
        weight=0.4,
        params={"target_cfg": SceneEntityCfg("robot", body_names="link_base")},
    )
    base_stability = RewTerm(
        func=bm.base_stability_reward, 
        weight=0.4,
        params={"target_cfg": SceneEntityCfg("robot", body_names="link_base")},
    )
    hover_penalty = RewTerm(
        func=bm.base_hovering_penalty, 
        weight=-0.2,
        params={"target_cfg": SceneEntityCfg("robot", body_names="link_base")},
    )
    
    # Action penalties
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1e-4)
    joint_vel_l2 = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})

@configclass
class TerminationsCfg:
    """Termination specification for the USD-based MDP."""
    
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum specification for the USD-based MDP."""
    
    action_rate = CurrTerm(
        func=isaac_mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=isaac_mdp.modify_reward_weight, params={"term_name": "joint_vel_l2", "weight": -1e-1, "num_steps": 10000}
    )

##
# Environment configuration
##

@configclass
class BaseMoveEnvUSDCfg(ManagerBasedRLEnvCfg):
    """Configuration for the USD-based base-movement RL environment."""
    
    # Scene settings
    scene: BaseMoveSceneUSDCfg = BaseMoveSceneUSDCfg(num_envs=4096, env_spacing=2.0)
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # Viewer settings
        self.viewer.resolution = (1280, 720)
        self.viewer.origin_type = "world"
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.physx.solver_type = 1  # TGS
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.max_position_iteration_count = 255
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 255
        self.sim.physx.contact_collection = 2  # CC_ALL_CONTACTS
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.bounce_threshold_velocity = 0.01
        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_collision_stack_size = 1024 * 1024 * 8
        self.sim.physx.gpu_heap_capacity = 1024 * 1024 * 128
        
        print(f"[BaseMoveEnvUSDCfg] USD-based environment configured with scene: {self.scene}")
