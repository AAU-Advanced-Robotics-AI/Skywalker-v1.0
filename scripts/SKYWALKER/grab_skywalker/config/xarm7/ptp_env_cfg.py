# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

print(f"[PTPEnvCfg] Using ptp_env_cfg.py from: {__file__}")

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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_data import FrameTransformerData
from isaaclab.sim import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

import grab_skywalker.mdp as mdp
from grab_skywalker.mdp import ptp_functions as ptp
from grab_skywalker.mdp import target_respawn as respawn

# Command imports for target management
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg

# Import reach MDP functions for command-based rewards
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from grab_skywalker.xarm7 import XARM7_CFG

# Import Isaac Lab built-in functions that we need
import isaaclab.envs.mdp as isaac_mdp

##
# Pre-defined configs
##
from grab_skywalker.config.xarm7 import agents  # noqa: F401, F403

##
# Scene definition
##




@configclass
class PTPSceneCfg(InteractiveSceneCfg):
    """Configuration for the Point-To-Point scene with XARM7 robot (no assembly)."""

    # Robot initialization - positioned lower with platform underneath
    robot = XARM7_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.0)),  # Lowered from 0.42 to 0.25
        spawn=XARM7_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0,
                fix_root_link=True,  # Fix the base link to prevent sliding
            ),
        )
    )
    
    # Individual platform for each robot to grip onto with base surface gripper
    robot_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robot_platform",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.15),  # Bigger platform size under robot base
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # Make kinematic to be truly immovable
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Light since it's kinematic
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # Enable collisions for surface gripper interaction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply", 
                static_friction=2.0,  # High friction
                dynamic_friction=2.0,  # High friction
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -0.93), rot=(1.0, 0.0, 0.0, 0.0)),  # Lowered platform position
    )

    # end-effector frame transformer
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

        # Ground plane as AssetBaseCfg with global prim path to avoid multi-environment indexing issues
    #ground = AssetBaseCfg(
    #    prim_path="/World/ground",
    #    spawn=GroundPlaneCfg(),
    #    collision_group=-1,
    #)

    # Lights - Balanced lighting for good visibility without being too bright
    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=5000.0,  # Reduced from 20000
            angle=0.5,
            color=(1.0, 1.0, 1.0),
        ),
    )
    
    # Add dome light for ambient lighting with sky texture
    dome_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,  # Reduced from 5000
            color=(0.9, 0.95, 1.0),  # Slightly blue tinted for natural look
            visible_in_primary_ray=True,
            texture_file="", # Empty for pure color
        ),
    )
    
    # Add additional sphere light for direct illumination
    main_light = AssetBaseCfg(
        prim_path="/World/mainLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=4000.0,  # Reduced from 15000
            radius=1.0,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 2.0, 3.0)),
    )
    
    # Add fill light to reduce shadows
    fill_light = AssetBaseCfg(
        prim_path="/World/fillLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=2000.0,  # Reduced from 8000
            radius=0.5,
            color=(0.95, 0.95, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.5, -1.5, 2.0)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    # Target pose command for point-to-point reaching
    target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_eef",  # End-effector frame
        resampling_time_range=(7.0, 7.0),  # Match episode length
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.5),   # Forward reach range
            pos_y=(-0.2, 0.2),  # Left-right reach range  
            pos_z=(0.35, 0.65),   # Height range above base (robot at 0.25m + reach height)
            roll=(3.14159, 3.14159),    # π radians (180°) - flip Z-axis direction
            pitch=(-2.3, -1.2), # -π/2 radians (-90°) - X-axis pointing upward
            yaw=(-0.9, 0.9),    # Small yaw variation
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Robot joint control for arm movement - Using Isaac Lab's scale
    arm_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint.*"],
        scale=0.5,  # Reduced to Isaac Lab's scale for better control
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot joint positions and velocities
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        
        # End-effector position in robot base frame
        ee_pos = ObsTerm(
            func=mdp.ee_position,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        

        ee_quat = ObsTerm(
            func=mdp.ee_orientation,  # or similar function that returns the EE orientation
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        # Target pose command (position + orientation)
        target_pose_command = ObsTerm(
            func=isaac_mdp.generated_commands, 
            params={"command_name": "target_pose"}
        )
        
        # Last action for temporal continuity
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # Reset robot to random joint positions
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


        # --- DEBUG: print limits/defaults at reset ---
    # print_limits_once = EventTerm(
    #     func=ptp.debug_print_limits_on_reset,
    #     mode="reset",
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary reward: position tracking using command error
    position_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-0.2,  # Negative weight for error (lower error = higher reward)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link_eef"), 
            "command_name": "target_pose"
        },
    )

        # --- DEBUG: print what training actually sees/applies (no reward effect) ---
    #debug_probe = RewTerm(
    #    func=ptp.debug_log_step,
    #    weight=0.000000000000001,
    #    params={"step_mod": 1, "env_index": 0},
    #)

    orientation_alignment = RewTerm(
        func=ptp.orientation_command_error_tanh,
        weight=0.05,  # tune
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_eef"), 
                "command_name": "target_pose",
                "std": 0.1
        },
    )

    
    # Fine-grained position tracking with tanh shaping
    position_tracking_fine = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=0.1,  # Positive weight with tanh shaping
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link_eef"),
            "std": 0.1,  # Standard deviation for tanh shaping
            "command_name": "target_pose"
        },
    )
    
    # Orientation tracking (less weight for PTP task)
    orientation_tracking = RewTerm(
        func=reach_mdp.orientation_command_error,
        weight=-0.1,  # Reduced weight as orientation less critical for PTP
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link_eef"),
            "command_name": "target_pose"
        },
    )

    # Action smoothness penalty
    action_rate_l2 = RewTerm(
        func=isaac_mdp.action_rate_l2, 
        weight=-0.0001
    )
    
    # Joint velocity penalty
    joint_vel_l2 = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time limit
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    
    # Success: reached target - DISABLED (we want continuous target respawning, not episode termination)
    # target_reached = DoneTerm(
    #     func=mdp.ptp_target_reached,
    #     params={
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "target_cfg": SceneEntityCfg("target_marker"),
    #         "threshold": 0.05,
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # Gradually increase action rate penalty
    action_rate = CurrTerm(
        func=isaac_mdp.modify_reward_weight, 
        params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 4500}
    )
    
    # Gradually increase joint velocity penalty
    joint_vel = CurrTerm(
        func=isaac_mdp.modify_reward_weight, 
        params={"term_name": "joint_vel_l2", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class PTPEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Point-To-Point environment with XARM7."""

    # Scene settings
    scene: PTPSceneCfg = PTPSceneCfg(num_envs=4096, env_spacing=2.0)  # Match Isaac Lab's 4096 environments
    
    # Basic settings
    episode_length_s = 7.0  # 7 seconds episode length for faster training cycles
    decimation = 2
    num_actions = 7  # 7 arm joints only
    num_observations = 35  # joint_pos(7) + joint_vel(7) + ee_pos(3) + target_pose(7) + last_action(7) + ee_rot(4)
    num_states = 0

    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()  # Add commands configuration
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.viewer.eye = [3.5, 3.5, 3.5]
        self.viewer.lookat = [0.0, 0.0, 0.5]
        # step settings
        self.sim.dt = 0.01  # 100Hz


@configclass 
class PTPEnvCfg_PLAY(PTPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # keep the robot at proper height above platform (same as training)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.25)
        # reduce the number of steps for faster simulation
        self.episode_length_s = 15.0
