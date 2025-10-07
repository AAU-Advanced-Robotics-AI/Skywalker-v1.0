# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
import isaaclab.envs.mdp as isaac_mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from grab_skywalker.mdp import base_move_functions as bm
import math

USD_SCENE = "/home/bruno/IsaacLab/scripts/SKYWALKER/welding_scene3.usd"

# -----------------------------
# Scene
# -----------------------------
@configclass
class BaseMoveSceneCfg(InteractiveSceneCfg):
    """Pre-authored scene with robot welded (in USD) to anchor."""

    # Load scene once per env at a shallow mount point
    welding_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene",
        spawn=UsdFileCfg(usd_path=USD_SCENE),
    )

    # Wrap existing robot (NO spawning). Do NOT target 'world_joint' — it doesn't exist.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/skywalker_robot/xarm7",
        spawn=None,  # robot already exists inside your scene.usd
        init_state=ArticulationCfg.InitialStateCfg(
            # Values converted from your screenshot (deg → rad):
            joint_pos={
                "joint1": -0.01047198,   # -0.6°
                "joint2":  0.356047,     # 20.4°
                "joint3":  0.0523599,    # 3.0°
                "joint4":  0.764455,     # 43.8°
                "joint5":  0.0,          # 0°
                "joint6": -0.942478,     # -54.0°
                "joint7":  0.0,          # 0°
            }
        ),
        # inside BaseMoveSceneCfg.robot (replace your single "arm" actuator)
        actuators = {
            # big base joints
            "j1_j2": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2"],
                effort_limit_sim=179.4445,        # N·m (per joint cap)
                velocity_limit_sim=3.0,           # rad/s (set to your real speed if you have it)
                stiffness=600.0,                  # PD gains you can tune
                damping=40.0,
            ),
            # mid joints
            "j3": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit_sim=92.0,
                velocity_limit_sim=3.0,
                stiffness=600.0,
                damping=40.0,
            ),
            "j4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=92.0,
                velocity_limit_sim=3.0,
                stiffness=600.0,
                damping=40.0,
            ),
            "j5": ImplicitActuatorCfg(
                joint_names_expr=["joint5"],
                effort_limit_sim=81.6,
                velocity_limit_sim=3.0,
                stiffness=500.0,
                damping=35.0,
            ),
            # wrist joints
            "j6_j7": ImplicitActuatorCfg(
                joint_names_expr=["joint6", "joint7"],
                effort_limit_sim=30.6,
                velocity_limit_sim=4.0,
                stiffness=400.0,
                damping=30.0,
            ),
        }

    )

    # (optional) handles to existing rigid bodies if you need them
    wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/WallAssembly/wall",
        spawn=None,
    )
    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/WallAssembly/anchor",
        spawn=None,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/SkyDome",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            visible_in_primary_ray=False,   # hide HDRI in the background
        ),
    )


    # Ground
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1e-4, dynamic_friction=1e-4, restitution=0.0,
    #             friction_combine_mode="multiply", restitution_combine_mode="multiply"
    #         ),
    #     ),
    #     collision_group=-1,
    # )

# in BaseMoveSceneCfg
    target_pose_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_pose_marker",   # <- back under env root
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.32), rot=(1, 0, 0, 0)),
    )

# -----------------------------
# MDP
# -----------------------------
@configclass
class CommandsCfg:
    
     target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_base",
        resampling_time_range=(1000.0, 1000.0),  # Very long - we handle respawn manually
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(-0.15, 0.00),  # These ranges are only used for initial spawn
            pos_y=(-0.30, 0.30),  # All subsequent spawns handled by our functions
            pos_z=(0.00, 0.00),   # Keep at same height as base
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-0.9, 0.9),      # ±17 degrees
        ),
    )

@configclass
class ActionsCfg:
    arm_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=False
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        # 🎯 NEW: EE-relative observations (much better for real robot deployment!)
        base_pos_rel_ee  = ObsTerm(func=bm.base_position_relative_to_ee)   # Base position relative to EE (3D)
        base_quat_rel_ee = ObsTerm(func=bm.base_orientation_relative_to_ee) # Base orientation relative to EE (4D)
        target_pose_command = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=isaac_mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

# Alternative: Legacy observations for comparison
@configclass 
class ObservationsCfgLegacy:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        base_pos_w  = ObsTerm(func=bm.base_position_w)   
        base_quat_w = ObsTerm(func=bm.base_orientation_w)
        target_pose_command = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=isaac_mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
    init_grounded_pose = EventTerm(
        func=bm.simple_robot_reset_event,
        mode="reset",
        params={
            "base_xy": (0.0, 0.0),
            "base_z": 0.30,
            "base_rot": (1.0, 0.0, 0.0, 0.0),
            "zero_joints": False,   # let set_start_pose do the joints
        },
    )

    set_start_pose = EventTerm(        # <-- only pass joint_pos
        func=bm.set_start_pose,
        mode="reset",
        params={"joint_pos": {
            "joint1": -0.01047198,
            "joint2":  0.356047,
            "joint3":  0.0523599,
            "joint4":  0.764455,
            "joint5":  0.0,
            "joint6": -0.942478,
            "joint7":  0.0,
        }},
    )
    
    # Warmup event to keep robot stable during initial period
    robot_warmup = EventTerm(
        func=bm.robot_warmup_event,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # Check every step
        params={
            "warmup_time": 1.0,  # Disable actions for 1 second
        },
    )

    # DEBUG EVENTS - Temporarily enabled for debugging
    debug_print = EventTerm(
        func=bm.debug_print_bases_and_targets_world,
        mode="interval",
        interval_range_s=(2.0, 2.0),   # Print every 2 seconds
        params={
            "command_name": "target_pose",
            "body_name": "link_base",
            "first_n": 10,              # Print first 10 envs as requested
            "decimals": 3,
        },
    )

    # debug_goal_spawning = EventTerm(
    #     func=bm.debug_print_goal_relative_to_initial_base,
    #     mode="interval", 
    #     interval_range_s=(3.0, 3.0),   # Print every 3 seconds
    #     params={
    #         "command_name": "target_pose",
    #         "body_name": "link_base", 
    #         "first_n": 6,              # Print first 6 envs
    #         "decimals": 3,
    #     },
    # )


    enforce_min_dist = EventTerm(
    func=bm.push_target_outside_radius_event,
    mode="reset",
    params={
        "command_name": "target_pose",
        "base_body": "link_base",
        "min_radius": 0.20,
        "keep_z_at_base": True,
    },
)

    place_marker_at_cmd = EventTerm(
        func=bm.set_marker_to_command_at_base_height_event,
        mode="interval",
        interval_range_s=(0.2, 0.2),   # update after commands are sampled
        params={
            "marker_name": "target_pose_marker",
            "command_name": "target_pose",
            "base_body": "link_base",
            "z_offset": 0.0,
        },
    )


    debug_print_targets = EventTerm(
        func=bm.print_target_xy_debug,
        mode="interval",
        interval_range_s=(3.0, 3.0),   # print every 3 seconds 
        params={
            "command_name": "target_pose",
            "body_name": "link_base",
            "k": 10,                    # Print 10 envs as requested
            "decimals": 3,
            "success_radius": 0.015,    # 15mm radius
        },
    )
    #     },
    # )

    # Event removed - now handled by dwell_time_success reward


@configclass
class RewardsCfg:
    # 🎯 IMPROVED: EE-relative rewards (proper frame consistency!)
    position_tracking_fine = RewTerm(
        func=bm.position_tanh_xy_ee, weight=1.0,
        params={"std": 0.20, "command_name": "target_pose", "ee_body": "link_eef"},
    )
    
    position_progress_xy = RewTerm(
        func=bm.position_progress_xy_ee_relative, weight=0.8
    )

    # Motion quality (EE-relative with proper frame consistency)
    base_motion = RewTerm(
        func=bm.base_linear_motion_toward_goal_xy_ee, weight=0.5,
        params={"command_name": "target_pose", "ee_body": "link_eef"}
    )

    # Orientation (only when close) - using legacy for now
    orientation_align = RewTerm(
        func=bm.orientation_tanh_yaw, weight=0.3,
        params={"command_name": "target_pose", "std_deg": 10.0, "gate_with_distance": True, "gate_sigma": 0.15},
    )   

    # Essential safety constraints
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1e-3)
    
    base_upright_pen = RewTerm(
        func=bm.base_upright_penalty, weight=-10.0,
        params={"roll_limit": math.radians(5.0), "pitch_limit": math.radians(5.0)},
    )
    
    base_lift_pen = RewTerm(
        func=bm.base_lift_penalty, weight=-20.0,  # Increased penalty
        params={"z_nom": 0.32, "deadband": 0.002},  # Tighter tolerance, updated to 0.32m
    )
    
    # Additional vertical velocity penalty to discourage bouncing
    base_vertical_velocity_pen = RewTerm(
        func=bm.base_vertical_velocity_penalty, weight=-10.0
    )

    # 🎯 CORRECTED: EE-relative tracking + Base-relative spawning  
    dwell_time_success = RewTerm(
        func=bm.dwell_time_success_reward_ee_relative,  # Track using EE coords, spawn relative to initial base
        weight=1.0,
        params={
            "command_name": "target_pose",
            "radius": 0.070,  # 70mm (7.0cm)
            "bonus": 10.0,
            "dwell_time_s": 0.5,  # Must stay in target for 0.5 second
            "exit_penalty": -15.0,  # INCREASED penalty for falling out of position
            "staying_reward": 1.5,  # Small reward for staying (reduced from 2.0)
            "max_staying_time": 1.5,  # Force new target after 1 second
        },
    )

# Alternative: Legacy rewards for comparison
@configclass
class RewardsCfgLegacy:
    # Core navigation rewards - CLEANED UP
    position_tracking_fine = RewTerm(
        func=bm.position_command_error_tanh_xy, weight=1.0,
        params={"std": 0.20, "command_name": "target_pose"},
    )
    
    position_progress_xy = RewTerm(
        func=bm.position_progress_xy, weight=0.8
    )

    # Motion quality
    base_motion = RewTerm(
        func=bm.base_linear_motion_to_command_xy, weight=0.5,
        params={"command_name": "target_pose"}
    )

    # Orientation (only when close)
    orientation_align = RewTerm(
        func=bm.orientation_tanh_yaw, weight=0.3,
        params={"command_name": "target_pose", "std_deg": 10.0, "gate_with_distance": True, "gate_sigma": 0.15},
    )   

    # Essential safety constraints
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1e-3)
    
    base_upright_pen = RewTerm(
        func=bm.base_upright_penalty, weight=-2.0,
        params={"roll_limit": math.radians(5.0), "pitch_limit": math.radians(5.0)},
    )
    
    base_lift_pen = RewTerm(
        func=bm.base_lift_penalty, weight=-5.0,
        params={"z_nom": 0.30, "deadband": 0.005},
    )

    # Multi-goal success bonus (legacy version)
    dwell_time_success = RewTerm(
        func=bm.dwell_time_success_reward,
        weight=1.0,
        params={
            "command_name": "target_pose",
            "radius": 0.015,  # 15mm (1.5cm)
            "bonus": 10.0,
            "dwell_time_s": 0.5,  # Must stay in target for 0.5 second
            "exit_penalty": -2.0,  # Penalty for falling out of position
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    # Remove reached_xy termination for multi-goal episodes
    # anti_lift_tilt = DoneTerm(
    #     func=bm.terminate_if_tilted_or_lifted,
    #     params={
    #         "z_nom": 0.25,
    #         "z_max_delta": 0.2,                 # >10 cm above nominal ⇒ done
    #         "roll_max": math.radians(10.0),
    #         "pitch_max": math.radians(10.0),
    #     },
    # )
@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=isaac_mdp.modify_reward_weight,
                           params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 4500})
    #joint_vel   = CurrTerm(func=isaac_mdp.modify_reward_weight,
    #                       params={"term_name": "joint_vel_l2", "weight": -0.001, "num_steps": 4500})

# -----------------------------
# Env
# -----------------------------
@configclass
class BaseMoveEnvCfg(ManagerBasedRLEnvCfg):
    scene: BaseMoveSceneCfg = BaseMoveSceneCfg(num_envs=4096, env_spacing=4.0)
    episode_length_s = 7.0
    decimation = 2
    num_actions = 7
    num_observations = 35  # Updated for EE-relative observations: 7+7+3+4+7+7 (removed redundant ee_pos_w)
    num_states = 0

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.viewer.eye = [3.5, 3.5, 3.5]
        self.viewer.lookat = [0.0, 0.0, 0.5]
        self.viewer.render_interval = self.decimation
        self.sim.dt = 0.01

@configclass
class BaseMoveEnvCfg_PLAY(BaseMoveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 15.0
