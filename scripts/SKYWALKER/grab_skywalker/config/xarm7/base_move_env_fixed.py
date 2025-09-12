# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

print(f"[BaseMoveEnvCfg] Using base_move_env_cfg.py from: {__file__}")

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
class BaseMoveSceneCfg(InteractiveSceneCfg):
    """Scene: Free base, EE welded to a wall anchor."""

    # Robot — BASE IS FREE here
    robot = XARM7_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),  # Robot on ground
        spawn=XARM7_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=False,  # <-- key difference vs. PTP
            ),
        ),
    )

    # (Optional) EE frame — handy for debugging even if EE is welded
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

    # Ground plane with very low friction so the base can slide easily
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1e-4,
                dynamic_friction=1e-4,
                restitution=0.0,
            ),
        ),
        collision_group=-1,
    )

    # A kinematic wall in front of the robot
    wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.04, 2.0),  # wide, thin, tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.80, 0.0, 0.90),  # Move wall further back from robot
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity rotation
        ),
    )

    # Small kinematic pad to weld the EE to  (NOTE: not a child of /Wall!)
    EE_Anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EE_Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.04, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Position at expected XArm7 neutral EE pose (base at 0,0,0.25 -> EE around 0.6,0,0.6)
            pos=(0.60, 0.0, 0.60),  # Better estimate for XArm7 neutral EE position  
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity rotation - no 90 degree twist
        ),
    )

    # Visual target marker for shaping terms (bm.*)
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_marker",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.9, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Lights (keep yours like in PTP if you want)
    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim_utils.DistantLightCfg(intensity=5000.0, angle=0.5, color=(1.0, 1.0, 1.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0), visible_in_primary_ray=True),
    )
    main_light = AssetBaseCfg(
        prim_path="/World/mainLight",
        spawn=sim_utils.SphereLightCfg(intensity=4000.0, radius=1.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 2.0, 3.0)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/fillLight",
        spawn=sim_utils.SphereLightCfg(intensity=2000.0, radius=0.5, color=(0.95, 0.95, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.5, -1.5, 2.0)),
    )

# ---------------------------------------------------------------------------
# MDP: Commands / Actions / Observations / Rewards / Terminations / Curriculum
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Command the BASE pose (not EE). Keep z narrow around your base height.
    target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_base",              # <-- key: command for the base
        resampling_time_range=(7.0, 7.0),   # match episode length
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.6),
            pos_y=(-0.4, 0.4),
            pos_z=(0.20, 0.30),            # narrow band around your base height in USD
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-0.4, 0.4),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint.*"],
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the policy (kept at 35-D like PTP)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 7 + 7
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)

        # 3 + 4 (base world pose)
        base_pos_w  = ObsTerm(func=bm.base_position_w)
        base_quat_w = ObsTerm(func=bm.base_orientation_w)

        # 7 (x,y,z,qw,qx,qy,qz) — command pose for the base
        target_pose_command = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "target_pose"},
        )

        # 7 (last action)
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
    # Create the weld BEFORE physics starts (no snap) - simple approach
    weld_on_scene_init = EventTerm(
        func=bm.create_weld_on_scene_init,
        mode="scene_init",
        params={
            "ee_rel_path": "Robot/xarm7/link_eef",
            "anchor_name": "EE_Anchor",  # must match your scene key
            "joint_name": "EE_Weld",
        },
    )

    # Simple reset without weld recreation
    init_grounded_pose = EventTerm(
        func=bm.ground_upright_snap_anchor_no_weld_event,  # No weld recreation
        mode="reset",
        params={
            "base_xy": (0.0, 0.0),
            "base_z": 0.25,
            "wall_thickness": 0.04,
            "ee_rel_path": "Robot/xarm7/link_eef",
            "anchor_name": "EE_Anchor",
            "wall_name": "wall",
        },
    )

@configclass
class RewardsCfg:
    """Reward terms."""
    # Primary: base position command tracking (reuse generic reach terms on link_base)
    position_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-0.25,  # negative weight on error
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"),
                "command_name": "target_pose"},
    )

    # Smooth shaping near target
    position_tracking_fine = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"),
                "std": 0.1,
                "command_name": "target_pose"},
    )

    # Orientation tracking (small weight, base yaw mostly)
    orientation_tracking = RewTerm(
        func=reach_mdp.orientation_command_error,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="link_base"),
                "command_name": "target_pose"},
    )

    # Directional motion toward the target + stillness when close
    base_motion = RewTerm(
        func=bm.base_linear_motion_reward,
        weight=0.4,
        params={"target_cfg": SceneEntityCfg("target_marker")},  # optional if you also drop a marker
    )
    base_stability = RewTerm(
        func=bm.base_stability_reward,
        weight=0.4,
        params={"target_cfg": SceneEntityCfg("target_marker")},
    )
    hover_penalty = RewTerm(
        func=bm.base_hovering_penalty,
        weight=-0.2,
        params={"target_cfg": SceneEntityCfg("target_marker")},
    )

    # Action/joint smoothness
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1e-4)
    joint_vel_l2   = RewTerm(
        func=isaac_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class TerminationsCfg:
    """Termination terms."""
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    # If you want success termination, uncomment:
    # success = DoneTerm(
    #     func=bm.base_target_reached,
    #     params={"target_cfg": SceneEntityCfg("target_marker"), "threshold": 0.05},
    # )

@configclass
class CurriculumCfg:
    """Simple curriculum like your PTP setup."""
    action_rate = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 4500},
    )
    joint_vel = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "joint_vel_l2", "weight": -0.001, "num_steps": 4500},
    )

# ---------------------------------------------------------------------------
# Env config (PTP-style)
# ---------------------------------------------------------------------------

@configclass
class BaseMoveEnvCfg(ManagerBasedRLEnvCfg):
    """EE welded; base moves towards commanded base pose."""

    # Scene
    scene: BaseMoveSceneCfg = BaseMoveSceneCfg(num_envs=4096, env_spacing=2.0)

    # Basics
    episode_length_s = 7.0
    decimation = 2
    num_actions = 7
    # 7(q) + 7(dq) + 3(base p) + 4(base q) + 7(cmd) + 7(last a) = 35
    num_observations = 35
    num_states = 0

    # MDP
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
        self.sim.dt = 0.01  # 100 Hz

@configclass
class BaseMoveEnvCfg_PLAY(BaseMoveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 15.0
