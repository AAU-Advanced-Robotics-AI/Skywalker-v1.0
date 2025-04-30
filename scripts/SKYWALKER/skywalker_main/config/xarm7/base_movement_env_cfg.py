
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

# Dynamically add project root (one level above skywalker_main) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sim import GroundPlaneCfg, DomeLightCfg

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import random
import torch
import carb



from typing import Optional
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim import CuboidCfg, SphereCfg
from isaaclab.markers.config import FRAME_MARKER_CFG


from skywalker_main.skywalker_env import SkywalkerEnvCfg, SkywalkerSceneCfg
from skywalker_main import mdp
from skywalker_main.xarm7 import XARM7_CFG
from skywalker_main.fixed_cube import FixedCube

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip





@configclass
class BaseMoveSceneCfg(InteractiveSceneCfg):
    filter_collisions: bool = False
    """Scene config for base movement + grabbing."""
    # You can override or add custom assets here
    # Required fields
    num_envs: int = 1024
    env_spacing: float = 4.0




@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    # gripper_action: mdp.SurfaceGripperActionCfg = mdp.SurfaceGripperActionCfg(
    #     asset_name="robot",
    #     gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
    # )




@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.concatenate_terms = False

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass

    # Encourage the robot to move its base toward a target X,Y position
    robot_goal_tracking = RewTerm(
        func=mdp.robot_base_to_goal_distance,
        weight=10.0,
        params={"goal_pos": [0.8, 0.0]},  # right side of base
    )

    # # Reward for end-effector reaching object (Z offset applied inside the func)
    # reaching_object = RewTerm(
    #     func=mdp.object_ee_distance,
    #     weight=10,
    #     params={"std": 0.1},
    # )

    # self_grasp = RewTerm(
    #     func = mdp.self_grasp_penalty,
    #     weight = 5.0      # already negative inside
    # )


    # Reward for holding the object (surface gripper logic handles detection)

    # grab_cube = RewTerm(
    #         func=mdp.is_grasping_fixed_object,  # <- simple call now
    #         weight=2.0,
    #         params={}  # <- empty because function no longer expects object_cfg
    # )

    # Movement penalties
    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=1e-4,
    # )
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # # Penalize self-collision if detected
    # self_collision = RewTerm(
    #     func=mdp.self_collision_penalty,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_reached_goal = DoneTerm(
        func=mdp.robot_reached_goal,
        params={"goal_pos": [0.0, 0.0], "threshold": 0.05}
    )
    pass


@configclass
class CommandsCfg:
    pass
        # """Command terms for the MDP."""

        # object_pose = mdp.UniformPoseCommandCfg(
        #     asset_name="object",  # This should be the target object you're commanding
        #     body_name="FixedCube",   # This should match what’s in your scene config
        #     resampling_time_range=(0.0, 0.0),
        #     debug_vis=False,
        #     ranges=mdp.UniformPoseCommandCfg.Ranges(
        #         pos_x=(0.25, 0.55),
        #         pos_y=(-0.2, 0.2),
        #         pos_z=(0.2, 0.55),
        #         roll=(0.0, 0.0),
        #         pitch=(0, 0),
        #         yaw=(0, 0),
        #     ),
        # )


# mdp/utils.py  (or wherever you put it)
def clear_buffers(env, env_ids=None, **kwargs):
    """Flush per-episode caches such as the prev-action tensor."""
    mdp.reset_action_rate_cache()
    # …add other book-keeping if you need it …


@configclass
class EventCfg:
    reset_scene  = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_caches = EventTerm(func=clear_buffers, mode="reset")

@configclass
class CurriculumCfg:
    pass

@configclass
class BaseMoveEnvCfg(ManagerBasedRLEnvCfg):
    """Full config for base movement + grabbing env."""

    scene: SkywalkerSceneCfg = SkywalkerSceneCfg(num_envs=100, env_spacing=4.0)



    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        super().__post_init__()



        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.viewer.eye = (3.5, 3.5, 3.5)
        #self.sim.physx.use_gpu_scene_query = False       # <- important


        print(f"[DEBUG] Action terms registered: {self.actions.__dict__.keys()}")
        self.scene.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=GroundPlaneCfg()
        )


        # Set goal position here (X, Y)
        goal_xy = [0.8, 0.0]
        goal_z = 0.47  # same as robot base Z height

        # Sync goal for reward and termination terms
        self.rewards.robot_goal_tracking.params["goal_pos"] = goal_xy
        self.terminations.robot_reached_goal.params["goal_pos"] = goal_xy


        # ---------------------------------------------------------------------
        #  FIX: spawn the cube as a *static* rigid body (no velocity writes)
        # ---------------------------------------------------------------------
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/FixedCube",

        #     # ── initial pose ────────────────────────────────────────────────
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[10.0, 0.0, 2.70],          # far outside the work-space at reset
        #         rot=[1.0, 0.0, 0.0, 0.0],       #   (w, x, y, z) quaternion
        #     ),

        #     # ── geometry + physics ──────────────────────────────────────────
        #     spawn=sim_utils.CuboidCfg(
        #         size=[0.10, 0.10, 0.10],        # 10-cm cube

        #         # ------------- STATIC actor settings -------------
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=False,    # <= static  (NEVER gets velocity writes)
        #             disable_gravity=True,       # cube just “floats” unless you move it
        #         ),
        #         # PhysX ignores mass on static actors, but keep it 0 to be explicit
                

        #         # --------------------------------------------------
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.0, 0.0)),
        #     ),
        # )
        # # ---------------------------------------------------------------------




        # Set up robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
        )




        # Set up synced goal marker
        self.scene.goal_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalMarker",
            spawn=SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[*goal_xy, goal_z]),
        )

       # Gripper action
        # self.actions.gripper_action = mdp.SurfaceGripperActionCfg(
        #     asset_name="robot",
        #     gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
        # )
        # self.actions.gripper_action.apply_stabilization = False
        # self.actions.gripper_action.use_offset_for_attachment = True



        # EE frame config (frame visualizer)
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
        #     debug_vis=True,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.06],
        #                 rot=[0, 0, 0, 1],
        #             ),
        #         ),
        #     ],
        # )






@configclass
class BaseMoveCfg_PLAY(BaseMoveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        #self.observations.policy.enable_corruption = False


