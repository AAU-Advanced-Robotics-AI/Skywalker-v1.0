
from dataclasses import MISSING

import math

#from scripts.SKYWALKER.skywalker2.mdp.terminations import robot_reached_goal
from sympy import Q
import torch

#import reach_skywalker.mdp as mdp
import grab_skywalker.mdp as mdp
#import mdp
#import isaaclab.envs.mdp as mdp
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

import isaacsim.core.utils.prims as prim_utils


from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg 
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import random
import torch




from typing import Optional

from xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG

from grab_skywalker.mdp.reset_goal_away_from_origin import reset_goal_within_reach
from grab_skywalker.mdp.reset_goal_fixed import reset_goal_fixed



@configclass
class SkywalkerGrabSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # static_friction=0.1,
            # dynamic_friction=0.1,
            # # ensure combine with any other material yields zero
            # friction_combine_mode="min",
        )
        ))
 

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot defined as an articulation
    robot : ArticulationCfg = MISSING
    # robot: ArticulationCfg = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
    #                                 init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.47)) # Dont know why this offset make it start on the floor
    #                             )
    
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING
    
    #wall: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING
    
    goal_marker: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING

    cube1: FrameTransformerCfg = MISSING

    cube2: FrameTransformerCfg = MISSING

    cube3: FrameTransformerCfg = MISSING

    cylinder_frame: FrameTransformerCfg = MISSING
    



@configclass
class CommandsCfg:
    """Command terms for the MDP."""


pass

@configclass
class ActionsCfg:
    """Action specifications for the environment."""



    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    
    gripper_action: mdp.SurfaceGripperActionCfg = MISSING
    gripper_action2: mdp.SurfaceGripperActionCfg = MISSING






@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    
    @configclass
    class PolicyCfg(ObsGroup):
        ee_in_base      = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        joint_pos_rel   = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel   = ObsTerm(func=mdp.joint_vel_rel)
        last_action     = ObsTerm(func=mdp.last_action)

        cube1_pos       = ObsTerm(
                            func=mdp.cube_position_in_robot_root_frame,
                            params={"cube_cfg": SceneEntityCfg("cube1")}
                        )
        cube2_pos       = ObsTerm(
                            func=mdp.cube_position_in_robot_root_frame,
                            params={"cube_cfg": SceneEntityCfg("cube2")}
                        )

        ee_to_c1    = ObsTerm(func=mdp.ee_to_cube1_distance
                                    )
        ee_to_c2    = ObsTerm(func=mdp.ee_to_cube2_distance
                                    )
        time_frac   = ObsTerm(func=mdp.time_fraction)

        goal_delta      = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        gripper_contact = ObsTerm(func=mdp.gripper_closed)
        #grasp_flag      = ObsTerm(func=mdp.is_cube_grasped)
        cylinder_closed = ObsTerm(func=mdp.cylinder_closed)
        root_vel_xy     = ObsTerm(func=mdp.root_lin_vel_xy)

        def __post_init__(self):
            self.enable_corruption   = False
            self.concatenate_terms   = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # on reset
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
        )
    reset_arm_position = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]),
            "position_range": (0.5, 1.5),
            "velocity_range": (0, 0),
        },
    )

    reset_goal_fixed = EventTerm(
        func   = reset_goal_fixed,
        mode   = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("goal_marker"),
            "pos": (0.30, -0.45, 0.40),   # X, Y, Z in world frame
            #"yaw": 0.0,                  # facing forward
        },
    )
    


    # reset_goal = EventTerm(
    #     func=reset_goal_within_reach,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("goal_marker"),
    #         "EEL": 0.068,   # adjust based on your robot
    #         "LA": 0.70,
    #         "HW": 0.47,
    #         "HR": 0.26+0.337,
    #         "RR": 0.7/2,
    #         "z": 0.40,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

        # 1a) Always-on reward for _any_ successful grasp of cube1
    grab_cube1 = RewTerm(
        func   = mdp.is_grasping_fixed_object,
        weight = 4,   # tune this up so that a single grasp gives ∼1–2 reward/step
        params = {"cube_cfg": SceneEntityCfg("cube1"),
                  "grip_term": "gripper_action"},
    )

    # 1b) And similarly for cube2 if you like
    grab_cube2 = RewTerm(
        func   = mdp.is_grasping_fixed_object,
        weight = 4,
        params = {"cube_cfg": SceneEntityCfg("cube2"),
                  "grip_term": "gripper_action"},
    )

    # ------------------------------------------------------------------
    # 0) Potential shaping  (navigation progress signal)
    # ------------------------------------------------------------------
    goal_potential = RewTerm(func=mdp.goal_potential, weight=3.0)

    # ------------------------------------------------------------------
    # 1) Grasp value – pays more for the cube nearer the goal
    # ------------------------------------------------------------------
    drag_cube1 = RewTerm(
        func   = mdp.drag_carry_reward,
        weight = 2,
        params = {"cube_cfg": SceneEntityCfg("cube1"),
                  "grip_term": "gripper_action"},
    )
    drag_cube2 = RewTerm(
        func   = mdp.drag_carry_reward,
        weight = 2,
        params = {"cube_cfg": SceneEntityCfg("cube2"),
                  "grip_term": "gripper_action"},
    )

    # ------------------------------------------------------------------
    # 2) Exploration lures  (fade out later via curriculum if desired)
    # ------------------------------------------------------------------
    mount_affinity_c1 = RewTerm(
        func   = mdp.mount_affinity, weight = 5.0,
        params = {"cube_cfg": SceneEntityCfg("cube1"),
                  "grip_term": "gripper_action"},
    )


    mount_affinity_c2 = RewTerm(
        func   = mdp.mount_affinity, weight = 5.0,
        params = {"cube_cfg": SceneEntityCfg("cube2"),
                  "grip_term": "gripper_action"},
    )

    # ------------------------------------------------------------------
    # 3) Hand-off logic
    # ------------------------------------------------------------------
    hold_far_cube_penalty = RewTerm(
        func   = mdp.hold_far_cube_penalty,
        weight = 2.0,                       # λ baked in helper
        params = {"reach_thresh": 0.5,
                  "release_bonus": 2.0},
    )

    # ------------------------------------------------------------------
    # 4) Dock gripper-2 on floor
    # ------------------------------------------------------------------
    dock_near_cube2 = RewTerm(
        func   = mdp.gripper2_docking_reward,
        weight = 5,
        params = {"target_cfg": SceneEntityCfg("cube2")},
    )
    # dock_at_goal = RewTerm(
    #     func   = mdp.gripper2_docking_reward,
    #     weight = 10,
    #     params = {"target_cfg": SceneEntityCfg("goal_marker")},
    # )

    # ------------------------------------------------------------------
    # 5) Pose hygiene
    # ------------------------------------------------------------------
    ee_alignment          = RewTerm(func=mdp.ee_cube_orientation_alignment, weight=1.0)
    ee_approach_alignment = RewTerm(func=mdp.ee_approach_alignment_in_base, weight=0.8)

    # ------------------------------------------------------------------
    # 6) Gripper-state discipline
    # ------------------------------------------------------------------
    sim_grab_penalty = RewTerm(  # both closed (after grace)
        func   = mdp.simultaneous_gripper_penalty,
        weight = 2.0,
    )

    # ------------------------------------------------------------------
    # 7) Smoothness / safety regularisers
    # ------------------------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2,   weight=1e-3)
    joint_vel   = RewTerm(func=mdp.joint_vel_l2,     weight=1e-3,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    
    self_collision = RewTerm(func=mdp.self_collision_penalty,
                             weight=1.0,
                             params={"asset_cfg": SceneEntityCfg("robot")})




    wall_grasp = RewTerm(
        func=mdp.wall_proximity_penalty,
        weight=0.5,
        params={}
    )
    
    cylinder_grasp = RewTerm(
        func=mdp.cylinder_self_grasp_penalty,
        weight=1.0,
        params={}
    )


    robot_fixed_in_goal = RewTerm(func=mdp.is_gripper2_closed_around_goal, weight=20.0, params={"tol": 0.1})

    time_step_penalty = RewTerm(
        func   = mdp.time_step_penalty,
        weight = -0.01,                     # small negative per step
        params = {},                        # no extra args
    )

# ----------------------------------------------------------------------
#  Curriculum: progressive weight-schedules
# ----------------------------------------------------------------------
@configclass
class CurriculumCfg:
    """4‐phase curriculum for Skywalker mount‐grasping."""

    # ── Phase A (0 steps): only learn to grasp cube1 ───────────────────────────
    grab1_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"grab_cube1", "weight":6.0, "num_steps":0},
    )
    drag1_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"drag_cube1", "weight":0.0, "num_steps":0},
    )
    grab2_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"grab_cube2", "weight":0.0, "num_steps":0},
    )
    mount_c1_affinity = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"mount_affinity_c1", "weight":5.0, "num_steps":0},
    )
    wall_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"wall_grasp", "weight":1.0, "num_steps":0},
    )
    cyl_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"cylinder_grasp", "weight":1.0, "num_steps":0},
    )
    selfcol_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"self_collision", "weight":1.0, "num_steps":0},
    )
    timepen_off = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"time_step_penalty", "weight":0.0, "num_steps":0},
    )

    # ── Phase B (10 000 steps): enable drag_cube1 and cube2 lure ──────────────
    drag1_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"drag_cube1", "weight":2.0, "num_steps":3_000},
    )
    grab2_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"grab_cube2", "weight":5.0, "num_steps":3_000},
    )
    mount_c2_affinity = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"mount_affinity_c2", "weight":5.0, "num_steps":3_000},
    )

    # ── Phase C (20 000 steps): ramp in wall & base/collision penalties ──────
    wall_ramp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"wall_grasp", "weight":0.5, "num_steps":5000},
    )
    cyl_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"cylinder_grasp", "weight":1.0, "num_steps":5000},
    )
    selfcol_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"self_collision", "weight":1.0, "num_steps":5000},
    )

    # ── Phase D (30 000 steps+): full‐task weights & time‐penalty ─────────────
    wall_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"wall_grasp", "weight":1.0, "num_steps":10_000},
    )
    cyl_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"cylinder_grasp", "weight":0.5, "num_steps":10_000},
    )
    selfcol_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"self_collision", "weight":0.2, "num_steps":10_000},
    )
    timepen_on = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"time_step_penalty", "weight":0.01, "num_steps":10_000},
    )

    # ── Always‐on smoothing terms ─────────────────────────────────────────────
    joint_vel_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"joint_vel", "weight":1e-3, "num_steps":0},
    )
    action_rate_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name":"action_rate", "weight":1e-3, "num_steps":0},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # robot_reached_goal = DoneTerm(
    #     func=mdp.robot_reached_goal,
    #     params={"threshold": 0.05}  # Remove goal_pos param
    # )
    robot_fixed_in_goal = DoneTerm(func=mdp.is_gripper2_closed_around_goal, params={"tol": 0.1})

@configclass
class GrabEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    # Scene settings
    scene: SkywalkerGrabSceneCfg = SkywalkerGrabSceneCfg(num_envs=1024, env_spacing=6.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    sim: SimulationCfg = SimulationCfg(
            #device="cuda:0",
            #dt=1 / 120,
            #gravity=(0.0, 0.0, -9.81),
            # physx=PhysxCfg(
            #     solver_type=1,
            #     max_position_iteration_count=192,  # Important to avoid interpenetration.
            #     max_velocity_iteration_count=1,
            #     bounce_threshold_velocity=0.2,
            #     friction_offset_threshold=0.01,
            #     friction_correlation_distance=0.00625,
            #     gpu_max_rigid_contact_count=2**23,
            #     gpu_max_rigid_patch_count=2**23,
            #     gpu_max_num_partitions=1,  # Important for stable simulation.
            # ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
            ),
        )
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 7.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 0.01



        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625



  