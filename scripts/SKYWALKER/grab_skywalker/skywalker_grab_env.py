
from dataclasses import MISSING

import math

#from scripts.SKYWALKER.skywalker2.mdp.terminations import robot_reached_goal
#from scripts.SKYWALKER.grab_skywalker.mdp.rewards import goal_potential
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
from grab_skywalker.mdp.reset_goal_fixed import reset_goal_fixed as  _reset_goal_fixed



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

    dock_marker: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING

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

        cube1_grasped  = ObsTerm(func=mdp.is_cube1_grasped)
        cube2_grasped  = ObsTerm(func=mdp.is_cube2_grasped)
        cube2_relative = ObsTerm(func=mdp.cube2_relative_to_cube1)


        goal_delta      = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        gripper_contact = ObsTerm(func=mdp.gripper_closed)
        #grasp_flag      = ObsTerm(func=mdp.is_cube_grasped)
        cylinder_closed = ObsTerm(func=mdp.cylinder_closed)
        root_vel_xy     = ObsTerm(func=mdp.root_lin_vel_xy)


        # cube1_grasp_count = ObsTerm(func=mdp.cube1_grasp_count)
        # cube2_grasp_count = ObsTerm(func=mdp.cube2_grasp_count)


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
        func   = _reset_goal_fixed,   # now unambiguously the function
        mode   = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("goal_marker"),
            "pos":       (0.20, -0.25, 0.42),
        },
    )

    reset_dock_fixed = EventTerm(
        func   = _reset_goal_fixed,   # again, the function, not the class var
        mode   = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("dock_marker"),
            "pos":       (0.0, 0.4, 0.42),
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
    goal_potential1 = RewTerm(func=mdp.goal_potential, weight=0.0)

    # ------------------------------------------------------------------
    # 1) Grasp value – pays more for the cube nearer the goal
    # ------------------------------------------------------------------
    drag_cube1 = RewTerm(
        func   = mdp.drag_carry_reward,
        weight = 5,
        params = {
            "cube_cfg":  SceneEntityCfg("cube1"),
            "grip_term": "gripper_action",
            "goal_cfg":  SceneEntityCfg("dock_marker"),   # ← drag cube1 to dock
        },
    )
    drag_cube2 = RewTerm(
        func   = mdp.drag_carry_reward,
        weight = 0,
        params = {
            "cube_cfg":  SceneEntityCfg("cube2"),
            "grip_term": "gripper_action",
            "goal_cfg":  SceneEntityCfg("goal_marker"),   # ← drag cube1 to dock
        },
    )

    # ------------------------------------------------------------------
    # 2) Exploration lures  (fade out later via curriculum if desired)
    # ------------------------------------------------------------------
    mount_affinity_c1 = RewTerm(
        func   = mdp.mount_affinity, weight = 0.0,
        params = {"cube_cfg": SceneEntityCfg("cube1"),
                  "std" : 0.25,
                  "grip_term": "gripper_action"},
    )


    mount_affinity_c2 = RewTerm(
        func   = mdp.mount_affinity, weight = 0.0,
        params = {"cube_cfg": SceneEntityCfg("cube2"),
                  "std" : 0.35,
                  "grip_term": "gripper_action"},
    )

    # ── 4b) Dock gripper-1 onto cube2’s mounting point ───────────────────────────
    gripper1_docking_cube2 = RewTerm(
        func   = mdp.gripper1_docking_reward,
        weight = 0.0,  # curriculum will turn this on later
        params = {
            "cube_cfg":     SceneEntityCfg("cube2"),
            "grip_term":    "gripper_action",
            "dock_cfg":     SceneEntityCfg("dock_marker"),   # ⬅ use this!
            "std":          0.20,
            "tol":          0.08,
            "dock_tol":     0.1,      # or adjust to your scene scale
            "close_bonus":  1.0,
            "far_penalty": -0.4,
        },
    )

    # gripper_attempt_c2 = RewTerm(
    #     func = mdp.gripper_close_near_cube2,
    #     weight = 0,
    #     params = {
    #         "cube_cfg":  SceneEntityCfg("cube2"),
    #         "grip_term": "gripper_action",
    #         "tol":       0.05,
    #         "reward":    0.4,
    #         "reuse_delay": 60,
    #     }
    # )


    # ------------------------------------------------------------------
    # 3) Hand-off logic
    # ------------------------------------------------------------------
    # hold_far_cube_penalty = RewTerm(
    #     func   = mdp.hold_far_cube_penalty,
    #     weight = 0.0,                       # λ baked in helper
    #     params = {"reach_thresh": 0.5,
    #               "release_bonus": 3.0},
    hold_far_cube_penalty = RewTerm(
        func   = mdp.hold_far_cube_penalty,
        weight = 0.0,  # curriculum sets to 1.0 at step 13,000
        params = {
            "reach_thresh": 0.40,     # slightly tighter than 0.50
            "falloff":      0.30,     # keep falloff gentle
            "lam":          1.5,      # ↓ less harsh
            "release_bonus": 1.0,     # ↑ encourage smoother releases
        },
    )

    hold_far_cube_penalty_tight = RewTerm(
        func   = mdp.hold_far_cube_penalty,
        weight = 0.0,  # curriculum sets to 2.0 at step 30,000
        params = {
            "reach_thresh": 0.45,     # wider zone
            "falloff":      0.20,     # mild tapering
            "lam":          2.0,      # ↑ stronger than soft version
            "release_bonus": 0.75,    # balanced
        },
    )


    let_go_bonus = RewTerm(
        func   = mdp.gripper_open_near_marker,
        weight = 4.0,                       # scale as you like
        params = {
            "marker_cfg": SceneEntityCfg("dock_marker"),
            "grip_term":  "gripper_action",
            "tol":        0.15,
            "reward":     1.0,
            "reuse_delay":60,               # ~0.5 s lock-out
        },
    )


    # ------------------------------------------------------------------
    # 4) Dock gripper-2 on floor
    # ------------------------------------------------------------------
    dock_near_cube2 = RewTerm(
        func   = mdp.gripper2_docking_reward,
        weight = 2,
        params = {"target_cfg": SceneEntityCfg("dock_marker"),
                  "std": 0.15,

                  "tol": 0.08},
    )
    # dock_at_goal = RewTerm(
    #     func   = mdp.gripper2_docking_reward,
    #     weight = 10,
    #     params = {"target_cfg": SceneEntityCfg("goal_marker")},
    # )

    # ------------------------------------------------------------------
    # 5) Pose hygiene
    # ------------------------------------------------------------------
    #ee_alignment          = RewTerm(func=mdp.ee_cube_orientation_alignment, weight=1.0)
    #ee_approach_alignment = RewTerm(func=mdp.ee_approach_alignment_in_base, weight=0.8)

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
        weight=1.0,
        params={}
    )
    
    cylinder_grasp = RewTerm(
        func=mdp.cylinder_self_grasp_penalty,
        weight=2.0,
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
    """Curriculum for phased handoff from Cube 1 → Cube 2 + docking."""

    # ─── Phase A (0–8k): Cube 1 mastery ───
    grab1_strong    = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube1", "weight": 1.0, "num_steps": 0})
    drag1_on        = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube1", "weight": 1.0, "num_steps": 0})
    mount_c1_on     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "mount_affinity_c1", "weight": 0.5, "num_steps": 0})
    grab2_off       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube2", "weight": 0.0, "num_steps": 0})
    drag2_off       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube2", "weight": 0.0, "num_steps": 0})
    dock2_off       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "gripper1_docking_cube2", "weight": 0.0, "num_steps": 0})
    goal_pot_off    = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "goal_potential1", "weight": 0.0, "num_steps": 0})

    # ─── Phase B (8k–18k): Cube 2 intro, Cube 1 decays ───
    mount_c2_on     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "mount_affinity_c2", "weight": 1.0, "num_steps": 9000})
    grab2_on        = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube2", "weight": 5.0, "num_steps": 11000})
    drag2_on        = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube2", "weight": 5.0, "num_steps": 13000})
    let_go_on       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "let_go_bonus", "weight": 5.0, "num_steps": 12000})
    hold_far_soft   = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "hold_far_cube_penalty", "weight": 1.0, "num_steps": 13000})
    goal_pot_temp   = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "goal_potential1", "weight": 1.0, "num_steps": 14000})
    dock2_early     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "gripper1_docking_cube2", "weight": 3.0, "num_steps": 15000})

    grab1_low       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube1", "weight": 0.4, "num_steps": 16000})
    drag1_low       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube1", "weight": 0.4, "num_steps": 16000})
    grab1_off       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube1", "weight": 0.0, "num_steps": 30000})
    drag1_off       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube1", "weight": 0.0, "num_steps": 30000})
    goal_pot_off2   = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "goal_potential1", "weight": 1.5, "num_steps": 20000})
    mount_c2_mid     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "mount_affinity_c2", "weight": 2.0, "num_steps": 20000})

    # ─── Phase C (20k–30k): Cube 2 build-up ───
    grab2_boost     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube2", "weight": 7.0, "num_steps": 22000})
    drag2_boost     = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube2", "weight": 7.0, "num_steps": 23000})
    dock2_mid       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "gripper1_docking_cube2", "weight": 7.0, "num_steps": 24000})
    mount_c1_fade   = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "mount_affinity_c1", "weight": 0.1, "num_steps": 25000})

    # ─── Phase D (30k+): Full task + strict penalties ───
    dock2_full       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "gripper1_docking_cube2", "weight": 10.0, "num_steps": 30000})
    hold_far_strict  = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "hold_far_cube_penalty_tight", "weight": 2.0, "num_steps": 30000})
    wall_on          = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "wall_grasp", "weight": 6.0, "num_steps": 30000})
    cyl_on           = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "cylinder_grasp", "weight": 1.5, "num_steps": 30000})
    selfcol_on       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "self_collision", "weight": 1.2, "num_steps": 30000})
    mount_c2_boost   = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "mount_affinity_c2", "weight": 3.0, "num_steps": 30000})
    grab2_peak       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "grab_cube2", "weight": 12.0, "num_steps": 30000})
    drag2_peak       = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "drag_cube2", "weight": 12.0, "num_steps": 30000})
    dock2_bonus      = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "gripper1_docking_cube2", "weight": 20.0, "num_steps": 35000})
    goal_pot_on  = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "goal_potential1", "weight": 20, "num_steps": 50000})

    # ─── Always-on regularisation ───
    joint_vel_l2    = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": 1e-3, "num_steps": 0})
    action_rate_l2  = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": 1e-3, "num_steps": 0})




# @configclass
# class CurriculumCfg:
#     """4‐phase curriculum for Skywalker mount‐grasping."""

#     # ── Phase A (0 steps): only learn to grasp cube1 ───────────────────────────
#     grab1_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"grab_cube1", "weight":7.0, "num_steps":0},
#     )
#     drag1_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"drag_cube1", "weight":7.0, "num_steps":0},
#     )
#     goal_potential_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"goal_potential1", "weight":0.0, "num_steps":0},
#     )

#     grab2_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"grab_cube2", "weight":0.0, "num_steps":0},
#     )
#     mount_c1_affinity = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"mount_affinity_c1", "weight":5.0, "num_steps":0},
#     )
#     wall_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"wall_grasp", "weight":1.0, "num_steps":0},
#     )
#         # 2) Turn on gripper1_docking_cube2 with full strength
#     disable_gripper1_dock2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "gripper1_docking_cube2",
#             "weight":    0.0,    # same as in RewardsCfg
#             "num_steps": 0,
#         },
#     )
#     cyl_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"cylinder_grasp", "weight":1.0, "num_steps":0},
#     )
#     selfcol_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"self_collision", "weight":1.0, "num_steps":0},
#     )
#     timepen_off = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"time_step_penalty", "weight":0.0, "num_steps":0},
#     )

#     # ── Phase B (10 000 steps): enable drag_cube1 and cube2 lure ──────────────

#     grab1_down = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"grab_cube1", "weight":2, "num_steps":6000},
#     )
#     enable_dock2_soft = CurrTerm(
#     func   = mdp.modify_reward_weight,
#     params = {
#         "term_name": "gripper1_docking_cube2",
#         "weight":    4.0,     # gentle: far_penalty = −0.4 × 4 = −1.6 max
#         "num_steps": 4000,    # ramp over 2 k env-steps (≈ 2 k / 1024 episodes)
#     },
# )
#     mount_c1_affinity = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"mount_affinity_c1", "weight":3.0, "num_steps":4000},
#     )
#     grab2_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"grab_cube2", "weight":8.0, "num_steps":6000},
#     )
#     mount_c2_affinity = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"mount_affinity_c2", "weight":7.0, "num_steps":4000},
#     )
#     # ── Phase B tweaks (at 5 000 steps): sharpen the dock-zone penalty ──────────
#     # turn OFF the old, wide penalty
#     enable_hold_far = CurrTerm(
#         func = mdp.modify_reward_weight,
#         params = {
#             "term_name":"hold_far_cube_penalty",
#             "weight":    0.5,
#             "num_steps": 6000,
#         },
#     )
#     disable_hold_far = CurrTerm(
#         func = mdp.modify_reward_weight,
#         params = {
#             "term_name":"hold_far_cube_penalty",
#             "weight":    0.0,
#             "num_steps": 8000,
#         },
#     )
#     # turn ON the new, tight penalty
#     enable_hold_far_tight = CurrTerm(
#         func = mdp.modify_reward_weight,
#         params = {
#             "term_name":"hold_far_cube_penalty_tight",
#             "weight":    2.0,       # choose whatever max penalty you prefer
#             "num_steps": 8000,
#         },
#     )

#     # ── Phase C (20 000 steps): ramp in wall & base/collision penalties ──────
#     wall_ramp = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"wall_grasp", "weight":17, "num_steps":6000},
#     )
#     cyl_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"cylinder_grasp", "weight":1.0, "num_steps":6000},
#     )
#     selfcol_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"self_collision", "weight":1.0, "num_steps":6000},
#     )
#     fade_mount_c2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "mount_affinity_c2",
#             "weight":    6.0,
#             "num_steps": 6000,
#         },
#     )
#     # 2) Turn on gripper1_docking_cube2 with full strength
#     enable_gripper1_dock2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "gripper1_docking_cube2",
#             "weight":    10.0,    # same as in RewardsCfg
#             "num_steps": 6000,
#         },
#     )

#     dock_near_cube2_low = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "dock_near_cube2",
#             "weight":    3.0,    # same as in RewardsCfg
#             "num_steps": 8000,
#         },
#     )
#     fade_mount_c2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "mount_affinity_c2",
#             "weight":    6.0,
#             "num_steps": 8000,
#         },
#     )

#     grab1_down = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"grab_cube1", "weight":1.5, "num_steps":8000},
#     )
#     mount_c1_affinity = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"mount_affinity_c1", "weight":1.0, "num_steps":6000},
#     )
#     drag1_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"drag_cube1", "weight":1.5, "num_steps":6000},
#     )
#     # grab1_down2 = CurrTerm(
#     #     func=mdp.modify_reward_weight,
#     #     params={"term_name":"grab_cube1", "weight":0.1, "num_steps":8000},
#     # )
#     # ── Phase D (30 000 steps+): full‐task weights & time‐penalty ─────────────
#     wall_full = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"wall_grasp", "weight":7.0, "num_steps":13_000},
#     )
#     cyl_full = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"cylinder_grasp", "weight":1.5, "num_steps":13_000},
#     )
#     selfcol_full = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"self_collision", "weight":1.2, "num_steps":13_000},
#     )

#     enable_mount_c2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "mount_affinity_c2",
#             "weight":   6,
#             "num_steps": 13000,
#         },
#     )

#     timepen_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"time_step_penalty", "weight":0.01, "num_steps":13_000},
#     )
#     goal_potential_on = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"goal_potential1", "weight":5.0, "num_steps":13000},
#     )


#     # ── Always‐on smoothing terms ─────────────────────────────────────────────
#     joint_vel_l2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"joint_vel", "weight":1e-3, "num_steps":0},
#     )
#     action_rate_l2 = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name":"action_rate", "weight":1e-3, "num_steps":0},
#     )



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



  