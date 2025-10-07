# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
# parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# print("I am alive")
# simulation_app = app_launcher.app

# """Rest everything follows."""

from dataclasses import MISSING

import math
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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg, SurfaceGripperCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg 
from isaaclab.sim import SimulationContext
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import random
import torch

from typing import Optional

from .xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG

#from xarm7_scene import SkywalkerSceneCfg

@configclass
class SkywalkerGrabSceneCfg(InteractiveSceneCfg):
    

    
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

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
    object: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg | ArticulationCfg = MISSING
    
    # assembly wall with multiple components
    assembly_wall: AssetBaseCfg = MISSING
    
    goal_marker: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING

    dock_marker: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING

    cube1: FrameTransformerCfg = MISSING

    cube2: FrameTransformerCfg = MISSING

    cube3: FrameTransformerCfg = MISSING

    floor: FrameTransformerCfg = MISSING

    cylinder_frame: FrameTransformerCfg = MISSING

    cylinder_SG_frame: FrameTransformerCfg = MISSING

    EE_SG_frame: FrameTransformerCfg = MISSING

    # Note: Surface grippers are NOT declared as scene assets when they are part of a robot articulation
    # They are automatically detected by Isaac Lab when using SurfaceGripperActionTerm with the robot
    
    # Surface grippers from robot USD - need to be wrapped as Isaac Lab assets
    ee_gripper: SurfaceGripperCfg = SurfaceGripperCfg(
        prim_expr="{ENV_REGEX_NS}/Robot/xarm7/EE_SG/SurfaceGripper_EE",
        max_grip_distance=0.1,
        coaxial_force_limit=500.0,
        shear_force_limit=500.0,
        retry_interval=0.1,
    )
    
    cylinder_gripper: SurfaceGripperCfg = SurfaceGripperCfg(
        prim_expr="{ENV_REGEX_NS}/Robot/xarm7/Cylinder_SG/SurfaceGripper_Cylinder", 
        max_grip_distance=0.1,
        coaxial_force_limit=500.0,
        shear_force_limit=500.0,
        retry_interval=0.1,
    )
    



@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="link_eef",
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.25, 0.55),
    #         pos_y=(-0.2, 0.2),
    #         pos_z=(0.2, 0.55),
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi, math.pi), 
    #         yaw=(-3.14, 3.14),
    #     ),
    # )
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.25, 0.55),
                pos_y=(-0.2, 0.2),
                pos_z=(0.2, 0.55),
                roll=(0.0, 0.0),
                pitch=(0, 0), 
                yaw=(0, 0),
            ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # joint_efforts = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["joint.*"],
    #     #joint_names=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"],
    #     scale=5.0, # dont quite understand what scale does yet
    #     use_default_offset=True
    # )

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.SurfaceGripperActionCfg = MISSING
    gripper_action2: mdp.SurfaceGripperActionCfg = MISSING






@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ── proprioception ───────────────────────────────────────────────
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action   = ObsTerm(func=mdp.last_action)

        # (optional) raw EE position in base frame
        # ee_in_base  = ObsTerm(func=mdp.ee_position_in_robot_root_frame)

        # ── NEW minimalist task signals ─────────────────────────────────
        progress      = ObsTerm(func=mdp.progress_delta)
        ee_to_cube    = ObsTerm(func=mdp.ee_to_cube_vec)

        # ── manager flags ───────────────────────────────────────────────
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

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

    reset_dock_fixed = EventTerm(
        func   = mdp.reset_goal_fixed,   # again, the function, not the class var
        mode   = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("dock_marker"),
            "pos":       (0.0, 0.4, 0.43),
        },
    )

    reset_goal_fixed = EventTerm(
        func   = mdp.reset_goal_fixed,   # again, the function, not the class var
        mode   = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("goal_marker"),
            "pos":       (0.0, -0.25, 0.43),
        },
    )




@configclass
class RewardsCfg:
    """Minimal, PPO-friendly reward slate."""

    # ── Task rewards ───────────────────────────────────────────────
    forward_progress = RewTerm(
        func   = mdp.forward_progress,     # +ΔX each step
        weight = 1.0,
    )
    task = RewTerm(func=mdp.skywalker_reward, weight=4)

    Done = RewTerm(func=mdp.robot_reached_goal, weight=10.0)
    

    # ── Safety / smoothness (kept) ─────────────────────────────────
    sim_grab_penalty = RewTerm(func=mdp.simultaneous_gripper_penalty, weight=3.0)
    joint_vel        = RewTerm(func=mdp.joint_vel_l2,                weight=1e-3)
    action_rate      = RewTerm(func=mdp.action_rate_l2,              weight=1e-4)
    self_collision   = RewTerm(func=mdp.self_collision_penalty,      weight=1.0)
    cylinder_grasp   = RewTerm(func=mdp.cylinder_self_grasp_penalty, weight=1.0)
    wall_grasp       = RewTerm(func=mdp.wall_proximity_penalty,      weight=1.0)

    # ── Stability rewards for surface grippers ──────────────────────
    cylinder_stability = RewTerm(func=mdp.cylinder_stability_reward, weight=2.0)
    anti_slip = RewTerm(func=mdp.anti_slip_reward, weight=1.0)

    # small step cost: mdp.time_step_penalty returns **-1.0**
    step_penalty     = RewTerm(func=mdp.time_step_penalty,           weight=-0.01)

# ----------------------------------------------------------------------
#  Curriculum: progressive weight-schedules
# ----------------------------------------------------------------------
@configclass
class CurriculumCfg:
    pass




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.object_reached_goal)


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

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 0.01

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

# def main():
#     """Main function."""
#     # parse the arguments
#     env_cfg = SkywalkerEnvCfg()
#     env_cfg.scene.num_envs = args_cli.num_envs
#     # setup base environment
#     env = ManagerBasedRLEnv(cfg=env_cfg)

#     # simulate physics
#     count = 0
#     while simulation_app.is_running():
#         with torch.inference_mode():
#             # reset
#             if count % 300 == 0:
#                 count = 0
#                 env.reset()
#                 print("-" * 80)
#                 print("[INFO]: Resetting environment...")
#             # sample random actions
#             joint_efforts = torch.randn_like(env.action_manager.action)
#             # step the environment
#             obs, rew, terminated, truncated, info = env.step(joint_efforts)
#             # print current orientation of pole
#             print("Full observation for policy:", obs["policy"][0])
#             #print("[Env 0]: Joints ", obs["policy"][0][1].item())

#             # update counter
#             count += 1

#     # close the environment
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()
