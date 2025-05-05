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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg 
from isaaclab.sim import SimulationContext
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import random
import torch

from typing import Optional

from xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG

from grab_skywalker.mdp.reset_goal_away_from_origin import reset_goal_away_from_origin



@configclass
class SkywalkerGrabSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    
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
    object: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING
    
    #wall: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING
    
    goal_marker: AssetBaseCfg | RigidObjectCfg | DeformableObjectCfg = MISSING

    cube: FrameTransformerCfg = MISSING
    



@configclass
class CommandsCfg:
    """Command terms for the MDP."""


pass

@configclass
class ActionsCfg:
    """Action specifications for the environment."""



    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    
    gripper_action: mdp.SurfaceGripperActionCfg = MISSING






@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        ee_in_base = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)


        def __post_init__(self) -> None:
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

    reset_goal = EventTerm(
        func=reset_goal_away_from_origin,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("goal_marker"),
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.45, 0.45),
                "z": (0.47, 0.47),
            },
            "velocity_range": {},  # still required, even if unused here
            "min_radius": 0.3,
        },
    )


    # reset_goal = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("goal_marker"),
    #         "pose_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (0.47, 0.47),
    #         },
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

@configclass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
   # Encourage the robot to move its base toward a target X,Y position
    robot_goal_tracking = RewTerm(
        func=mdp.robot_base_to_goal_distance,
        weight=0.0,
        params={}, 
    )

    # Reward for end-effector reaching object (Z offset applied inside the func)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=10,
        params={"std": 0.1},
    )

    self_grasp = RewTerm(
        func = mdp.self_grasp_penalty,
        weight = 5.0     # already negative inside
    )

    ee_alignment = RewTerm(
        func=mdp.ee_cube_orientation_alignment,
        weight=2.0  # Start small, increase if needed
    )

    ee_approach_alignment = RewTerm(
        func=mdp.ee_approach_alignment_in_base,
        weight=2.0,  # You can tune this weight
    )


    grab_cube = RewTerm(
            func=mdp.is_grasping_fixed_object,  # <- simple call now
            weight=5.0,
            params={}  # <- empty because function no longer expects object_cfg
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=1e-5,
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalize self-collision if detected
    self_collision = RewTerm(
        func=mdp.self_collision_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_reached_goal = DoneTerm(
        func=mdp.robot_reached_goal,
        params={"threshold": 0.05}  # Remove goal_pos param
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )
    self_grasp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "self_grasp", "weight": 5.0, "num_steps": 5000}
    )

    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "reaching_object", "weight": 3.0, "num_steps": 5000}
    )

    robot_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "robot_goal_tracking", "weight": 10.0, "num_steps": 5000}
    )





@configclass
class GrabEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    # Scene settings
    scene: SkywalkerGrabSceneCfg = SkywalkerGrabSceneCfg(num_envs=1024, env_spacing=4.0)
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



  