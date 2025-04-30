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

    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(1, 1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, kinematic_enabled=True, enable_gyroscopic_forces=True, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 0.55)),
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
        debug_vis=False,
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

    # gripper_action: mdp.SurfaceGripperActionCfg = mdp.SurfaceGripperActionCfg(
    #                                                 asset_name="robot",
    #                                                 surface_grippers=SkywalkerGrabSceneCfg.surface_grippers,
    #                                             )


    # Initialize Gripper Action
    # gripper_action = mdp.GripperImpulseAction()

    # # Example: RL Model Outputs an Action Value
    # action_value = policy_output_from_rl_model  # Positive -> Open, Negative/Zero -> Close

    # # Apply the Gripper Action
    # gripper_action.apply(action_value)




@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        #ose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"}) # STILL TO UNDERSTAND
        actions = ObsTerm(func=mdp.last_action)

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
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    #alive = RewTerm(func=mdp.is_alive, weight=-0.01)
    # terminating = RewTerm(func=mdp.is_terminated, weight=25.0)
    # task terms
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.66}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.66, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.66, "command_name": "object_pose"},
        weight=5.0,
    )


    ee_orientation_alignment = RewTerm(
        func=mdp.object_ee_orientation_alignment,
        weight=3.0,
    )


    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #success = DoneTerm(func=mdp.position_command_error_bonus_terminate,params={"asset_cfg": SceneEntityCfg("robot", body_names="link_eef"), "command_name": "ee_pose"})
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.40, "asset_cfg": SceneEntityCfg("object")}
    )
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
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
