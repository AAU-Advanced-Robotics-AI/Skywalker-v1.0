
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
        """Observations for policy group."""
        ee_in_base = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        gripper_contact = ObsTerm(func=mdp.gripper_contact_state)
        goal_delta = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        root_vel_xy   = ObsTerm(func=mdp.root_lin_vel_xy)
        grasp_flag    = ObsTerm(func=mdp.is_cube_grasped)
        cylinder_closed     = ObsTerm(func=mdp.cylinder_closed)




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
        func=reset_goal_within_reach,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("goal_marker"),
            "EEL": 0.068,   # adjust based on your robot
            "LA": 0.70,
            "HW": 0.47,
            "HR": 0.26+0.337,
            "RR": 0.7/2,
            "z": 0.40,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    robot_goal_tracking = RewTerm(
        func=mdp.robot_base_to_goal_distance,
        weight=0.2,
        params={}, 
    )

    robot_goal_tracking_fine = RewTerm(
        func=mdp.robot_base_to_goal_distance_fine,
        weight=1.0,
        params={}, 
    )

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=5,
        params={"std": 0.1},
    )

    self_grasp = RewTerm(
        func=mdp.self_grasp_penalty,
        weight=1.0,
        params={}
    )

    ee_alignment = RewTerm(
        func=mdp.ee_cube_orientation_alignment,
        weight=1.5,
        params={}
    )

    ee_approach_alignment = RewTerm(
        func=mdp.ee_approach_alignment_in_base,
        weight=1.2,
        params={}
    )

    grab_cube = RewTerm(
        func=mdp.is_grasping_fixed_object,
        weight=3.0,
        params={}
    )

    sim_grab_penalty = RewTerm(
        func=mdp.simultaneous_gripper_penalty,
        weight=0.5,
        params={}
    )

    cylinder_to_goal = RewTerm(
    func   = mdp.robot_goal_docking_reward,
    weight = 5.0,
    params = {}          # use defaults above; tune later if needed
    )


    # grab_floor = RewTerm(
    #     func=mdp.is_gripper2_closed_around_goal,
    #     weight=0.0,
    #     params={"tol": 0.15}
    # )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=1e-5,
        params={}
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    self_collision = RewTerm(
        func=mdp.self_collision_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_fixed_in_goal = RewTerm(func=mdp.is_gripper2_closed_around_goal, weight=20.0, params={"tol": 0.1})


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
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    grab_cube = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grab_cube", "weight": 1.5, "num_steps": 1500}
    )

    self_grasp = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "self_grasp", "weight": 2.0, "num_steps": 1000}
    )


    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "reaching_object", "weight": 1.5, "num_steps": 1000}
    )

    robot_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "robot_goal_tracking", "weight": 1.5, "num_steps": 1000}
    )

    robot_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "robot_goal_tracking_fine", "weight": 25.0, "num_steps": 1000}
    )

    # grab_floor = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "grab_floor", "weight": 20.0, "num_steps": 5000}
    # )

    sim_grab_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "sim_grab_penalty", "weight": 1.0, "num_steps": 2000}
    )



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



  