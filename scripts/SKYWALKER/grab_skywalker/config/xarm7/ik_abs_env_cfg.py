# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import sys

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

sys.path.append('/home/bruno/IsaacLab/scripts/SKYWALKER')
import reach_skywalker.mdp as mdp
import reach_skywalker
from reach_skywalker.skywalker_reach_env import SkywalkerEnvCfg
from reach_skywalker.xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG
from isaaclab.assets import ArticulationCfg

from . import joint_pos_env_cfg

from reach_skywalker.mdpactions import GripperImpulseAc

# Initialize Gripper Action
gripper_action = GripperImpulseAction()

# Example: RL Model Outputs an Action Value
action_value = policy_output_from_rl_model  # Positive -> Open, Negative/Zero -> Close

# Apply the Gripper Action
gripper_action.apply(action_value)



##
# Pre-defined configs
##
#from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class SkywalkerReachEnvCfg(joint_pos_env_cfg.SkywalkerReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.47))
                            )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            body_name="link_eef",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            #body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )


@configclass
class SkywalkerReachEnvCfg_PLAY(SkywalkerReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
