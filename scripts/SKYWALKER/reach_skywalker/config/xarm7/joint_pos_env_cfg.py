# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys


from isaaclab.utils import configclass

#import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
#from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

sys.path.append('/home/bruno/IsaacLab/scripts/SKYWALKER')
import reach_skywalker.mdp as mdp
import reach_skywalker
from reach_skywalker.skywalker_reach_env import SkywalkerEnvCfg
from reach_skywalker.xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG
from isaaclab.assets import ArticulationCfg

##
# Pre-defined configs
##
#from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class SkywalkerReachEnvCfg(SkywalkerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.47))
                            )
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_eef"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_eef"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_eef"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "link_eef"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

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
