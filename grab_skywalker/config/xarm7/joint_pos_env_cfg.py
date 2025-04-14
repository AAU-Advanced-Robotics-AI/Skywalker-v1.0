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
import grab_skywalker.mdp as mdp
import grab_skywalker
from grab_skywalker.skywalker_grab_env import GrabEnvCfg
from grab_skywalker.xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
##
# Pre-defined configs
##
#from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class SkywalkerGrabEnvCfg(GrabEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = XARM7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.47))
                            )


        self.commands.object_pose.body_name = "link_eef"
        
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 0.5], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass= 0.5)
            ),
        )

        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_eef"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_eef"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_eef"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.SurfaceGripperActionCfg(
            asset_name="robot",
            gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
        )

                # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

@configclass
class SkywalkerGrabEnvCfg_PLAY(SkywalkerGrabEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
