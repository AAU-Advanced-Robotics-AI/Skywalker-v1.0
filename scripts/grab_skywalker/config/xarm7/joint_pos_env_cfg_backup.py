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
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
)
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)


import grab_skywalker.mdp as mdp
from math import sqrt
BASE_DIR = os.path.dirname(__file__)
USD_PATH  = os.path.join(BASE_DIR, "wall_cube2.02.usd")

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
                usd_path=XARM7_CFG.spawn.usd_path,
                # copy across any other spawn fields you need (e.g. scale, semantic tags)
                articulation_props=ArticulationRootPropertiesCfg(
                    # ensure articulation is enabled...
                    articulation_enabled=True,
                    # ...and *this* turns on link↔link self‐contacts
                    enabled_self_collisions=True,
                    # you can tune solver iters if you like
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    fix_root_link=False,
                ),
                # leave any other physics_material / rigid_body_props as-is,
                # or raise contact_threshold there if you’re getting jitter.
                **{ k:v for k,v in vars(XARM7_CFG.spawn).items()
                    if k not in ("usd_path","articulation_props") }
            ),

            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0.85, 0.42),
                joint_pos={},
                joint_vel={},
            ),
        )

# inside your __post_init__ of SkywalkerGrabEnvCfg, replace your wall_object/articulation block with:

        self.scene.object = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Assembly",
            spawn=UsdFileCfg(
                usd_path=USD_PATH,
                # uniformly scale down to 50%
                #scale=(1, 1, 1),
                articulation_props=ArticulationRootPropertiesCfg(
                    fix_root_link=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.8, 0, 0.65),
                # 180° rotation about Z (w, x, y, z)
                rot=(0,0, 0.0, 1.0),
                joint_pos={},  
                joint_vel={},
            ),
            actuators={},  # if you don’t need to actuate the assembly directly
        )




        # ── goal marker, rewards, terminations ───────────────────────


        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.goal_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GoalMarker",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=MassPropertiesCfg(mass=0.0),
            ),
             init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, -0.2, 0]),
        )

        self.scene.dock_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DockMarker",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.0, 1.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=MassPropertiesCfg(mass=0.0),
            ),
        )

        self.scene.cube2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, -0.15, 0.5], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3)
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
        # Dummy gripper action for XARM7 - controls joint7 with minimal movement
        # Since XARM7 doesn't have separate finger joints, we use the wrist joint as dummy
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["joint7"], 
            scale=0.01,  # Very small scale to minimize effect
            use_default_offset=True
        )
        # Disable surface gripper actions for main branch compatibility
        # self.actions.gripper_action = mdp.SurfaceGripperActionCfg(
        #     asset_name="robot",
        #     gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
        # )

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
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.06], rot=[0, 0, 0, 1]),
                ),
            ],
        )

        self.scene.EE_SG_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/EE_SG",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[0, 0, 0, 1]),
                ),
            ],
        )

        self.scene.cylinder_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/cylinder/Cylinder",
                    offset=OffsetCfg(pos=[0.0, 0, 0], rot=[0, 0.707,0 ,0.707]),  # ✅ Set offset here
                ),
            ],
        )

        self.scene.cylinder_SG_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/Cylinder_SG",
                    offset=OffsetCfg(pos=[0.0, 0, 0], rot=[0, 0,0 ,1]), 
                ),
            ],
        )

        self.scene.cube1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube1_xform/Cube1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube1_xform/Cube1"
                ),
            ],
        )

        self.scene.cube2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube2_xform/Cube2",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube2_xform/Cube2"
                ),
            ],
        )

        self.scene.cube3 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube3_xform/Cube3",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube3_xform/Cube3"
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
