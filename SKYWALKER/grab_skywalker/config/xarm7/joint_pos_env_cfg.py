# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
import os
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg

sys.path.append('/home/bruno/IsaacLab/scripts/SKYWALKER')
import grab_skywalker.mdp as mdp
from grab_skywalker.skywalker_grab_env import GrabEnvCfg
from grab_skywalker.xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import SphereCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim import spawners as spawn
from isaaclab.sim import utils as sim_utils



BASE_DIR = os.path.dirname(__file__)
USD_PATH  = os.path.join(BASE_DIR, "wall_cube2.02.usd")

@configclass
class SkywalkerGrabEnvCfg(GrabEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to XARM7
        self.scene.robot = XARM7_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 1, 0.42))  # Robot behind the wall
        )

        self.commands.object_pose.body_name = "link_eef"

        # Use a single small object for the scene.object requirement (hidden/out of the way)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SimpleObject",
            spawn=spawn.CuboidCfg(
                size=(0.01, 0.01, 0.01),  # Very small cube (almost invisible)
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), opacity=0.1),  # Semi-transparent
                collision_props=CollisionPropertiesCfg(collision_enabled=False),  # No collision
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=False,  # Fix kinematic error
                ),
                mass_props=MassPropertiesCfg(mass=0.1),  # Small mass instead of 0
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 5.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),  # Far away from scene
        )

        # Load the assembly as a separate entity using AssetBaseCfg (position it in front of robot)
        self.scene.assembly_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Assembly",
            spawn=UsdFileCfg(
                usd_path=USD_PATH,
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=False,  # Make them proper static rigid bodies, not kinematic
                    disable_gravity=True,
                ),
                mass_props=MassPropertiesCfg(mass=0.0),  # Static mass
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))  # Assembly in front of robot
        )

        # ── goal marker, rewards, terminations ───────────────────────
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
       
        
        self.scene.goal_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GoalMarker",
            spawn=SphereCfg(
                radius=0.03,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=MassPropertiesCfg(mass=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, -0.2, 0]),
        )

        self.scene.dock_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DockMarker",
            spawn=SphereCfg(
                radius=0.03,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.0, 1.0)),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=MassPropertiesCfg(mass=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, -0.4, 0]),
        )

        # ── actions ───────────────────────────────────────────────────
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # EE Surface Gripper action - controls the end-effector surface gripper for object manipulation
        self.actions.gripper_action = mdp.SurfaceGripperActionCfg(
            asset_name="robot",
            gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/EE_SG",
        )
        
        # Cylinder Surface Gripper action - controls the cylinder surface gripper for ground friction
        self.actions.gripper_action2 = mdp.SurfaceGripperActionCfg(
            asset_name="robot", 
            gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/Cylinder_SG",
        )

        # ── frame transformer ────────────────────────────────────────
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.06], rot=[0, 0, 0, 1]),
                ),
            ],
        )

         # Frame transformer for the assembly floor (for reference)
        self.scene.floor = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Floor_xform/Floor",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Floor_xform/Floor"
                ),
            ],
        )

        self.scene.cylinder_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/cylinder/Cylinder",
                    offset=OffsetCfg(pos=[0.0, 0, 0], rot=[0, 0.707,0 ,0.707]),  # ✅ Set offset here
                ),
            ],
        )

        # Frame transformers for the anchored cubes in the assembly
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


        # Add missing frame transformers for surface grippers  
        self.scene.cylinder_SG_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/Cylinder_SG",
                    offset=OffsetCfg(pos=[0.0, 0, 0], rot=[0, 0,0 ,1]), 
                ),
            ],
        )

        self.scene.EE_SG_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/xarm7/link1",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/xarm7/EE_SG",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[0, 0, 0, 1]),
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
        self.scene.env_spacing = 6
        # disable randomization for play
        self.observations.policy.enable_corruption = False
