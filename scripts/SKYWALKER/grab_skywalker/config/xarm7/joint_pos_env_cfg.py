import os
from isaaclab.utils import configclass
from grab_skywalker.skywalker_grab_env import GrabEnvCfg
from grab_skywalker.xarm7 import XARM7_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
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
USD_PATH  = os.path.join(BASE_DIR, "wall_cube2.usd")

@configclass
class SkywalkerGrabEnvCfg(GrabEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── robot (unchanged) ───────────────────────────────────────
        self.scene.robot = XARM7_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.47)),
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
                pos=(0.8, 0, 0.5),
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
             init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0]),
        )

        # ── actions ───────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.SurfaceGripperActionCfg(
            asset_name="robot",
            gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
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

        self.scene.cube = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube_xform/Cube",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Assembly/Assembly_xform/Cube_xform/Cube"
                ),
            ],
        )

@configclass
class SkywalkerGrabEnvCfg_PLAY(SkywalkerGrabEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
