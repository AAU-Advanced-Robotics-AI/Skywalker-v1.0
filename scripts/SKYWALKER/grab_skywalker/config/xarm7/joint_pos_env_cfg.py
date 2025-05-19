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

        # ── Spawn the robot with self-collision ON ───────────────────────────────

        self.scene.robot = XARM7_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",

            # mutate the UsdFileCfg that loads the robot
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
            offset=[0,0,0.1],
            rot_offset = [0, -0.707,0 ,0.707],
            grip_threshold= 0.1
        )

        self.actions.gripper_action2 = mdp.SurfaceGripperActionCfg(
            asset_name="robot",
            gripper_prim_path="{ENV_REGEX_NS}/Robot/xarm7/cylinder/Cylinder",
            offset=[0,0.5,-0.15],
            rot_offset = [0, 0.707,0 ,0.707],
            grip_threshold=  0.1
            
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
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
