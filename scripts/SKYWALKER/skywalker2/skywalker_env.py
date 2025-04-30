from dataclasses import MISSING

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import GroundPlaneCfg, DomeLightCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
#from xarm7 import XARM7_CFG, XARM7_HIGH_PD_CFG


@configclass
class SkywalkerSceneCfg(InteractiveSceneCfg):
    """General-purpose scene for XArm7-based tasks."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=GroundPlaneCfg()
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot placeholder (task-specific env should override this)
    robot: ArticulationCfg = MISSING

    # End-effector frame placeholder (should be set in task config)
    #ee_frame: FrameTransformerCfg = MISSING

    # Optional object (fixed or movable) used in task
    #object: RigidObjectCfg = MISSING


@configclass
class SkywalkerEnvCfg(ManagerBasedRLEnvCfg):
    """Base config for Skywalker-based manipulation tasks."""

    # Basic scene layout
    scene: SkywalkerSceneCfg = SkywalkerSceneCfg(num_envs=1024, env_spacing=4.0)



    # These are to be overridden by task configs
    observations = MISSING
    actions = MISSING
    rewards = MISSING
    terminations = MISSING
    events = MISSING
    curriculum = MISSING

    def __post_init__(self):
        # Simulation config
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.filter_collisions=False

        # Viewer camera setup
        self.viewer.eye = (3.5, 3.5, 3.5)

        # Optional additional sim tuning
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
