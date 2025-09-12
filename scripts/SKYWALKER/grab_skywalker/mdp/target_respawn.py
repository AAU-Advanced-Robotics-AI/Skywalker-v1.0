"""Target respawn functions using built-in Isaac Lab events."""

from typing import Optional
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import RigidObject
import isaaclab.envs.mdp as mdp


def respawn_target_on_distance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float,
    pose_range: dict,
    velocity_range: Optional[dict] = None,
) -> None:
    """Respawn target when end-effector reaches within threshold distance.
    
    This function uses the built-in Isaac Lab reset_root_state_uniform function
    for proper USD handling, only triggering when distance condition is met.
    """
    # Default empty velocity range if not provided
    if velocity_range is None:
        velocity_range = {}
    
    # Get the frame transformer and target asset
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    
    # Get end-effector positions (in world frame)
    ee_positions = ee_frame.data.target_pos_w[env_ids, 0, :]  # [N, 3]
    
    # Get target positions (in world frame)
    target_positions = target_asset.data.root_pos_w[env_ids]  # [N, 3]
    
    # Calculate distances
    distances = torch.norm(ee_positions - target_positions, dim=1)
    
    # Find environments where target is reached
    reached_mask = distances < threshold
    reached_env_ids = env_ids[reached_mask]
    
    if len(reached_env_ids) > 0:
        # Use built-in Isaac Lab function to respawn targets
        mdp.reset_root_state_uniform(
            env=env,
            env_ids=reached_env_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=target_cfg,
        )
