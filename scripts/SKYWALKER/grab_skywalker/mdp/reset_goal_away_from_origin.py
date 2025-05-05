import torch
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

def reset_goal_away_from_origin(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    velocity_range: dict,
    min_radius: float = 0.25,
):
    """Reset goal but avoid spawning within a radius of the origin (X, Y)."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    num_envs = len(env_ids)

    # Sample until all points are outside the forbidden radius
    accepted = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
   

# Initialize with environment origins
    positions = env.scene.env_origins[env_ids].clone()

    while not accepted.all():
        x = sample_uniform(*pose_range["x"], (num_envs,), device=env.device)
        y = sample_uniform(*pose_range["y"], (num_envs,), device=env.device)
        z = torch.full((num_envs,), pose_range["z"][0], device=env.device)

        offsets = torch.stack([x, y, z], dim=-1)
        dist = torch.norm(offsets[:, :2], dim=1)
        keep = dist >= min_radius

        accepted = accepted | keep
        positions[keep] = env.scene.env_origins[env_ids][keep] + offsets[keep]


    # Orientation (Euler XYZ)
    r = quat_from_euler_xyz(
        torch.zeros(num_envs, device=env.device),
        torch.zeros(num_envs, device=env.device),
        torch.zeros(num_envs, device=env.device),
    )

    # Velocities
    velocities = torch.zeros((num_envs, 6), device=env.device)

    # Apply reset
    asset.write_root_pose_to_sim(torch.cat([positions, r], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
