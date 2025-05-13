import torch
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv




import torch
import math
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

def reset_goal_within_reach(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    EEL: float,
    LA: float,
    HW: float,
    HR: float,
    RR: float,
    z: float = 0.337,
    min_radius: float = 0.25,  # avoid spawning at base
):
    """
    Reset goal position using robot reach and kinematic constraints centered around the cube,
    while also avoiding a radius around the robot base.
    """
    asset = env.scene[asset_cfg.name]
    cube = env.scene["cube1"]
    cube_pos = cube.data.target_pos_w[env_ids, 0, :]  # shape [len(env_ids), 3]
    base_pos = env.scene["robot"].data.root_pos_w[env_ids, :3]  # shape [len(env_ids), 3]

    num_envs = len(env_ids)
    vertical = abs(HW - HR)
    max_radius = EEL + math.sqrt(max(LA**2 - vertical**2, 0.01))

    accepted = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    positions = env.scene.env_origins[env_ids].clone()

    while not accepted.all():
        x = torch.empty(num_envs, device=env.device).uniform_(RR, max_radius)
        maxy = torch.sqrt(torch.clamp(max_radius**2 - x**2, min=0.0))
        y = torch.empty(num_envs, device=env.device).uniform_(-1.0, 1.0) * maxy

        sample_x = cube_pos[:, 0] - x
        sample_y = cube_pos[:, 1] + y

        dist_to_base = torch.norm(torch.stack([
            sample_x - base_pos[:, 0],
            sample_y - base_pos[:, 1]
        ], dim=1), dim=1)

        keep = (dist_to_base >= min_radius) & (~accepted)

        positions[keep, 0] = sample_x[keep]
        positions[keep, 1] = sample_y[keep]
        positions[keep, 2] = z
        accepted = accepted | keep

    r = quat_from_euler_xyz(
        torch.zeros(num_envs, device=env.device),
        torch.zeros(num_envs, device=env.device),
        torch.zeros(num_envs, device=env.device),
    )
    velocities = torch.zeros((num_envs, 6), device=env.device)

    asset.write_root_pose_to_sim(torch.cat([positions, r], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)




# def reset_goal_away_from_origin(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     asset_cfg: SceneEntityCfg,
#     pose_range: dict,
#     velocity_range: dict,
#     min_radius: float = 0.25,
# ):
#     """Reset goal but avoid spawning within a radius of the origin (X, Y)."""
#     asset = env.scene[asset_cfg.name]
#     root_states = asset.data.default_root_state[env_ids].clone()

#     num_envs = len(env_ids)

#     # Sample until all points are outside the forbidden radius
#     accepted = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
   

# # Initialize with environment origins
#     positions = env.scene.env_origins[env_ids].clone()

#     while not accepted.all():
#         x = sample_uniform(*pose_range["x"], (num_envs,), device=env.device)
#         y = sample_uniform(*pose_range["y"], (num_envs,), device=env.device)
#         z = torch.full((num_envs,), pose_range["z"][0], device=env.device)

#         offsets = torch.stack([x, y, z], dim=-1)
#         dist = torch.norm(offsets[:, :2], dim=1)
#         keep = dist >= min_radius

#         accepted = accepted | keep
#         positions[keep] = env.scene.env_origins[env_ids][keep] + offsets[keep]


#     # Orientation (Euler XYZ)
#     r = quat_from_euler_xyz(
#         torch.zeros(num_envs, device=env.device),
#         torch.zeros(num_envs, device=env.device),
#         torch.zeros(num_envs, device=env.device),
#     )

#     # Velocities
#     velocities = torch.zeros((num_envs, 6), device=env.device)

#     # Apply reset
#     asset.write_root_pose_to_sim(torch.cat([positions, r], dim=-1), env_ids=env_ids)
#     asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
