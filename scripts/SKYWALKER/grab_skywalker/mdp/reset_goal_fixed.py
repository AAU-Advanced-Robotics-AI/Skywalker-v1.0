# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------
#  Fixed (or cyclic) goal reset – same conventions as reset_goal_within_reach
# --------------------------------------------------------------------
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

# Per-env index for cycling through presets (lazy-initialised)
_GOAL_CYCLE_IDX = None       # type: torch.Tensor | None

def reset_goal_fixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pos: tuple[float, float, float] | None = None,                # local (x,y,z)
    yaw: float = 0.0,                                             # optional heading
    presets: list[tuple[float, float, float]] | None = None,      # list of locals
):
    """
    • *pos*      → place the goal at that **local** offset every episode.
    • *presets*  → cycle through the list of local offsets per-env.
    All coordinates are interpreted **relative to env.scene.env_origins**,
    so every sandbox gets an identical layout, just like reset_goal_within_reach.
    """
    global _GOAL_CYCLE_IDX
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # -----------------------------------------------------------------
    # Select target local offset for each env
    # -----------------------------------------------------------------
    if pos is not None:
        # accept 3- or 4-tuple, drop extras
        xyz_local = torch.tensor(pos[:3], device=env.device)          # (3,)
        positions = (env.scene.env_origins[env_ids] + xyz_local)      # (N,3)

    elif presets is not None:
        if _GOAL_CYCLE_IDX is None:                                   # lazy init
            _GOAL_CYCLE_IDX = torch.zeros(env.num_envs,
                                          dtype=torch.long,
                                          device=env.device)
        idx = _GOAL_CYCLE_IDX[env_ids] % len(presets)                 # (N,)
        _GOAL_CYCLE_IDX[env_ids] += 1

        xyz_local = torch.tensor(presets, device=env.device)[idx]     # (N,3)
        positions = env.scene.env_origins[env_ids] + xyz_local        # (N,3)
    else:
        # Neither pos nor presets specified → keep whatever was there
        return

    # -----------------------------------------------------------------
    # Orientation (flat marker → no tilt)
    # -----------------------------------------------------------------
    quat = quat_from_euler_xyz(
        torch.zeros(num_envs, device=env.device),   # roll
        torch.zeros(num_envs, device=env.device),   # pitch
        torch.full((num_envs,), yaw, device=env.device),  # yaw
    )

    # -----------------------------------------------------------------
    # Velocities (zero)
    # -----------------------------------------------------------------
    root_pose = torch.cat([positions, quat], dim=-1)          # (N,7)
    root_vel  = torch.zeros((num_envs, 6), device=env.device)

    # -----------------------------------------------------------------
    # Write to simulator
    # -----------------------------------------------------------------
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel,  env_ids=env_ids)
