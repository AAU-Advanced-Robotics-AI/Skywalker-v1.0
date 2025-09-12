# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Export the functions
__all__ = ["object_position_in_robot_root_frame", "progress_delta", "ee_to_cube_vec"]


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def progress_delta(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Progress delta - a normalized time indicator for the environment."""
    # Simple progress based on episode time
    progress = env.episode_length_buf / env.max_episode_length
    return progress.unsqueeze(-1)  # Shape: (N, 1) instead of (N,)


def ee_to_cube_vec(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Vector from end-effector to cube object."""
    robot = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get end-effector position (assuming you have ee_frame configured)
    if hasattr(env.scene, "ee_frame"):
        ee_pos_w = env.scene.ee_frame.data.target_pos_w[..., 0, :]
    else:
        # Fallback to robot root position if ee_frame not available
        ee_pos_w = robot.data.root_pos_w[:, :3]
    
    # Get object position
    object_pos_w = object.data.root_pos_w[:, :3]
    
    # Return vector from EE to object
    return object_pos_w - ee_pos_w
