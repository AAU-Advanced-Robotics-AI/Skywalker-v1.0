# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.sensors import FrameTransformer
from .actions import SurfaceGripperActionTerm


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

from isaaclab.envs import ManagerBasedRLEnv



def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (..., 4) to rotation matrix (..., 3, 3)"""
    # Normalize quaternion just in case
    q = q / q.norm(dim=-1, keepdim=True)

    w, x, y, z = q.unbind(-1)

    # Build rotation matrix
    B = q.shape[:-1]  # batch shape
    rot = torch.empty(*B, 3, 3, device=q.device)

    rot[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)

    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    rot[..., 1, 2] = 2 * (y * z - x * w)

    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot

def self_grasp_penalty(
    env,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    obj_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grip_term: str = "gripper_action",
    reach_tol: float = 0.20,   # metres – tweak
    penalty: float = -1.0,
):
    term   = env.action_manager.get_term(grip_term)
    closed = term.is_closed().float()                       # (N,)

    ee_pos   = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]  # (N,3)
    cube_pos = env.scene[obj_cfg.name].data.root_pos_w             # (N,3)
    near     = (torch.norm(ee_pos - cube_pos, dim=1) < reach_tol).float()

    bad = closed * (1.0 - near)    # 1 if suction is closed *and* cube not near → self-grab
    return penalty * bad


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    z_offset = torch.tensor([0.0, 0.0, 0.0], device=object.data.root_pos_w.device)

    cube_pos_w = object.data.root_pos_w + z_offset
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def robot_base_to_goal_distance(
    env: ManagerBasedRLEnv,
    goal_pos: list[float] = [0.8, 0.0],  # X, Y target goal
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for moving its base closer to the goal position."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # X, Y
    goal = torch.tensor(goal_pos, device=root_pos.device)
    distance = torch.norm(root_pos - goal, dim=1)
    return 1 - torch.tanh(distance / 0.3)  # Smooth reward


def object_ee_orientation_alignment(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for aligning EE Z-axis opposite to cube's Z-axis using cosine similarity."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Convert quaternions to rotation matrices
    cube_rot_w = quat_to_rot_matrix(object.data.root_quat_w)                      # (num_envs, 3, 3)
    ee_rot_w = quat_to_rot_matrix(ee_frame.data.target_quat_w[..., 0, :])         # (num_envs, 3, 3)

    # Extract Z-axes from rotation matrices (3rd column)
    cube_z_axis = cube_rot_w[..., :, 2]  # shape (num_envs, 3)
    ee_z_axis = ee_rot_w[..., :, 2]      # shape (num_envs, 3)

    # Cosine similarity between EE Z and negative cube Z (we want them pointing opposite)
    alignment = torch.sum(ee_z_axis * -cube_z_axis, dim=1)

    # Normalize [-1, 1] → [0, 1] for reward
    reward = (alignment + 1) / 2

    return reward

def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    # Isaac Lab ≥ v2.x ⇒ field is now `joint_vel`
    return -torch.sum(robot.data.joint_vel ** 2, dim=1)




_prev = None        # module-level cache

def action_rate_l2(env):
    global _prev
    am = env.action_manager

    # use public tensors in v2
    curr = am.action            # shape (N, dim)
    if _prev is None:
        _prev = am.prev_action  # first call after reset
    diff = curr - _prev
    _prev = curr.clone()        # update cache

    return -torch.sum(diff ** 2, dim=1)

def reset_action_rate_cache():
    global _prev
    _prev = None



def self_collision_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    if hasattr(robot, "get_self_collisions"):
        return (robot.get_self_collisions() > 0).float() * -1.0
    else:
        return torch.zeros(env.num_envs, device=env.device)

def is_grasping_fixed_object(env):
    try:
        gripper_term = env.action_manager.get_term("gripper_action")
    except KeyError:                    # term missing → no crash
        return torch.zeros(env.num_envs, device=env.device)

    return gripper_term.is_closed().float()






