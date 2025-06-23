# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import torch.nn.functional as F
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the end-effector position in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee = env.scene[ee_cfg.name]

    ee_pos_w = ee.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )

    # ⬇️ Print EE position for the first env
    #print(f"[Debug] EE pos (env 0): {ee_pos_b[0].cpu().numpy()}")

    return ee_pos_b

# def cube2_position_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     cube_cfg:  SceneEntityCfg = SceneEntityCfg("cube2"),
# ):
#     return object_position_in_robot_root_frame(env, robot_cfg, cube_cfg)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object's position in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    try:
        obj_pos_w = obj.data.root_pos_w  # works for RigidObject
    except AttributeError:
        obj_pos_w, _ = obj.get_world_poses()  # fallback for XFormPrim, etc.

    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj_pos_w
    )

    # Optional debug
    #print(f"[Debug] Object pos in base (env 0): {obj_pos_b[0].cpu().numpy()}")

    return obj_pos_b



def cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the object's position in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    root_pos  = robot.data.root_state_w[:, 0:3]   # (N,3)
    root_quat = robot.data.root_state_w[:, 3:7]   # (N,4)
    cube = env.scene[cube_cfg.name]
    obj_pos_w = cube.data.target_pos_w[..., 0, :]   # works for RigidObject
    obj_base, _ = subtract_frame_transforms(root_pos, root_quat, obj_pos_w)

    # Optional debug
    #print(f"[Debug] Object pos in base (env 0): {obj_pos_b[0].cpu().numpy()}")

    return obj_base



# --- 1. Dense forward progress ΔX (positive part) -------------------------- #
def progress_delta(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis: int = 1,          # 0=X, 1=Y, 2=Z
    forward: float = -1.0,  # +1 → positive direction, –1 → negative direction
) -> torch.Tensor:
    """(N,1)  – positive displacement along the chosen axis *in the
    sign specified by `forward`*.
    """
    coord = env.scene[robot_cfg.name].data.root_pos_w[:, axis]      # (N,)

    # static buffer
    if not hasattr(progress_delta, "_prev"):
        progress_delta._prev = coord.clone()

    # signed delta, then keep only the forward-direction part
    delta_signed = (coord - progress_delta._prev) * forward        # flip if −1
    delta        = torch.clamp(delta_signed, min=0.0)              # (N,)
    # reset on env-reset
    if hasattr(env, "reset_buf"):
        progress_delta._prev[env.reset_buf.bool()] = coord[env.reset_buf.bool()]
    else:
        progress_delta._prev = coord.clone()

    

    return delta.unsqueeze(1)
                                          # (N,1)

# --- 2. Vector EE → active cube in robot-root frame ------------------------ #
_STAGE2CUBE = {0: "cube1", 2: "cube2"}        # extend as needed
def ee_to_cube_vec(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg:    SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """(N,3)  – vector from the EE to the *current* mounting cube."""
    stage  = getattr(env, "stage", torch.zeros(env.num_envs,
                                              device=env.device,
                                              dtype=torch.long))
    # Build cube world-positions batch-wise
    cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    for stg, cube in _STAGE2CUBE.items():
        mask = stage == stg
        if torch.any(mask):
            cube_pos_w[mask] = env.scene[cube].data.target_pos_w[..., 0, :][mask]

    # EE and robot base poses
    robot = env.scene[robot_cfg.name]
    ee_pos_w = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]
    root_pos, root_quat = robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7]

    # transform to robot-root frame
    ee_b,   _ = subtract_frame_transforms(root_pos, root_quat, ee_pos_w)
    cube_b, _ = subtract_frame_transforms(root_pos, root_quat, cube_pos_w)
    return cube_b - ee_b                                                  # (N,3)



def gripper_closed(env, grip_term="gripper_action"):
    b = env.action_manager.get_term(grip_term)   # this is your base gripper
    return b.is_closed().float().unsqueeze(1)

def cylinder_closed(env, grip_term="gripper_action2"):
    g = env.action_manager.get_term(grip_term)   # this is your base gripper
    return g.is_closed().float().unsqueeze(1)

def root_lin_vel_xy(env, robot_cfg=SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    return robot.data.root_lin_vel_w[:, :2]                 # (N,2)


