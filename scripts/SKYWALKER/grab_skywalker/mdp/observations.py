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
    print(f"[Debug] EE pos (env 0): {ee_pos_b[0].cpu().numpy()}")

    return ee_pos_b


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
    print(f"[Debug] Object pos in base (env 0): {obj_pos_b[0].cpu().numpy()}")

    return obj_pos_b




# def object_position_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """The position of the object in the robot's root frame."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     object_pos_w = object.data.root_pos_w[:, :3]
#     object_pos_b, _ = subtract_frame_transforms(
#         robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
#     )
#     return object_pos_b
