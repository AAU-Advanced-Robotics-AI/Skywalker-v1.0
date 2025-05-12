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
    #print(f"[Debug] EE pos (env 0): {ee_pos_b[0].cpu().numpy()}")

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
    #print(f"[Debug] Object pos in base (env 0): {obj_pos_b[0].cpu().numpy()}")

    return obj_pos_b


def gripper_contact_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    term1 = env.action_manager.get_term("gripper_action")
    term2 = env.action_manager.get_term("gripper_action2")

    state1 = term1.get_grasping_mask()
    state2 = term2.get_grasping_mask()

    #print(f"[Debug] Gripper states (env 0): {state1[0].item()}, {state2[0].item()}")

    return torch.stack([state1, state2], dim=1)

def cylinder_closed(env, grip_term="gripper_action2"):
    g = env.action_manager.get_term(grip_term)   # this is your base gripper
    return g.is_closed().float().unsqueeze(1)

def goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    radius: float = 0.7,
) -> torch.Tensor:
    """Return the goal marker’s position in the robot’s root frame and
    debug-print it for the first 30 vectorised environments."""
    robot: RigidObject = env.scene[robot_cfg.name]
    goal = env.scene[goal_cfg.name]

    # World-frame position of the goal marker
    if hasattr(goal.data, "root_pos_w"):
        goal_pos_w = goal.data.root_pos_w          # RigidObject
    else:
        goal_pos_w, _ = goal.get_world_poses()     # XFormPrim or similar

    goal_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],            # robot base position
        robot.data.root_state_w[:, 3:7],           # robot base orientation (quat)
        goal_pos_w,
    )

    # Print positions for envs 0-29 (or fewer if batch smaller)
    #slice_end = min(30, goal_pos_b.shape[0])
    #print(f"[Debug] Goal pos in base (env 0-{slice_end-1}): {goal_pos_b[:slice_end].cpu().numpy()}")

    return goal_pos_b/radius


def root_lin_vel_xy(env, robot_cfg=SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    return robot.data.root_lin_vel_w[:, :2]                 # (N,2)

def is_cube_grasped(env,
                    cube_cfg=SceneEntityCfg("cube"),
                    ee_cfg=SceneEntityCfg("ee_frame"),
                    grip_term="gripper_action") -> torch.Tensor:
    gripper = env.action_manager.get_term(grip_term)
    closed  = gripper.is_closed().float()                   # (N,)
    ee      = env.scene[ee_cfg.name]
    cube    = env.scene[cube_cfg.name]
    dist    = torch.norm(ee.data.target_pos_w[...,0,:] -
                         cube.data.target_pos_w[...,0,:], dim=1)
    near    = (dist < 0.15).float()
    return (closed * near).unsqueeze(1)                     # (N,1)





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
