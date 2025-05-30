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


def is_cube1_grasped(env) -> torch.Tensor:
    return is_cube_grasped(env,
        cube_cfg=SceneEntityCfg("cube1"),
        ee_cfg=SceneEntityCfg("ee_frame"),
        grip_term="gripper_action")

def is_cube2_grasped(env) -> torch.Tensor:
    return is_cube_grasped(env,
        cube_cfg=SceneEntityCfg("cube2"),
        ee_cfg=SceneEntityCfg("ee_frame"),
        grip_term="gripper_action")
def cube2_relative_to_cube1(env) -> torch.Tensor:
    c1_pos = cube_position_in_robot_root_frame(env, SceneEntityCfg("cube1"))  # (N,3)
    c2_pos = cube_position_in_robot_root_frame(env, SceneEntityCfg("cube2"))  # (N,3)
    return c2_pos - c1_pos  # (N,3)


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


def gripper_closed(env, grip_term="gripper_action"):
    b = env.action_manager.get_term(grip_term)   # this is your base gripper
    return b.is_closed().float().unsqueeze(1)

def cylinder_closed(env, grip_term="gripper_action2"):
    g = env.action_manager.get_term(grip_term)   # this is your base gripper
    return g.is_closed().float().unsqueeze(1)

def goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    radius: float = 0.2,
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
                    cube_cfg=SceneEntityCfg("cube1"),
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



def time_fraction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Returns how far each env is through its episode, ∈ [0,1].
    Internally keeps a per‐env counter (reset on env.reset), and
    divides by (episode_length_s / sim.dt) drawn from cfg.
    """
    # lazy‐init static timer
    if not hasattr(time_fraction, "_timer"):
        # shape (num_envs,)
        time_fraction._timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    timer = time_fraction._timer

    # increment every step
    timer += 1.0

    # reset where env.reset just happened
    reset_buf = getattr(env, "reset_buf", None)       # shape (num_envs,) with 1 on reset
    if reset_buf is not None:
        # zero out those entries
        mask = (reset_buf > 0).float().to(env.device)
        timer = timer * (1.0 - mask)

    # write back
    time_fraction._timer = timer

    # normalize
    total_steps = env.cfg.episode_length_s / env.cfg.sim.dt
    frac = (timer / total_steps).clamp(0.0, 1.0)

    # return (num_envs,1)
    return frac.unsqueeze(1)
def ee_to_cube1_distance(env) -> torch.Tensor:
    """(N,1) – Euclidean distance from EE to cube1 in world frame."""
    ee_pos = ee_position_in_robot_root_frame(env)                    # (N,3)
    c1_pos = cube_position_in_robot_root_frame(
        env, cube_cfg=SceneEntityCfg("cube1"))                       # (N,3)
    return torch.norm(ee_pos - c1_pos, dim=1, keepdim=True)         # (N,1)

def ee_to_cube2_distance(env) -> torch.Tensor:
    """(N,1) – Euclidean distance from EE to cube1 in world frame."""
    ee_pos = ee_position_in_robot_root_frame(env)                    # (N,3)
    c2_pos = cube_position_in_robot_root_frame(
        env, cube_cfg=SceneEntityCfg("cube2"))                       # (N,3)
    return torch.norm(ee_pos - c2_pos, dim=1, keepdim=True)         # (N,1)



def cube_grasp_counter_obs(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg,
    grip_term: str = "gripper_action",
    tol: float = 0.15,
) -> torch.Tensor:
    """
    Per-env counter of how many times the gripper closed near a specific cube.
    • Resets to 0 on env reset.
    """

    cube_name = cube_cfg.name
    counter_key = f"_counter_{cube_name}"
    prev_key    = f"_prev_closed_{cube_name}"

    # Gripper state
    grip = env.action_manager.get_term(grip_term)
    is_closed = grip.is_closed().float()  # (N,)

    # Get cube and EE position in world
    cube = env.scene[cube_name]
    ee   = env.scene["ee_frame"]
    dist = torch.norm(ee.data.target_pos_w[..., 0, :] - cube.data.target_pos_w[..., 0, :], dim=1)
    near = (dist < tol).float()  # (N,)

    # Detect closing edge
    if not hasattr(cube_grasp_counter_obs, prev_key):
        setattr(cube_grasp_counter_obs, prev_key, is_closed.clone())
    prev_closed = getattr(cube_grasp_counter_obs, prev_key)
    just_closed = (is_closed > 0.5) & (prev_closed < 0.5)
    setattr(cube_grasp_counter_obs, prev_key, is_closed.clone())

    # Lazy-init counter
    if not hasattr(cube_grasp_counter_obs, counter_key):
        setattr(cube_grasp_counter_obs, counter_key, torch.zeros(env.num_envs, device=env.device))
    counter = getattr(cube_grasp_counter_obs, counter_key)

    # Increment where grasp event happened
    counter += (just_closed * near)

    # Reset on env reset
    if hasattr(env, "reset_buf"):
        reset_mask = (env.reset_buf > 0).float()
        counter *= (1.0 - reset_mask)

    # Save back
    setattr(cube_grasp_counter_obs, counter_key, counter)

    return counter.unsqueeze(1)


def cube1_grasp_count(env): 
    return cube_grasp_counter_obs(env, cube_cfg=SceneEntityCfg("cube1"))

def cube2_grasp_count(env): 
    return cube_grasp_counter_obs(env, cube_cfg=SceneEntityCfg("cube2"))



