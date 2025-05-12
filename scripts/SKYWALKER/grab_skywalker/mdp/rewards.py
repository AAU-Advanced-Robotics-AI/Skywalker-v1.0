# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets.asset_base import AssetBase
import torch
from typing import TYPE_CHECKING
import grab_skywalker.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms


if TYPE_CHECKING:
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
    reach_tol: float = 0.20,
    penalty: float = -1.0,
):
    # Gripper status
    term = env.action_manager.get_term(grip_term)
    closed = term.is_closed().float()  # shape (N,)

    # EE is always a FrameTransformer
    ee_frame = env.scene[ee_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # shape (N, 3)

    # Object could be any prim
    obj_prim = env.scene[obj_cfg.name]
    try:
        obj_pos = obj_prim.data.root_pos_w
    except AttributeError:
        obj_pos, _ = obj_prim.get_world_poses()

    # Distance check
    near = (torch.norm(ee_pos - obj_pos, dim=1) < reach_tol).float()
    bad = closed * (1.0 - near)

    return penalty * bad



from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms



def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Exponential‐decay reward on the distance between EE and the cube link under Assembly/Object,
    never considering the wall.
    """
    # 1) Robot base pose in world frame
    robot = env.scene[robot_cfg.name]
    root_pos  = robot.data.root_state_w[:, 0:3]   # (N,3)
    root_quat = robot.data.root_state_w[:, 3:7]   # (N,4)

    # 2) End‐effector world positions
    ee = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee.data.target_pos_w[..., 0, :]    # (N,3)

    # 3) Cube world positions via the RigidObject API on the 'object' asset
    obj = env.scene[cube_cfg.name]
    obj_pos_w = obj.data.target_pos_w[..., 0, :]    # (N,3)

    # 4) Transform both into robot‐base frame
    obj_base, _ = subtract_frame_transforms(root_pos, root_quat, obj_pos_w)
    ee_base,  _ = subtract_frame_transforms(root_pos, root_quat, ee_pos_w)

    # 5) Euclidean distance & exponential reward
    dist = torch.norm(obj_base - ee_base, dim=1)   # (N,)
    return torch.exp(-dist / std)



def ee_approach_alignment_in_base(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage the EE to approach the cube from the front (+X) direction of the robot's base frame."""
    # Positions and orientations
    robot = env.scene[robot_cfg.name]
    root_pos  = robot.data.root_state_w[:, 0:3]
    root_quat = robot.data.root_state_w[:, 3:7]

    ee = env.scene[ee_frame_cfg.name]
    cube = env.scene[cube_cfg.name]

    ee_pos_w = ee.data.target_pos_w[..., 0, :]    # (N,3)
    cube_pos_w = cube.data.target_pos_w[..., 0, :]  # (N,3)

    # Transform both positions into the robot's base frame
    ee_in_base, _ = subtract_frame_transforms(root_pos, root_quat, ee_pos_w)
    cube_in_base, _ = subtract_frame_transforms(root_pos, root_quat, cube_pos_w)

    # Vector from EE to cube in base frame
    vec = cube_in_base - ee_in_base  # (N, 3)
    vec = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-6)

    # Compare with robot’s +X direction: [1, 0, 0]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=vec.device)

    alignment = torch.sum(vec * x_axis, dim=1)  # cosine of angle between them

    # Normalize [-1, 1] → [0, 1]
    return (alignment + 1.0) / 2.0


def robot_base_to_goal_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Compute reward based on distance from robot base to per-env goal marker."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # shape (num_envs, 2)
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]   # shape (num_envs, 2)

    distance = torch.norm(root_pos - goal_pos, dim=1)

    # Optional: print first 30 distances
    #print("Distances (first 30 envs):", distance[:30].cpu().numpy())

    return 1 - torch.tanh(distance)


def robot_base_to_goal_distance_fine(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Compute reward based on distance from robot base to per-env goal marker."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # shape (num_envs, 2)
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]   # shape (num_envs, 2)

    distance = torch.norm(root_pos - goal_pos, dim=1)

    # Optional: print first 30 distances
    #print("Distances (first 30 envs):", distance[:30].cpu().numpy())

    return 1 - torch.tanh(distance / 0.1)


def object_ee_orientation_alignment(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for aligning EE Z-axis opposite to cube's Z-axis using cosine similarity."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Convert quaternions to rotation matrices
    cube_rot_w = quat_to_rot_matrix(cube.data.root_quat_w)                      # (num_envs, 3, 3)
    ee_rot_w = quat_to_rot_matrix(ee_frame.data.target_quat_w[..., 0, :])         # (num_envs, 3, 3)

    # Extract Z-axes from rotation matrices (3rd column)
    cube_z_axis = cube_rot_w[..., :, 2]  # shape (num_envs, 3)
    ee_z_axis = ee_rot_w[..., :, 2]      # shape (num_envs, 3)

    # Cosine similarity between EE Z and negative cube Z (we want them pointing opposite)
    alignment = torch.sum(ee_z_axis * -cube_z_axis, dim=1)

    # Normalize [-1, 1] → [0, 1] for reward
    reward = (alignment + 1) / 2

    return reward

def ee_cube_orientation_alignment(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward EE Z-axis being horizontal (i.e., perpendicular to world Z)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]  # (N, 4)

    ee_rot = quat_to_rot_matrix(ee_quat)  # (N, 3, 3)
    ee_z = ee_rot[..., :, 2]              # EE Z-axis in world frame

    world_z = torch.tensor([0.0, 0.0, 1.0], device=ee_z.device)
    vertical_alignment = torch.abs(torch.sum(ee_z * world_z, dim=1))  # ∈ [0, 1]
    return 1.0 - vertical_alignment  # max reward when EE Z is horizontal





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

# def is_grasping_fixed_object(env):
#     try:
#         gripper_term = env.action_manager.get_term("gripper_action")
#     except KeyError:                    # term missing → no crash
#         return torch.zeros(env.num_envs, device=env.device)

#     return gripper_term.is_closed().float()



def is_grasping_fixed_object(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    tol: float = 0.2,
    grip_term: str = "gripper_action",
) -> torch.Tensor:
    """Reward if the gripper is closed and EE is near the cube (likely grasp)."""
    try:
        gripper = env.action_manager.get_term(grip_term)
        is_closed = gripper.is_closed().float()
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)

    ee = env.scene[ee_cfg.name]
    cube = env.scene[cube_cfg.name]

    ee_pos = ee.data.target_pos_w[..., 0, :]
    cube_pos = cube.data.target_pos_w[..., 0, :]

    dist = torch.norm(ee_pos - cube_pos, dim=1)
    is_near = (dist < tol).float()

    reward = is_closed * is_near
    #print(f"[DEBUG] is_closed: {is_closed[0].item()}, is_near: {is_near[0].item()}, reward: {reward[0].item()}")


    return is_closed * is_near  # Only reward if closed near the cube

def simultaneous_gripper_penalty(
    env: ManagerBasedRLEnv,
    grip_term_1: str = "gripper_action",
    grip_term_2: str = "gripper_action2",
    grace_steps: int = 30,    # ~2 seconds
    penalty: float = -1.0,
) -> torch.Tensor:
    try:
        term1 = env.action_manager.get_term(grip_term_1)
        term2 = env.action_manager.get_term(grip_term_2)
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)

    g1 = term1.is_closed().float()
    g2 = term2.is_closed().float()

    # Access the shared timer from either term (they should be the same size)
    timers = term1._shared_gripper_state_timer

    grace_mask = (timers >= grace_steps).float()
    same_state = (g1 == g2).float()

    penalty_applied = penalty * same_state * grace_mask

    # Optional debug
    print(f"[DEBUG] g1: {g1[0].item()}, g2: {g2[0].item()}, grace: {grace_mask[0].item()}, penalty: {penalty_applied[0].item()}")

    return penalty_applied

def cylinder_goal_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg:  SceneEntityCfg = SceneEntityCfg("goal_marker"),
    grip_term: str = "gripper_action2",
    std: float = 0.25,           # distance at which reward ≈ e-1 ≈ 0.37
    tol: float = 0.12,           # “close-enough” zone (same units as distance)
    close_bonus: float = 1.0,    # extra bump for closing inside tol
    far_penalty: float = -0.3,   # small negative for closing too early
) -> torch.Tensor:
    """
    Dense docking reward in robot-base frame.

    • Always gives exp(-d / std) so the critic sees a gradient.
    • Adds +close_bonus when gripper2 *closes* within tol of the goal.
    • Adds –far_penalty when gripper2 closes farther than tol.
    """
    robot = env.scene[robot_cfg.name]
    goal  = env.scene[goal_cfg.name]
    gripper = env.action_manager.get_term(grip_term)

    # distance in XY plane (robot frame == cylinder frame)
    root_pos = robot.data.root_pos_w[:, :2]
    goal_pos = goal.data.root_pos_w[:, :2]
    dist = torch.norm(root_pos - goal_pos, dim=1)        # (N,)

    base_reward = torch.exp(-dist / std)                 # (0, 1]

    closed = gripper.is_closed().float()                 # (N,)
    near   = (dist < tol).float()

    # combine signals
    shaped = base_reward + close_bonus * closed * near + far_penalty * closed * (1 - near)
    return shaped

# def is_gripper2_closed_around_goal(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
#     grip_term: str = "gripper_action2",
#     tol: float = 0.05,
# ) -> torch.Tensor:
#     """
#     Reward if gripper2 is closed and robot base is near the goal_marker.
#     """
#     try:
#         gripper = env.action_manager.get_term(grip_term)
#         is_closed = gripper.is_closed().float()
#     except KeyError:
#         return torch.zeros(env.num_envs, device=env.device)

#     robot = env.scene[robot_cfg.name]
#     goal = env.scene[goal_cfg.name]

#     robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
#     goal_pos = goal.data.root_pos_w[:, :2]    # (N, 2)

#     dist = torch.norm(robot_pos - goal_pos, dim=1)  # (N,)
#     is_near = (dist < tol).float()

#     reward = is_closed * is_near

#     print(f"[DEBUG] gripper2 is_closed: {is_closed[0].item()}, robot near goal: {is_near[0].item()}, reward: {reward[0].item()}")

#     return reward








# def is_gripper2_closed_reward(
#     env: ManagerBasedRLEnv,
#     ee_cfg: SceneEntityCfg = SceneEntityCfg("cylinder_frame"),
#     grip_term: str = "gripper_action2",
# ) -> torch.Tensor:
#     """
#     Reward if gripper2 is closed (regardless of proximity). Meant to test grasping behavior.
#     """
#     try:
#         gripper = env.action_manager.get_term(grip_term)
#         is_closed = gripper.is_closed().float()
#     except KeyError:
#         return torch.zeros(env.num_envs, device=env.device)

#     # Optional: print debug info
#     print(f"[DEBUG] gripper2 is_closed (env 0): {is_closed[0].item()}")

#     return is_closed
