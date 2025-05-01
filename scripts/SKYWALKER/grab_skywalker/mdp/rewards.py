# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets.asset_base import AssetBase
import torch
from typing import TYPE_CHECKING

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

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Get scene elements
    obj_prim = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Get robot root pose (world frame)
    root_pos = robot.data.root_state_w[:, 0:3]     # (N, 3)
    root_quat = robot.data.root_state_w[:, 3:7]    # (N, 4)

    # Get object position (world frame)
    try:
        obj_pos_w = obj_prim.data.root_pos_w       # (N, 3) or (1, 3)
        if obj_pos_w.shape[0] == 1:
            obj_pos_w = obj_pos_w.expand(env.num_envs, -1)
    except AttributeError:
        obj_pos_w, _ = obj_prim.get_world_poses()
        if obj_pos_w.shape[0] == 1:
            obj_pos_w = obj_pos_w.expand(env.num_envs, -1)

    # Get EE position (world frame)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)

    # Transform both to base frame
    obj_pos_base, _ = subtract_frame_transforms(root_pos, root_quat, obj_pos_w)
    ee_pos_base, _ = subtract_frame_transforms(root_pos, root_quat, ee_pos_w)

    # Print per-environment object/EE base positions
    # print("\n[DEBUG] EE and Object positions (base frame) for all envs:")
    # for i in range(env.num_envs):
    #     print(f"Env {i:3d} | EE: {ee_pos_base[i].cpu().numpy()} | Object: {obj_pos_base[i].cpu().numpy()}")

    # Compute distance in base frame
    dist = torch.norm(obj_pos_base - ee_pos_base, dim=1)

    # Summary stats
    #print(f"\n[DIST DEBUG] Min: {dist.min():.3f} Max: {dist.max():.3f} Mean: {dist.mean():.3f}")

    # Reward: exponential decay
    return torch.exp(-dist / std)










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
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tol: float = 0.1,
    grip_term: str = "gripper_action",
) -> torch.Tensor:
    """Reward if the gripper is closed and EE is near the cube (likely grasp)."""
    try:
        gripper = env.action_manager.get_term(grip_term)
        is_closed = gripper.is_closed().float()
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)

    ee = env.scene[ee_cfg.name]
    obj = env.scene[object_cfg.name]

    ee_pos = ee.data.target_pos_w[..., 0, :]
    try:
        obj_pos = obj.data.root_pos_w
    except AttributeError:
        obj_pos, _ = obj.get_world_poses()

    dist = torch.norm(ee_pos - obj_pos, dim=1)
    is_near = (dist < tol).float()

    reward = is_closed * is_near
    print(f"[DEBUG] is_closed: {is_closed[0].item()}, is_near: {is_near[0].item()}, reward: {reward[0].item()}")


    return is_closed * is_near  # Only reward if closed near the cube


##OLD REWARDS

# def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
#     """Convert quaternion (..., 4) to rotation matrix (..., 3, 3)"""
#     # Normalize quaternion just in case
#     q = q / q.norm(dim=-1, keepdim=True)

#     w, x, y, z = q.unbind(-1)

#     # Build rotation matrix
#     B = q.shape[:-1]  # batch shape
#     rot = torch.empty(*B, 3, 3, device=q.device)

#     rot[..., 0, 0] = 1 - 2 * (y**2 + z**2)
#     rot[..., 0, 1] = 2 * (x * y - z * w)
#     rot[..., 0, 2] = 2 * (x * z + y * w)

#     rot[..., 1, 0] = 2 * (x * y + z * w)
#     rot[..., 1, 1] = 1 - 2 * (x**2 + z**2)
#     rot[..., 1, 2] = 2 * (y * z - x * w)

#     rot[..., 2, 0] = 2 * (x * z - y * w)
#     rot[..., 2, 1] = 2 * (y * z + x * w)
#     rot[..., 2, 2] = 1 - 2 * (x**2 + y**2)

#     return rot


# def object_is_lifted(
#     env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     z_offset = torch.tensor([0.0, 0.0, 0.4], device=object.data.root_pos_w.device)

#     cube_pos_w = object.data.root_pos_w + z_offset
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)



# def object_ee_orientation_alignment(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for aligning EE Z-axis opposite to cube's Z-axis using cosine similarity."""
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     # Convert quaternions to rotation matrices
#     cube_rot_w = quat_to_rot_matrix(object.data.root_quat_w)                      # (num_envs, 3, 3)
#     ee_rot_w = quat_to_rot_matrix(ee_frame.data.target_quat_w[..., 0, :])         # (num_envs, 3, 3)

#     # Extract Z-axes from rotation matrices (3rd column)
#     cube_z_axis = cube_rot_w[..., :, 2]  # shape (num_envs, 3)
#     ee_z_axis = ee_rot_w[..., :, 2]      # shape (num_envs, 3)

#     # Cosine similarity between EE Z and negative cube Z (we want them pointing opposite)
#     alignment = torch.sum(ee_z_axis * -cube_z_axis, dim=1)

#     # Normalize [-1, 1] → [0, 1] for reward
#     reward = (alignment + 1) / 2

#     return reward




# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# # def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
# #     """Penalize tracking orientation error using shortest path.

# #     The function computes the orientation error between the desired orientation (from the command) and the
# #     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
# #     path between the desired and current orientations.
# #     """
# #     # extract the asset (to enable type hinting)
# #     asset: RigidObject = env.scene[asset_cfg.name]
# #     command = env.command_manager.get_command(command_name)
# #     # obtain the desired and current orientations
# #     des_quat_b = command[:, 3:7]
# #     des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
# #     curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
# #     return quat_error_magnitude(curr_quat_w, des_quat_w)

# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)
