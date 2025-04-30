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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

position_history = []
orientation_history = []

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_bonus(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < 0.05).to(torch.int).to(asset.data.device)


def position_command_error_bonus_terminate(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""

    # Get the position error for the batch of environments
    distance = position_command_error_bonus(env, command_name, asset_cfg)

    # Get the orientation error for the batch of environments
    orientation_error = orientation_command_error(env, command_name, asset_cfg)

    # Set the thresholds (for position and orientation)
    position_threshold = 0.02  # Distance threshold (2 cm)
    orientation_threshold = 0.02  # Orientation threshold (2 degrees in radians)

    # Check if each environment meets the termination criteria (position and orientation)
    position_terminated = distance < position_threshold  # Position terminated if distance is below the threshold
    orientation_terminated = orientation_error < orientation_threshold  # Orientation terminated if error is below the threshold

    # Ensure the history is kept for each environment separately
    device = distance.device  # Make sure device consistency
    
    # Stack history and append the new batch of terminated states (position and orientation)
    if len(position_history) < 5:
        position_history.append(position_terminated)
        orientation_history.append(orientation_terminated)
    else:
        # Remove the oldest entry (shift history) and add the new one
        position_history.pop(0)
        orientation_history.pop(0)
        position_history.append(position_terminated)
        orientation_history.append(orientation_terminated)

    # Efficient check for termination (for each environment over the last 10 steps)
    position_history_tensor = torch.stack(position_history, dim=0)  # Shape: [10, batch_size]
    orientation_history_tensor = torch.stack(orientation_history, dim=0)  # Shape: [10, batch_size]

    # Check if the last 10 steps were all terminated (True) for position and orientation
    position_check = torch.all(position_history_tensor, dim=0)  # Check over the last 10 steps for position
    orientation_check = torch.all(orientation_history_tensor, dim=0)  # Check over the last 10 steps for orientation

    # Termination condition met if both position and orientation are satisfied
    termination_condition_met = position_check & orientation_check

    # Return a tensor with True/False for each environment based on whether the termination condition is met
    return termination_condition_met.to(device)
# Ensure the tensor is on the same device


# def position_command_error_bonus(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
#     return (distance < 0.02).to(torch.int).to(asset.data.device)

# def position_command_error_bonus_terminate(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """


#     distance = position_command_error_bonus(env, command_name, asset_cfg)

#     # Check the orientation error for each environment
#     orientation_error = orientation_command_error(env, command_name, asset_cfg)

#     # Set the thresholds (for position and orientation)
#     position_threshold = 0.02  # Distance threshold (2 cm)
#     orientation_threshold = 0.035  # Orientation threshold (2 degrees in radians)

#     # Check if each environment meets the termination criteria
#     position_terminated = distance < position_threshold  # Position terminated if distance is below the threshold
#     orientation_terminated = orientation_error < orientation_threshold  # Orientation terminated if error is below the threshold

#     # Add to the history
#     position_history.append(position_terminated)
#     orientation_history.append(orientation_terminated)

#     # Ensure the history only contains the last 10 values for each environment
#     if len(position_history) > 10:
#         position_history.pop(0)  # Remove the oldest value if the list exceeds size 10

#     if len(orientation_history) > 10:
#         orientation_history.pop(0)  # Remove the oldest value if the list exceeds size 10

#     # For each environment, check if it has been terminated for the last 10 steps
#     termination_condition_met = []
#     for i in range(len(position_terminated)):  # Iterate over each environment (batch of environments)
#         # Check if both position and orientation have been within the thresholds for the last 10 steps
#         if len(position_history) == 10 and len(orientation_history) == 10:
#             position_check = all(position_history[j][i] for j in range(10))
#             orientation_check = all(orientation_history[j][i] for j in range(10))
#             if position_check and orientation_check:
#                 termination_condition_met.append(True)
#             else:
#                 termination_condition_met.append(False)
#         else:
#             termination_condition_met.append(False)

#     # Return a tensor with True/False for each environment based on whether the termination condition is met
#     return torch.tensor(termination_condition_met)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
