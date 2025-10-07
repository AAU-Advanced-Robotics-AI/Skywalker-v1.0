# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-to-Point specific MDP functions."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --- DEBUG: step-by-step action/obs logger (no effect on reward) ---
import numpy as np
import isaaclab.envs.mdp as isaac_mdp
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

def debug_log_step(
    env: ManagerBasedRLEnv,
    step_mod: int = 120,        # print every N env steps
    env_index: int = 0,         # which env to print
) -> torch.Tensor:
    # throttle prints
    c = getattr(debug_log_step, "_c", 0) + 1
    debug_log_step._c = c
    if c % step_mod != 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene["robot"]

    # current & default joints
    jpos = robot.data.joint_pos[env_index, :7].detach().cpu().numpy()
    jvel = robot.data.joint_vel[env_index, :7].detach().cpu().numpy()
    jdef = robot.data.default_joint_pos[env_index, :7].detach().cpu().numpy()

    # EE pose (what your obs uses)
    ee_pos  = env.scene["ee_frame"].data.target_pos_w[env_index, 0, :3].detach().cpu().numpy()
    ee_quat = env.scene["ee_frame"].data.target_quat_w[env_index, 0, :].detach().cpu().numpy()  # wxyz

    # Command that goes into observations during training
    target_pose_cmd = isaac_mdp.generated_commands(env, command_name="target_pose")[env_index].detach().cpu().numpy()
    # [x,y,z, qw,qx,qy,qz]

    # "last_action" as used by the obs during training (should be in [-1, 1])
    last_action = isaac_mdp.last_action(env)[env_index, :7].detach().cpu().numpy()

    # action term config (what training actually used)
    act_term = env.action_manager.get_term("arm_action")
    scale = float(getattr(act_term.cfg, "scale", 0.5))
    use_offset = bool(getattr(act_term.cfg, "use_default_offset", True))

    # what the *absolute* target would be if the term uses offset + scale*action
    expected_abs = jdef + scale * last_action

    # try to peek a target buffer from the action term, if it exists
    target_buf = None
    for name in ("_target_positions", "target_positions", "_targets", "targets"):
        if hasattr(act_term, name):
            t = getattr(act_term, name)
            try:
                target_buf = t[env_index, :7].detach().cpu().numpy()
            except Exception:
                pass
            break

    sat = np.sum(np.abs(last_action) > 0.99)

    print(f"\n[TRAIN DEBUG] step={c} env={env_index}")
    print(f" last_action (pre-scale, ~tanh): {np.round(last_action, 3)}  (sat={sat}/7)")
    print(f" scale={scale}  use_default_offset={use_offset}")
    print(f" default_joint_pos: {np.round(jdef, 3)}")
    print(f" expected_abs_target = default + scale*action: {np.round(expected_abs, 3)}")
    print(f" current_joint_pos:  {np.round(jpos, 3)}")
    if target_buf is not None:
        print(f" action_term target buffer: {np.round(target_buf, 3)}")
    print(f" ee_pos: {np.round(ee_pos, 3)}  ee_quat(wxyz): {np.round(ee_quat, 3)}")
    print(f" target_pose_command (pos[0:3],wxyz[3:]): {np.round(target_pose_cmd, 3)}")

    # return zeros (weight=0 in cfg → no reward impact)
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def debug_print_limits_on_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Print limits & defaults once at reset for env 0."""
    if 0 in env_ids.tolist():
        robot: Articulation = env.scene["robot"]
        low  = robot.data.soft_joint_pos_limits[0, :7, 0].detach().cpu().numpy()
        high = robot.data.soft_joint_pos_limits[0, :7, 1].detach().cpu().numpy()
        jdef = robot.data.default_joint_pos[0, :7].detach().cpu().numpy()
        print("\n[TRAIN DEBUG] (reset) joint soft limits (rad):")
        print(" low :", np.round(low,  6))
        print(" high:", np.round(high, 6))
        print(" default:", np.round(jdef, 6))


def ee_position(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)

def ee_position_base_frame(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position relative to robot base frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    # Get EE position in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)
    
    # Get robot base position in world frame  
    base_pos_w = robot.data.root_pos_w[:, :3]  # (N, 3)
    
    # Return EE position relative to base
    ee_pos_base = ee_pos_w - base_pos_w  # (N, 3)
    
    # DEBUG: Print EE position (first environment only, every 100 steps)
    if hasattr(ee_position_base_frame, '_step_count'):
        ee_position_base_frame._step_count += 1
    else:
        ee_position_base_frame._step_count = 0
    
    if ee_position_base_frame._step_count % 100 == 0:
        print(f"[PTP DEBUG] EE pos WORLD: {ee_pos_w[0].detach().cpu().numpy()}")
        print(f"[PTP DEBUG] Base pos WORLD: {base_pos_w[0].detach().cpu().numpy()}")  
        print(f"[PTP DEBUG] EE pos BASE: {ee_pos_base[0].detach().cpu().numpy()}")
    
    return ee_pos_base

def ee_orientation(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]  # (N, 4)

def ee_orientation_base_frame(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation relative to robot base frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    # Get EE orientation in world frame
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (N, 4) [w,x,y,z]
    
    # Get robot base orientation in world frame
    base_quat_w = robot.data.root_quat_w  # (N, 4) [w,x,y,z]
    
    # Compute relative quaternion: q_rel = q_ee * q_base^(-1)
    # For inverse: q^(-1) = [w, -x, -y, -z] / ||q||^2, but since ||q||=1, just negate xyz
    base_quat_inv = torch.cat([base_quat_w[:, 0:1], -base_quat_w[:, 1:4]], dim=1)  # (N, 4)
    
    # Quaternion multiplication: q_ee * q_base_inv
    w1, x1, y1, z1 = ee_quat_w[:, 0], ee_quat_w[:, 1], ee_quat_w[:, 2], ee_quat_w[:, 3]
    w2, x2, y2, z2 = base_quat_inv[:, 0], base_quat_inv[:, 1], base_quat_inv[:, 2], base_quat_inv[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    ee_quat_base = torch.stack([w, x, y, z], dim=1)  # (N, 4) [w,x,y,z]
    
    # DEBUG: Print EE orientation (first environment only, every 100 steps)
    if hasattr(ee_orientation_base_frame, '_step_count'):
        ee_orientation_base_frame._step_count += 1
    else:
        ee_orientation_base_frame._step_count = 0
    
    if ee_orientation_base_frame._step_count % 100 == 0:
        print(f"[PTP DEBUG] EE quat WORLD: {ee_quat_w[0].detach().cpu().numpy()}")
        print(f"[PTP DEBUG] Base quat WORLD: {base_quat_w[0].detach().cpu().numpy()}")
        print(f"[PTP DEBUG] EE quat BASE: {ee_quat_base[0].detach().cpu().numpy()}")
    
    return ee_quat_base


def target_position(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker")) -> torch.Tensor:
    """Target marker position in world frame."""
    target: RigidObject = env.scene[target_cfg.name]
    return target.data.root_pos_w[:, :3]  # (N, 3)


def ee_to_target_distance(
    env: ManagerBasedRLEnv, 
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    reward_threshold: float | None = None,
) -> torch.Tensor:
    """Distance from end-effector to target.
    
    Args:
        env: The environment.
        ee_frame_cfg: End-effector frame configuration.
        target_cfg: Target configuration.
        reward_threshold: If provided, distances within this threshold return 0 
                         (indicating maximum reward zone).
    
    Returns:
        Distance tensor. If reward_threshold is set, distances within threshold are clamped to 0.
    """
    ee_pos = ee_position(env, ee_frame_cfg)
    target_pos = target_position(env, target_cfg)
    
    # Convert from world coordinates to local environment coordinates
    env_origins = env.scene.env_origins
    ee_pos_local = ee_pos - env_origins
    target_pos_local = target_pos - env_origins
    
    distance = torch.norm(ee_pos_local - target_pos_local, dim=1, keepdim=True)
    
    # Apply reward threshold if specified - distances within threshold get 0 (max reward)
    if reward_threshold is not None:
        distance = torch.where(distance <= reward_threshold, 
                              torch.zeros_like(distance), 
                              distance)
    
    return distance  # (N, 1)

def ptp_orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    command_name: str = "target_pose",
    tolerance_deg: float = 1.0,     # full reward inside this angle
    max_deg: float = 180.0,          # zero reward by here
) -> torch.Tensor:
    """Reward EE orientation matching the commanded quaternion (from UniformPoseCommand).

    Returns:
        (N,) shaped reward in [0, 1].
    """
    # EE orientation (wxyz)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    q_ee = ee_frame.data.target_quat_w[..., 0, :]              # (N, 4)

    # Commanded pose: [x,y,z, qw,qx,qy,qz]
    cmd = isaac_mdp.generated_commands(env, command_name=command_name)  # (N, 7)
    q_cmd = cmd[:, 3:7]                                                 # (N, 4) wxyz

    # Normalize quats (robustness)
    def _normalize(q):
        return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-9)

    q_ee  = _normalize(q_ee)
    q_cmd = _normalize(q_cmd)

    # Smallest angle between orientations:
    # angle = 2*acos(|dot(q1, q2)|)
    dot = torch.sum(q_ee * q_cmd, dim=-1).abs().clamp(0.0, 1.0)
    ang_rad = 2.0 * torch.acos(dot)
    ang_deg = ang_rad * (180.0 / math.pi)

    # Piecewise-linear shaping: 1 inside tolerance, then down to 0 by max_deg
    inside = (ang_deg <= tolerance_deg)
    slope  = torch.clamp((ang_deg - tolerance_deg) / max(1e-6, (max_deg - tolerance_deg)), 0.0, 1.0)
    reward = torch.where(inside, torch.ones_like(ang_deg), 1.0 - slope)
    return reward

def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the orientation using a tanh kernel.

    Returns:
        (N,) tensor in [0, 1]:
            - 1.0 when perfectly aligned (zero angle error)
            - ~0.0 as the angular error grows large

    Notes:
        - Give this term a **positive** weight in your reward table.
        - `std` is a length scale (in radians). Smaller -> sharper drop-off.
          For example, use `std = math.radians(10)` for ~10° characteristic scale.
    """
    # extract the asset and command
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired orientation is the command quaternion given in the asset's root frame
    # command format: [x, y, z, qw, qx, qy, qz]
    des_quat_b = command[:, 3:7]  # (N, 4) in asset root frame
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)  # (N, 4) → world frame

    # current body orientation (world)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (N, 4)

    # angular error magnitude (radians) along the shortest rotation
    ang = quat_error_magnitude(curr_quat_w, des_quat_w)  # (N,)

    # tanh-shaped reward: 1 at zero error, decays toward 0 as error grows
    # clamp std for numerical safety
    std_safe = max(float(std), 1e-8)
    return 1.0 - torch.tanh(ang / std_safe)



def ptp_ee_to_target_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Binary reward for reaching target within threshold."""
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)
    return (distance < threshold).float()


def ptp_distance_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    max_distance: float = 2.0,
    reward_threshold: float = 0.05,
) -> torch.Tensor:
    """Dense distance-based reward that maximizes reward for staying within threshold.
    
    Gives maximum reward (1.0) when within reward_threshold distance,
    then decreases linearly as distance increases beyond threshold.
    This encourages staying within the target zone rather than just touching it.
    """
    # Get raw distance (without threshold clamping)
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg, reward_threshold=None).squeeze(-1)
    
    # Debug prints every 100 steps to avoid spam
    if hasattr(env, '_debug_counter'):
        env._debug_counter += 1
    else:
        env._debug_counter = 0
    
    if env._debug_counter % 100 == 0:
        ee_pos = ee_position(env, ee_frame_cfg)
        target_pos = target_position(env, target_cfg)
        
        # Get actual sphere positions from the physics simulation
        target_asset = env.scene[target_cfg.name]
        actual_sphere_pos = target_asset.data.root_pos_w
        
        # Convert to local environment coordinates for display
        env_origins = env.scene.env_origins
        ee_pos_local = ee_pos - env_origins
        target_pos_local = target_pos - env_origins
        actual_sphere_pos_local = actual_sphere_pos - env_origins
        
        print(f"\n=== DEBUG STEP {env._debug_counter} ===")
        print(f"Number of environments: {distance.shape[0]}")
        for env_idx in range(min(30, distance.shape[0])):  # Print first 5 envs
            print(f"Env {env_idx}:")
            print(f"  EE position (world): [{ee_pos[env_idx, 0]:.3f}, {ee_pos[env_idx, 1]:.3f}, {ee_pos[env_idx, 2]:.3f}]")
            print(f"  EE position (local): [{ee_pos_local[env_idx, 0]:.3f}, {ee_pos_local[env_idx, 1]:.3f}, {ee_pos_local[env_idx, 2]:.3f}]")
            print(f"  Target pos (world):  [{target_pos[env_idx, 0]:.3f}, {target_pos[env_idx, 1]:.3f}, {target_pos[env_idx, 2]:.3f}]")
            print(f"  Target pos (local):  [{target_pos_local[env_idx, 0]:.3f}, {target_pos_local[env_idx, 1]:.3f}, {target_pos_local[env_idx, 2]:.3f}]")
            print(f"  Sphere pos (world):  [{actual_sphere_pos[env_idx, 0]:.3f}, {actual_sphere_pos[env_idx, 1]:.3f}, {actual_sphere_pos[env_idx, 2]:.3f}]")
            print(f"  Sphere pos (local):  [{actual_sphere_pos_local[env_idx, 0]:.3f}, {actual_sphere_pos_local[env_idx, 1]:.3f}, {actual_sphere_pos_local[env_idx, 2]:.3f}]")
            print(f"  Env origin:          [{env_origins[env_idx, 0]:.3f}, {env_origins[env_idx, 1]:.3f}, {env_origins[env_idx, 2]:.3f}]")
            print(f"  Distance:            {distance[env_idx]:.3f}m")
            
            # Show reward calculation
            within_threshold = distance[env_idx] <= reward_threshold
            if within_threshold:
                reward_val = 1.0
            else:
                normalized_distance = torch.clamp((distance[env_idx] - reward_threshold) / (max_distance - reward_threshold), 0.0, 1.0)
                reward_val = 1.0 - normalized_distance
            print(f"  Within threshold ({reward_threshold}m): {within_threshold}")
            print(f"  Distance reward: {reward_val:.3f}")
        print("=" * 40)
    
    # BALANCED ANTI-OSCILLATION REWARD STRUCTURE
    # Give good reward for being within threshold, reasonable reward for being close
    within_threshold = distance <= reward_threshold
    
    # For distances outside threshold, give scaled rewards that make reaching threshold attractive
    excess_distance = distance - reward_threshold  # How much beyond threshold
    max_excess = max_distance - reward_threshold   # Maximum excess distance to consider
    normalized_excess = torch.clamp(excess_distance / max_excess, 0.0, 1.0)
    outside_threshold_reward = 1.0 - normalized_excess  # Scale from 1.0 to 0.0
    
    # Combine: STRONG reward within threshold, good scaling outside
    final_reward = torch.where(
        within_threshold,
        torch.full_like(distance, 5.0),     # 5x reward when within threshold (achievable)
        outside_threshold_reward            # Scaled reward when outside (0.0 to 1.0)
    )
    
    return final_reward


def ptp_target_reached(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Termination condition: target reached."""
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)
    return distance < threshold


def respawn_target_when_reached(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    threshold: float = 0.05,
    pose_range: dict | None = None,
    velocity_range: dict | None = None,
) -> None:
    """Environment-specific event: respawn target when reached (only affects specific environments)."""
    # Calculate distances for ALL environments
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)
    
    # Find which environments have reached their targets (from ALL environments, not just env_ids)
    target_reached = distance < threshold
    reached_env_ids = target_reached.nonzero(as_tuple=True)[0]
    
    # Only proceed if some environments actually reached their targets
    if len(reached_env_ids) > 0:
        print(f"=== ENVIRONMENT-SPECIFIC TARGET RESPAWN ===")
        print(f"Total environments: {env.num_envs}")
        print(f"Environments that reached targets: {reached_env_ids.tolist()}")
        print(f"Distances for reached envs: {distance[reached_env_ids].tolist()}")
        
        # Get the target asset
        target_asset = env.scene[target_cfg.name]
        num_envs = len(reached_env_ids)
        
        # Store old positions for debug
        old_positions = target_asset.data.root_pos_w[reached_env_ids].clone()
        
        # Use default pose range if not provided
        if pose_range is None:
            pose_range = {
                "x": (0.25, 0.55),
                "y": (-0.25, 0.25),  
                "z": (0.75, 0.82),
            }
        
        # Sample random positions for reached environments - use same logic as reset function
        local_positions = torch.zeros((num_envs, 3), device=env.device)
        
        # Sample within the pose range
        x_min, x_max = pose_range["x"]
        y_min, y_max = pose_range["y"] 
        z_min, z_max = pose_range["z"]
        
        local_positions[:, 0] = torch.rand(num_envs, device=env.device) * (x_max - x_min) + x_min
        local_positions[:, 1] = torch.rand(num_envs, device=env.device) * (y_max - y_min) + y_min
        local_positions[:, 2] = torch.rand(num_envs, device=env.device) * (z_max - z_min) + z_min
        
        # Convert to world coordinates by adding environment origins
        env_origins = env.scene.env_origins[reached_env_ids]
        world_positions = local_positions + env_origins
        
        # Keep default rotation (identity quaternion)
        from isaaclab.utils.math import quat_from_euler_xyz
        rotation = quat_from_euler_xyz(
            torch.zeros(num_envs, device=env.device),
            torch.zeros(num_envs, device=env.device),
            torch.zeros(num_envs, device=env.device),
        )
        
        # Zero velocities
        velocities = torch.zeros((num_envs, 6), device=env.device)
        
        # Apply the new pose using proven method
        pose_data = torch.cat([world_positions, rotation], dim=-1)
        target_asset.write_root_pose_to_sim(pose_data, env_ids=reached_env_ids)
        target_asset.write_root_velocity_to_sim(velocities, env_ids=reached_env_ids)
        
        # Force immediate visual update through multiple approaches
        target_asset.data.root_pos_w[reached_env_ids] = world_positions
        target_asset.data.root_quat_w[reached_env_ids] = rotation
        target_asset.data.root_vel_w[reached_env_ids] = velocities
        
        # Force visual update via USD on the exact prims we moved
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            context = omni.usd.get_context()
            stage = context.get_stage()
            if stage:
                # Try to get correct prim paths
                prim_paths = None
                
                # Method 1: Check if target asset has prim_paths attribute
                if hasattr(target_asset, 'prim_paths') and target_asset.prim_paths:
                    prim_paths = target_asset.prim_paths
                
                # Method 2: Try to construct from cfg.prim_path (this was causing the issue)
                elif hasattr(target_asset, 'cfg') and hasattr(target_asset.cfg, 'prim_path'):
                    base_path = target_asset.cfg.prim_path
                    # Handle the template path properly - replace the template with actual env paths
                    if "{ENV_REGEX_NS}" in base_path:
                        # Extract the suffix after the template
                        suffix = base_path.replace("{ENV_REGEX_NS}", "")
                        prim_paths = [f"/World/envs/env_{env_id}{suffix}" for env_id in range(env.num_envs)]
                    else:
                        # If no template, assume it's already a specific path
                        prim_paths = [base_path for env_id in range(env.num_envs)]
                
                # Method 3: Final fallback with direct paths
                if prim_paths is None:
                    prim_paths = [f"/World/envs/env_{env_id}/target_marker" for env_id in range(env.num_envs)]
                
                # Apply visual updates
                for i, env_id in enumerate(reached_env_ids.tolist()):
                    if env_id < len(prim_paths):
                        path = prim_paths[env_id]
                        prim = stage.GetPrimAtPath(path)
                        if prim and prim.IsValid():
                            xform = UsdGeom.Xformable(prim)
                            if xform:
                                pos = world_positions[i].detach().cpu().numpy()
                                # Reuse or add a Translate op
                                translate_op = next((op for op in xform.GetOrderedXformOps()
                                                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                                if translate_op:
                                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                                else:
                                    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                
                stage.GetRootLayer().dirty = True
        except Exception as e:
            # Silently handle USD visual update issues - they don't affect physics
            pass
        
        # Read back the actual sphere positions after update to verify visual movement
        actual_sphere_positions = target_asset.data.root_pos_w[reached_env_ids].clone()
        
        # Sanity checks to prove physics moved the marker
        print(f"[RESPAWN] wrote: {world_positions[:1].tolist()}")
        print(f"[RESPAWN] sim now: {target_asset.data.root_pos_w[reached_env_ids][:1].tolist()}")
        
        print(f"[RESPAWN] Targets respawned for {len(reached_env_ids)} environments ONLY")
        print(f"  Old positions: {old_positions[:3].tolist()}")  # Show first 3
        print(f"  Local positions: {local_positions[:3].tolist()}")  # Show first 3 local
        print(f"  World positions: {world_positions[:3].tolist()}")      # Show first 3 world
        print(f"  Actual sphere positions: {actual_sphere_positions[:3].tolist()}")  # Show actual sphere data
        print(f"  Other environments were NOT affected")
        print("=" * 50)


def check_and_respawn_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.05,
    min_radius: float = 0.25,
    max_radius: float = 0.55,
    min_height_offset: float = 0.1,
    max_height_offset: float = 0.4,
    min_azimuth: float = -120.0,
    max_azimuth: float = 120.0,
    min_elevation: float = -45.0,
    max_elevation: float = 45.0,
) -> None:
    """Check if target is reached and respawn it if so."""
    # Check which environments have reached their targets
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)
    target_reached = distance < threshold
    
    # Get the environment IDs that reached their targets
    reached_env_ids = target_reached.nonzero(as_tuple=True)[0]
    
    if len(reached_env_ids) > 0:
        print(f"=== TARGET RESPAWN ===")
        print(f"Environments that reached targets: {reached_env_ids.tolist()}")
        print(f"Distances: {distance[reached_env_ids].tolist()}")
        
        # Respawn targets for environments that reached them
        randomize_target_spherical(
            env,
            reached_env_ids,
            target_cfg,
            robot_cfg,
            min_radius=min_radius,
            max_radius=max_radius,
            min_height_offset=min_height_offset,
            max_height_offset=max_height_offset,
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
        )
        print(f"Targets respawned for {len(reached_env_ids)} environments")
        print("========================")


def randomize_target_spherical(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    min_radius: float = 0.3,   # 300mm minimum
    max_radius: float = 0.6,   # 600mm maximum (robot's reach)
    min_height_offset: float = 0.2,   # 0.2m above robot joint
    max_height_offset: float = 0.6,   # 0.6m above robot joint
    min_azimuth: float = -120.0, # -120 degrees (avoid back of robot)
    max_azimuth: float = 120.0,  # +120 degrees (avoid back of robot)
    min_elevation: float = -30.0, # -30 degrees (below joint level)
    max_elevation: float = 60.0,  # +60 degrees (above joint level)
) -> None:
    """Randomize target position in a spherical area relative to robot's first joint.
    
    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        asset_cfg: Asset configuration for the target.
        robot_cfg: Robot asset configuration to get joint position.
        min_radius: Minimum distance from robot first joint (meters).
        max_radius: Maximum distance from robot first joint (meters, max 600mm for safety).
        min_height_offset: Minimum height offset from robot joint (meters).
        max_height_offset: Maximum height offset from robot joint (meters).
        min_azimuth: Minimum azimuth angle from robot front in degrees (-180 to +180).
        max_azimuth: Maximum azimuth angle from robot front in degrees (-180 to +180).
        min_elevation: Minimum elevation angle in degrees (-90 to +90).
        max_elevation: Maximum elevation angle in degrees (-90 to +90).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get robot's first joint position as reference point
    # The robot body names follow pattern: link1, link2, etc.
    link1_body_idx = robot.find_bodies("link1")[0]
    joint1_positions = robot.data.body_pos_w[:, link1_body_idx, :]
    
    # Sample random positions
    num_resets = len(env_ids)
    
    # Generate random radius (uniform distribution in volume, not radius)
    radius_cubed = torch.rand(num_resets, device=env.device) * (max_radius**3 - min_radius**3) + min_radius**3
    radius = torch.pow(radius_cubed, 1.0/3.0)
    
    # Generate random azimuth angle (horizontal rotation)
    azimuth_rad = torch.rand(num_resets, device=env.device) * math.radians(max_azimuth - min_azimuth) + math.radians(min_azimuth)
    
    # Generate random elevation angle (vertical rotation)
    elevation_rad = torch.rand(num_resets, device=env.device) * math.radians(max_elevation - min_elevation) + math.radians(min_elevation)
    
    # Generate random height offset from robot joint
    height_offset = torch.rand(num_resets, device=env.device) * (max_height_offset - min_height_offset) + min_height_offset
    
    # Get joint1 position for the environments being reset
    joint1_pos_reset = joint1_positions[env_ids]  # Shape: [num_resets, 3] or [num_resets, 1, 3]
    
    # Flatten the tensor to ensure shape [num_resets, 3]
    if joint1_pos_reset.dim() == 3:
        joint1_pos_reset = joint1_pos_reset.squeeze(1)  # Remove middle dimension
    
    # Convert spherical to cartesian coordinates relative to robot's first joint
    # Spherical coordinates: (radius, azimuth, elevation)
    x = joint1_pos_reset[:, 0] + radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = joint1_pos_reset[:, 1] + radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = joint1_pos_reset[:, 2] + height_offset + radius * torch.sin(elevation_rad)
    
    # Create new positions
    pos = torch.stack([x, y, z], dim=1)
    
    # Debug: Show old and new positions
    print(f"Target sphere respawning:")
    for i, env_id in enumerate(env_ids):
        old_pos = asset.data.root_pos_w[env_id].tolist()
        new_pos = pos[i].tolist()
        print(f"  Env {env_id}: {old_pos} -> {new_pos}")
    
    # Create full state with position and zero velocity (for kinematic bodies)
    zero_vel = torch.zeros((len(env_ids), 6), device=env.device)
    new_state = torch.cat([
        pos,  # position
        asset.data.default_root_state[env_ids, 3:7],  # keep original rotation
        zero_vel  # zero velocity
    ], dim=1)
    
    # Set new positions using multiple methods to ensure update
    asset.write_root_state_to_sim(new_state, env_ids)
    
    # Force update internal data
    asset.data._root_pos_w.timestamp = -1.0
    asset.data._root_link_pose_w.timestamp = -1.0
    asset.data._root_state_w.timestamp = -1.0


def randomize_rigid_body_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    velocity_range: dict,
) -> None:
    """Randomize rigid body position within specified range relative to each environment's origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Sample random positions
    num_resets = len(env_ids)
    
    # Generate random positions relative to environment origins
    local_pos = torch.zeros((num_resets, 3), device=env.device)
    
    if "x" in pose_range:
        x_min, x_max = pose_range["x"]
        local_pos[:, 0] = torch.rand(num_resets, device=env.device) * (x_max - x_min) + x_min
    
    if "y" in pose_range:
        y_min, y_max = pose_range["y"] 
        local_pos[:, 1] = torch.rand(num_resets, device=env.device) * (y_max - y_min) + y_min
        
    if "z" in pose_range:
        z_min, z_max = pose_range["z"]
        local_pos[:, 2] = torch.rand(num_resets, device=env.device) * (z_max - z_min) + z_min
    
    # Add environment origins to convert from local to world coordinates
    env_origins = env.scene.env_origins[env_ids]
    world_pos = local_pos + env_origins
    
    print(f"[RESET] Randomizing targets for envs: {env_ids.tolist()}")
    print(f"  Local positions: {local_pos[:3].tolist()}")  # Show first 3
    print(f"  Env origins: {env_origins[:3].tolist()}")     # Show first 3  
    print(f"  World positions: {world_pos[:3].tolist()}")   # Show first 3
    
    # Create full pose with position and rotation
    default_quat = asset.data.default_root_state[env_ids, 3:7]
    full_pose = torch.cat([world_pos, default_quat], dim=1)
    
    # Set new positions using the proven method from respawn function
    asset.write_root_pose_to_sim(full_pose, env_ids)
    
    # Force visual update via USD on the exact prims we moved
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage:
            # Try to get correct prim paths
            prim_paths = None
            
            # Method 1: Check if target asset has prim_paths attribute
            if hasattr(asset, 'prim_paths') and asset.prim_paths:
                prim_paths = asset.prim_paths
            
            # Method 2: Try to construct from cfg.prim_path
            elif hasattr(asset, 'cfg') and hasattr(asset.cfg, 'prim_path'):
                base_path = asset.cfg.prim_path
                # Handle the template path properly - replace the template with actual env paths
                if "{ENV_REGEX_NS}" in base_path:
                    # Extract the suffix after the template
                    suffix = base_path.replace("{ENV_REGEX_NS}", "")
                    prim_paths = [f"/World/envs/env_{env_id}{suffix}" for env_id in range(env.num_envs)]
                else:
                    # If no template, assume it's already a specific path
                    prim_paths = [base_path for env_id in range(env.num_envs)]
            
            # Method 3: Final fallback with direct paths
            if prim_paths is None:
                prim_paths = [f"/World/envs/env_{env_id}/target_marker" for env_id in range(env.num_envs)]
            
            # Apply visual updates
            for i, env_id in enumerate(env_ids.tolist()):
                if env_id < len(prim_paths):
                    path = prim_paths[env_id]
                    prim = stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        xform = UsdGeom.Xformable(prim)
                        if xform:
                            pos = world_pos[i].detach().cpu().numpy()
                            # Reuse or add a Translate op
                            translate_op = next((op for op in xform.GetOrderedXformOps()
                                                 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                            if translate_op:
                                translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                            else:
                                xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            
            stage.GetRootLayer().dirty = True
    except Exception as e:
        # Silently handle USD visual update issues - they don't affect physics
        pass
    
    # Reset velocities to zero
    zero_velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(zero_velocities, env_ids)
    
    # Force immediate data update
    asset.data.root_pos_w[env_ids] = world_pos
    asset.data.root_quat_w[env_ids] = default_quat


def activate_surface_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    action_term_name: str = "base_gripper_action",
) -> None:
    """Activate a surface gripper to keep it permanently engaged.
    
    This function forces a surface gripper to close/activate, typically used
    for the base gripper to prevent robot sliding during manipulation tasks.
    """
    print(f"[DEBUG] Activating surface gripper '{action_term_name}' for envs: {env_ids}")
    
    # Get the action term for the surface gripper from action manager
    action_term = env.action_manager.get_term(action_term_name)
    
    # Create a "close gripper" action (-1.0 means close for surface grippers)
    close_action = torch.full((len(env_ids), 1), -1.0, device=env.device)
    
    print(f"[DEBUG] Processing close action: {close_action}")
    
    # Process the action to activate the gripper
    action_term.process_actions(close_action)


def keep_base_gripper_active(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    action_term_name: str = "base_gripper_action",
) -> None:
    """Keep the base surface gripper always active during simulation steps.
    
    This function ensures the base gripper remains engaged to prevent robot sliding.
    Should be called every step to maintain gripper activation.
    """
    # Only print every 100 calls to avoid spam
    if hasattr(keep_base_gripper_active, '_call_count'):
        keep_base_gripper_active._call_count += 1
    else:
        keep_base_gripper_active._call_count = 1
        
    # Get the action term for the surface gripper from action manager
    action_term = env.action_manager.get_term(action_term_name)
    
    # Create a "close gripper" action for specified environments (-1.0 means close)
    close_action = torch.full((len(env_ids), 1), -1.0, device=env.device)
    
    # Process the action to keep the gripper activated
    action_term.process_actions(close_action)
    
    # Enhanced debug output every 100 calls
    if keep_base_gripper_active._call_count % 100 == 1:
        print(f"[DEBUG] Maintaining surface gripper '{action_term_name}' for envs: {env_ids} (call #{keep_base_gripper_active._call_count})")
        
        # Check if we can access the gripper device
        try:
            # Try to get surface gripper information
            robot = env.scene["robot"]
            if hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w'):
                robot_z = robot.data.root_pos_w[env_ids, 2].mean().item()
                print(f"[DEBUG] Robot height: {robot_z:.3f}m")
                
            # Check for platform if it exists
            if "robot_platform" in env.scene:
                platform = env.scene["robot_platform"]
                if hasattr(platform, 'data') and hasattr(platform.data, 'root_pos_w'):
                    platform_z = platform.data.root_pos_w[env_ids, 2].mean().item()
                    print(f"[DEBUG] Platform height: {platform_z:.3f}m")
                    print(f"[DEBUG] Gap between robot and platform: {robot_z - platform_z:.3f}m")
                    
        except Exception as e:
            print(f"[DEBUG] Could not access position data: {e}")
            
        # Try to get gripper state information
        try:
            # Access the surface gripper device directly
            if hasattr(action_term, '_gripper'):
                gripper = action_term._gripper
                print(f"[DEBUG] Gripper type: {type(gripper)}")
                if hasattr(gripper, 'data'):
                    print(f"[DEBUG] Gripper has data attribute")
                if hasattr(gripper, 'is_grasping'):
                    is_grasping = gripper.is_grasping(env_ids)
                    print(f"[DEBUG] Gripper grasping state: {is_grasping}")
        except Exception as e:
            print(f"[DEBUG] Could not access gripper state: {e}")


def ptp_stability_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    threshold: float = 0.08,
    max_velocity: float = 0.3,  # More reasonable velocity threshold
) -> torch.Tensor:
    """Strong but achievable reward for staying still when within target threshold.
    
    Prevents oscillations by giving good rewards for stillness.
    Uses joint velocities as a proxy for overall movement.
    """
    # Get distance to target
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)  # (N,)
    
    # Get robot joint velocities as a proxy for movement
    robot: Articulation = env.scene["robot"]
    joint_vel = robot.data.joint_vel  # (N, num_joints)
    
    # Calculate overall movement magnitude using joint velocities
    movement_magnitude = torch.norm(joint_vel, dim=1)  # (N,)
    
    # Only reward stability when within threshold
    within_threshold = distance <= threshold  # (N,)
    
    # Balanced reward for stillness (linear instead of exponential)
    normalized_movement = torch.clamp(movement_magnitude / max_velocity, 0.0, 1.0)  # (N,)
    stability_reward = (1.0 - normalized_movement) * 3.0  # Strong 3x reward for stillness
    
    # Only apply reward when within threshold
    final_reward = stability_reward * within_threshold.float()  # (N,)
    
    return final_reward


def ptp_hovering_penalty(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    close_threshold: float = 0.12,
    reach_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalty for hovering near target without reaching it.
    
    This encourages the robot to either commit to reaching the target
    or move away, discouraging endless circling behavior.
    """
    distance = ee_to_target_distance(env, ee_frame_cfg, target_cfg).squeeze(-1)  # (N,)
    
    # Penalty if close but not reaching
    hovering_mask = (distance < close_threshold) & (distance > reach_threshold)
    penalty = torch.where(hovering_mask, torch.ones_like(distance), torch.zeros_like(distance))
    
    return penalty


def ptp_linear_motion_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    """Reward for moving in a straight line towards the target.
    
    This encourages direct, linear movement instead of curved or circular paths.
    Computes the dot product between EE velocity and the direction to target.
    """
    # Get current positions and velocities
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)
    target_pos = target.data.root_pos_w[:, :3]  # (N, 3)
    
    # Get EE velocity from the robot (we need the actual EE velocity)
    robot: Articulation = env.scene["robot"]
    
    # Find the end-effector link
    try:
        ee_link_idx = robot.find_bodies("link_eef")[0]
        ee_vel = robot.data.body_lin_vel_w[:, ee_link_idx, :]  # (N, 3)
    except (IndexError, KeyError):
        # Fallback: use overall robot velocity if EE link not found
        ee_vel = robot.data.root_lin_vel_w[:, :3]  # (N, 3)
    
    # Ensure all tensors have the correct number of environments
    num_envs = env.num_envs
    
    # Verify tensor shapes and fix if needed
    if ee_pos.shape[0] != num_envs:
        print(f"[DEBUG] ee_pos shape mismatch: {ee_pos.shape} vs {num_envs}")
        return torch.zeros(num_envs, device=env.device)
    
    if target_pos.shape[0] != num_envs:
        print(f"[DEBUG] target_pos shape mismatch: {target_pos.shape} vs {num_envs}")
        return torch.zeros(num_envs, device=env.device)
        
    if ee_vel.shape[0] != num_envs:
        print(f"[DEBUG] ee_vel shape mismatch: {ee_vel.shape} vs {num_envs}")
        return torch.zeros(num_envs, device=env.device)
    
    # Convert to local environment coordinates
    env_origins = env.scene.env_origins[:num_envs]  # Ensure correct size
    ee_pos_local = ee_pos - env_origins
    target_pos_local = target_pos - env_origins
    
    # Calculate direction to target (normalized)
    direction_to_target = target_pos_local - ee_pos_local  # (N, 3)
    distance = torch.norm(direction_to_target, dim=1)  # (N,)
    
    # Avoid division by zero
    safe_distance = torch.clamp(distance, min=1e-6)  # (N,)
    direction_normalized = direction_to_target / safe_distance.unsqueeze(-1)  # (N, 3)
    
    # Calculate velocity magnitude
    velocity_magnitude = torch.norm(ee_vel, dim=1)  # (N,)
    
    # Avoid division by zero for velocity
    safe_velocity_magnitude = torch.clamp(velocity_magnitude, min=1e-6)  # (N,)
    velocity_normalized = ee_vel / safe_velocity_magnitude.unsqueeze(-1)  # (N, 3)
    
    # Debug tensor shapes before multiplication
    if velocity_normalized.shape != direction_normalized.shape:
        print(f"[DEBUG] Shape mismatch: velocity_normalized {velocity_normalized.shape} vs direction_normalized {direction_normalized.shape}")
        return torch.zeros(num_envs, device=env.device)
    
    # Dot product: how aligned is velocity with direction to target
    alignment = torch.sum(velocity_normalized * direction_normalized, dim=1)  # (N,)
    
    # Only reward when moving (velocity > threshold) and close enough to target
    moving_threshold = 0.01  # m/s
    distance_threshold = 0.5   # Only reward linearity when within 50cm
    
    # All tensors should now be (N,) shape
    is_moving = velocity_magnitude > moving_threshold  # (N,)
    is_close = distance < distance_threshold  # (N,)
    
    # Apply conditions: only reward when moving and reasonably close to target
    linearity_reward = torch.where(
        is_moving & is_close,
        torch.clamp(alignment, 0.0, 1.0),  # Only positive alignment (moving towards target)
        torch.zeros_like(alignment)
    )
    
    return linearity_reward
