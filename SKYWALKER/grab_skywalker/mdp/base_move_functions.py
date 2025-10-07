# ptp_base_functions.py  (LOCAL-XY consistent)
# ---------------------------------------------------------------------
from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------- internal: fast body lookup ----------
def _body_index(env: "ManagerBasedRLEnv", body_name: str = "link_base") -> int:
    robot = env.scene["robot"]
    cache = getattr(_body_index, "_cache", {})
    if body_name in cache:
        return cache[body_name]
    idxs = robot.find_bodies(body_name)
    if not idxs:
        raise RuntimeError(f"Body '{body_name}' not found on robot.")
    i = idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0]
    cache[body_name] = int(i)
    _body_index._cache = cache
    return cache[body_name]

# ---------- GPT's EE-FRAME HELPER FUNCTIONS ----------
def _quat_yaw(qw, qx, qy, qz):
    """Extract yaw from quaternion (w, x, y, z)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)

def _rot2d(theta):
    """Returns 2x2 rotation matrix for each theta (batch)"""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], -1),
                     torch.stack([s,  c], -1)], -2)  # (N,2,2)
    return R

def ee_position_w(env, ee_body: str = "link_eef"):
    """End-effector position in world coordinates"""
    r = env.scene["robot"]
    i = _body_index(env, ee_body)
    return r.data.body_pos_w[:, i, :3]  # (N,3)

def ee_quat_w(env, ee_body: str = "link_eef"):
    """End-effector quaternion in world coordinates"""
    r = env.scene["robot"]
    i = _body_index(env, ee_body)
    return r.data.body_quat_w[:, i, :4]  # (N,4) (w,x,y,z)

def base_position_w(env):
    """Base position in world coordinates"""
    r = env.scene["robot"]
    i = _body_index(env, "link_base")
    return r.data.body_pos_w[:, i, :3]  # (N,3)

def base_linear_velocity_w(env):
    """Base linear velocity in world coordinates"""
    r = env.scene["robot"]
    i = _body_index(env, "link_base")
    return r.data.body_lin_vel_w[:, i, :3]  # (N,3)

def base_orientation_w(env):
    """Base orientation in world coordinates"""
    r = env.scene["robot"] 
    i = _body_index(env, "link_base")
    return r.data.body_quat_w[:, i, :4]  # (N,4)

def _canonicalize_quaternion(quat):
    """
    Canonicalize quaternion to ensure consistent representation.
    Makes w component positive by flipping sign if w < 0.
    
    Args:
        quat: (N, 4) tensor of quaternions in [w, x, y, z] format
    
    Returns:
        (N, 4) tensor of canonicalized quaternions
    """
    # Flip quaternion if w component is negative
    mask = quat[:, 0] < 0.0  # Check w component
    quat_canon = quat.clone()
    quat_canon[mask] = -quat_canon[mask]  # Flip entire quaternion
    return quat_canon

def base_orientation_w_canonical(env):
    """Base orientation in world coordinates - CANONICALIZED for training stability"""
    r = env.scene["robot"] 
    i = _body_index(env, "link_base")
    quat = r.data.body_quat_w[:, i, :4]  # (N,4) [w,x,y,z]
    return _canonicalize_quaternion(quat)

def target_pose_command_canonical(env, command_name: str = "target_pose"):
    """
    Target pose command with canonicalized quaternion for training stability.
    Returns 7D vector: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    """
    command = env.command_manager.get_command(command_name)  # (N, 7)
    pos = command[:, :3]  # (N, 3) position part
    quat = command[:, 3:7]  # (N, 4) quaternion part [w,x,y,z]
    quat_canon = _canonicalize_quaternion(quat)  # Canonicalize quaternion
    return torch.cat([pos, quat_canon], dim=1)  # (N, 7)

def goal_local(env, command_name="target_pose"):
    """Goal position in LOCAL coordinates"""
    return env.command_manager.get_command(command_name)[:, :3]  # (N,3) LOCAL-to-env

def goal_w(env, command_name="target_pose"):
    """Goal position in WORLD coordinates (LOCAL → WORLD with env origins)"""
    return env.scene.env_origins + goal_local(env, command_name)

def base_xy_in_ee_frame(env, ee_body="link_eef"):
    """Base XY expressed in the EE frame (EE is welded, so stable)"""
    b_w = base_position_w(env)[:, :2]         # (N,2)
    e_w = ee_position_w(env, ee_body)[:, :2]  # (N,2)
    qw, qx, qy, qz = ee_quat_w(env, ee_body).T
    yaw = _quat_yaw(qw, qx, qy, qz)           # (N,)
    R_T = _rot2d(-yaw)                         # (N,2,2)  rotate world→EE
    v = (b_w - e_w).unsqueeze(-1)             # (N,2,1)
    return torch.bmm(R_T, v).squeeze(-1)      # (N,2)

def goal_xy_in_ee_frame(env, command_name="target_pose", ee_body="link_eef"):
    """Goal XY expressed in the EE frame"""
    g_w = goal_w(env, command_name)[:, :2]
    e_w = ee_position_w(env, ee_body)[:, :2]
    qw, qx, qy, qz = ee_quat_w(env, ee_body).T
    yaw = _quat_yaw(qw, qx, qy, qz)
    R_T = _rot2d(-yaw)
    v = (g_w - e_w).unsqueeze(-1)
    return torch.bmm(R_T, v).squeeze(-1)      # (N,2)

def dist_xy_ee(env, command_name="target_pose", ee_body="link_eef"):
    """Distance from base to goal in EE frame (both measured relative to EE)"""
    b = base_xy_in_ee_frame(env, ee_body)     # (N,2)
    g = goal_xy_in_ee_frame(env, command_name, ee_body)
    return torch.norm(b - g, dim=1, keepdim=True)  # (N,1)

# ---------- REWARD FUNCTIONS IN EE FRAME ----------
def position_error_xy_ee(env, command_name="target_pose", ee_body="link_eef"):
    """Position error in EE frame (use with negative weight)"""
    return dist_xy_ee(env, command_name, ee_body).squeeze(-1)

def position_tanh_xy_ee(env, std=0.10, command_name="target_pose", ee_body="link_eef"):
    """Smooth saturating reward for reaching target in EE frame"""
    d = dist_xy_ee(env, command_name, ee_body).squeeze(-1)
    return torch.tanh((std - d) / (std + 1e-6))

def base_linear_motion_toward_goal_xy_ee(env, command_name="target_pose", ee_body="link_eef",
                                         moving_threshold=0.01, distance_threshold=0.5):
    """Reward for moving toward target in EE frame"""
    # Project velocity into EE frame
    v_w = base_linear_velocity_w(env)[:, :2]                # world
    e_w = ee_position_w(env, ee_body)[:, :2]
    qw, qx, qy, qz = ee_quat_w(env, ee_body).T
    yaw = _quat_yaw(qw, qx, qy, qz)
    R_T = _rot2d(-yaw)
    v_ee = torch.bmm(R_T, v_w.unsqueeze(-1)).squeeze(-1)    # (N,2)

    b = base_xy_in_ee_frame(env, ee_body)
    g = goal_xy_in_ee_frame(env, command_name, ee_body)
    vec = g - b
    dist = torch.norm(vec, dim=1).clamp_min(1e-6)
    dir_u = vec / dist.unsqueeze(-1)

    speed = torch.norm(v_ee, dim=1).clamp_min(1e-6)
    vel_u = v_ee / speed.unsqueeze(-1)

    align = (vel_u * dir_u).sum(dim=1)  # [-1,1]
    is_moving = speed > moving_threshold
    is_close = dist < distance_threshold
    return torch.where(is_moving & is_close, torch.clamp(align, 0.0, 1.0), torch.zeros_like(align))


# ---------- LEGACY COMPATIBILITY FUNCTIONS ----------
def base_position_relative_to_ee(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Base position relative to end-effector (base_pos - ee_pos) - LEGACY VERSION"""
    base_pos = base_position_w(env)
    ee_pos = ee_position_w(env, "link_eef")
    return base_pos - ee_pos

def base_orientation_relative_to_ee(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Base orientation relative to end-effector orientation - LEGACY VERSION"""
    base_quat = base_orientation_w(env)
    ee_quat = ee_quat_w(env, "link_eef")
    
    # For relative quaternion: q_rel = q_base * q_ee^(-1)
    # Quaternion inverse: q^(-1) = [w, -x, -y, -z] / |q|^2 (for unit quaternions, just conjugate)
    ee_quat_inv = torch.cat([ee_quat[:, 0:1], -ee_quat[:, 1:4]], dim=1)
    
    # Quaternion multiplication: q1 * q2
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=1)
    
    return quat_multiply(base_quat, ee_quat_inv)

# ---------- persistent initial base position buffer ----------
def get_initial_base_pos(env):
    if not hasattr(env, "_initial_base_pos"):
        # Set at episode start (reset)
        env._initial_base_pos = base_position_w(env).clone().detach()
    return env._initial_base_pos

def reset_initial_base_pos(env):
    env._initial_base_pos = base_position_w(env).clone().detach()

# ---------- EE-RELATIVE COMMAND SYSTEM ----------
def respawn_target_relative_to_ee(env, env_ids, command_name="target_pose"):
    """
    Sample a new target position relative to the END-EFFECTOR for each env in env_ids.
    This makes deployment much easier since EE is the fixed reference point!
    """
    device = env.scene.env_origins.device
    env_ids = torch.as_tensor(env_ids, device=device).view(-1).long()
    N = env_ids.numel()
    
    # Sample new target offsets relative to end-effector position
    # Use the ORIGINAL workspace ranges from UniformPoseCommandCfg for reachability!
    # These should match the initial spawn ranges: pos_x=(-0.15, 0.00), pos_y=(-0.30, 0.30)
    pos_x = torch.empty(N, device=device).uniform_(-0.15, 0.00)  # Match CommandsCfg
    pos_y = torch.empty(N, device=device).uniform_(-0.35, 0.35)  # Match CommandsCfg  
    
    # Get current base position (our spawning reference to stay within workspace)
    base_pos = base_position_w(env).index_select(0, env_ids)  # (N,3)
    
    # Since UniformPoseCommandCfg uses body_name="link_base", commands are relative to base
    # So pos_z=0.0 means "same height as base" which is exactly what we want
    pos_z = torch.zeros(N, device=device)  # Keep at same height as base
    roll = torch.zeros(N, device=device)
    pitch = torch.zeros(N, device=device) 
    yaw = torch.empty(N, device=device).uniform_(-0.9, 0.9)  # Match CommandsCfg: ±51 degrees
    
    # Create targets relative to base position (to stay within workspace limits)
    # This ensures targets spawn in front of the robot (negative X) where they're reachable
    target_relative_to_base = torch.stack([pos_x, pos_y, pos_z], dim=1)  # (N,3) - already in base frame
    
    # Compose quaternion from yaw
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2) 
    quat = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)  # (w,x,y,z)
    
    # Compose pose7 in base-relative coordinates 
    pose7_local = torch.cat([target_relative_to_base, quat], dim=1)
    
    # Set command (command system expects LOCAL coordinates)
    cmd = env.command_manager.get_command(command_name)
    cmd.index_copy_(0, env_ids, pose7_local)
    if hasattr(env.command_manager, "set_command"):
        env.command_manager.set_command(command_name, cmd)
    
    # print(f"   🎯 Respawned {N} targets RELATIVE TO EE: x∈[{pos_x.min():.2f},{pos_x.max():.2f}], y∈[{pos_y.min():.2f},{pos_y.max():.2f}] (EE-relative)")

# ---------- EE-RELATIVE DISTANCE CALCULATIONS ----------
def base_to_target_distance_ee_relative(env, command_name: str = "target_pose") -> torch.Tensor:
    """Distance from base to target, both measured relative to end-effector."""
    # Get base position relative to EE
    base_rel_ee = base_position_relative_to_ee(env)  # (N,3)
    
    # Get target position in LOCAL frame, convert to EE-relative
    target_local = _cmd_pos(env, command_name)  # (N,3) LOCAL
    env_origins = env.scene.env_origins  # (N,3)
    target_world = env_origins + target_local  # Convert to WORLD
    ee_pos = ee_position_w(env)  # (N,3) EE position in WORLD
    target_rel_ee = target_world - ee_pos  # (N,3) Target relative to EE
    
    # Calculate distance in EE-relative frame
    diff = base_rel_ee - target_rel_ee
    return torch.norm(diff, dim=1, keepdim=True)

def base_to_target_distance_xy_ee_relative(env, command_name: str = "target_pose") -> torch.Tensor:
    """XY distance from base to target, both measured relative to end-effector."""
    # Get base position relative to EE
    base_rel_ee = base_position_relative_to_ee(env)[:, :2]  # (N,2) XY only
    
    # Get target position in LOCAL frame, convert to EE-relative
    target_local = _cmd_pos(env, command_name)  # (N,3) LOCAL
    env_origins = env.scene.env_origins  # (N,3)
    target_world = env_origins + target_local  # Convert to WORLD
    ee_pos = ee_position_w(env)  # (N,3) EE position in WORLD
    target_rel_ee = target_world - ee_pos  # (N,3) Target relative to EE
    
    # Calculate XY distance in EE-relative frame
    diff_xy = base_rel_ee - target_rel_ee[:, :2]
    return torch.norm(diff_xy, dim=1, keepdim=True)

# ---------- sample new target at initial base frame (LEGACY - for compatibility) ----------
def respawn_target_at_initial_base(env, env_ids, command_name="target_pose"):
    """
    LEGACY: Sample a new target position relative to the initial base position for each env in env_ids.
    RECOMMEND: Use respawn_target_relative_to_ee() instead for better real-world deployment!
    """
    device = env.scene.env_origins.device
    env_ids = torch.as_tensor(env_ids, device=device).view(-1).long()
    N = env_ids.numel()
    
    # Sample new target offsets in LOCAL frame relative to initial base
    # Updated to match CommandsCfg ranges: pos_x=(-0.15, 0.00), pos_y=(-0.35, 0.35), yaw=(-0.9, 0.9)
    pos_x = torch.empty(N, device=device).uniform_(-0.15, 0.00)  # Match CommandsCfg: (-0.15, 0.00)
    pos_y = torch.empty(N, device=device).uniform_(-0.35, 0.35)  # Match CommandsCfg: (-0.35, 0.35)
    # Set targets at base height - commands are in LOCAL frame relative to env origins
    base_height = 0.172  # Updated for 172mm base height
    pos_z = torch.full((N,), base_height, device=device)  # Set at base height
    roll = torch.zeros(N, device=device)
    pitch = torch.zeros(N, device=device) 
    yaw = torch.empty(N, device=device).uniform_(-0.9, 0.9)  # Match CommandsCfg: (-0.9, 0.9) = ±51 degrees
    
    # Compose target pose in LOCAL frame (relative to env origins)
    target_local = torch.stack([pos_x, pos_y, pos_z], dim=1)
    
    # Compose quaternion from yaw
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2) 
    quat = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)  # (w,x,y,z)
    
    # Compose pose7 in LOCAL coordinates 
    pose7_local = torch.cat([target_local, quat], dim=1)
    
    # Set command (command system expects LOCAL coordinates)
    cmd = env.command_manager.get_command(command_name)
    cmd.index_copy_(0, env_ids, pose7_local)
    if hasattr(env.command_manager, "set_command"):
        env.command_manager.set_command(command_name, cmd)
    
    # print(f"   📍 Respawned {N} targets: x∈[{pos_x.min():.2f},{pos_x.max():.2f}], y∈[{pos_y.min():.2f},{pos_y.max():.2f}], z∈[{pos_z.min():.2f},{pos_z.max():.2f}] (LOCAL)")

# ---------- reward: dwell time success with EE-relative respawning ----------
def dwell_time_success_reward_ee_relative(env, command_name="target_pose", radius=0.015, bonus=10.0, dwell_time_s=1.0, exit_penalty=-15.0, staying_reward=0.5, max_staying_time=2.0):
    """
    Dwell time success reward using END-EFFECTOR relative positioning for tracking.
    Goals are respawned relative to END-EFFECTOR for consistent coordinate system!
    Added staying_reward: continuous reward for remaining in target zone after success.
    max_staying_time: Maximum time to stay in target before forcing new goal (for faster curriculum).
    """
    # Initialize dwell time tracking if needed
    if not hasattr(env, "_target_dwell_steps"):
        env._target_dwell_steps = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    if not hasattr(env, "_was_dwelling"):
        env._was_dwelling = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.bool)
    if not hasattr(env, "_completed_dwell"):
        env._completed_dwell = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.bool)
    if not hasattr(env, "_completed_time"):
        env._completed_time = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    
    # Initialize goal counter for tracking how many goals each env reaches
    if not hasattr(env, "_goals_reached"):
        env._goals_reached = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    
    # Calculate required steps for dwell time and max staying time
    dt = getattr(env.cfg, 'sim_dt', 0.01) * getattr(env.cfg, 'decimation', 1)
    required_steps = int(dwell_time_s / dt)
    max_staying_steps = int(max_staying_time / dt)
    
    # Check which envs are within target radius using EE-relative coordinates
    dists = base_to_target_distance_xy_ee_relative(env, command_name).squeeze(-1)
    within_radius = dists <= radius
    
    # Initialize reward tensor
    rewards = torch.zeros_like(dists)
    
    # Update dwell step counters and detect exits
    for env_id in range(env.scene.num_envs):
        if within_radius[env_id]:
            # Robot is within radius - increment counter
            env._target_dwell_steps[env_id] += 1
            env._was_dwelling[env_id] = True
            
            # If already completed dwell time, give staying reward and track completion time
            if env._completed_dwell[env_id]:
                env._completed_time[env_id] += 1
                rewards[env_id] = staying_reward
                
                # Force new target after max staying time
                if env._completed_time[env_id] >= max_staying_steps:
                    respawn_target_relative_to_ee(env, [env_id], command_name)
                    # print(f"   ⏰ Env {env_id} stayed too long ({env._completed_time[env_id] * dt:.2f}s) - forcing new target (goals reached: {env._goals_reached[env_id]})")
                    # Reset all tracking
                    env._target_dwell_steps[env_id] = 0
                    env._was_dwelling[env_id] = False
                    env._completed_dwell[env_id] = False
                    env._completed_time[env_id] = 0
                
        else:
            # Robot left target zone
            if env._was_dwelling[env_id] and env._target_dwell_steps[env_id] > 0:
                # Was dwelling but fell out - penalty!
                rewards[env_id] = exit_penalty
                # print(f"💥 Env {env_id} fell out of target after {env._target_dwell_steps[env_id]} steps ({env._target_dwell_steps[env_id] * dt:.2f}s)")
            
            # If completed target and now leaving, spawn new target
            if env._completed_dwell[env_id]:
                respawn_target_relative_to_ee(env, [env_id], command_name)
                # print(f"   🎯 Env {env_id} left completed target - spawning new target (goals reached: {env._goals_reached[env_id]})")
            
            # Reset all tracking when leaving target zone
            env._target_dwell_steps[env_id] = 0
            env._was_dwelling[env_id] = False
            env._completed_dwell[env_id] = False
            env._completed_time[env_id] = 0
    
    # Check for successful completions
    sufficient_dwell = env._target_dwell_steps >= required_steps
    just_completed = sufficient_dwell & (env._target_dwell_steps == required_steps)  # Only on the exact completion step
    
    if just_completed.any():
        success_ids = torch.nonzero(just_completed).view(-1)
        success_dists = dists.index_select(0, success_ids)
        dwell_times = env._target_dwell_steps.index_select(0, success_ids).float() * dt
        
        # Increment goal counter for successful environments
        env._goals_reached.index_add_(0, success_ids, torch.ones_like(success_ids, dtype=torch.long))
        goal_counts = env._goals_reached.index_select(0, success_ids)
        
        # DISABLE SUCCESS PRINTS FOR CPU PERFORMANCE
        # print(f"🎯 SUCCESS! {success_ids.numel()} envs completed target (radius={radius*1000:.1f}mm, dwell={dwell_time_s:.1f}s) [EE-RELATIVE TRACKING]")
        # print(f"   Success envs: {success_ids.tolist()}")
        # print(f"   Goal counts: {goal_counts.tolist()}")
        # print(f"   Final distances: {[f'{d:.3f}' for d in success_dists.tolist()]}")
        # print(f"   Dwell times: {[f'{t:.2f}s' for t in dwell_times.tolist()]}")
        
        # Give bonus for just completing + staying reward for this step
        rewards[success_ids] = bonus + staying_reward
        
        # Mark as completed but DON'T respawn yet - let them earn staying rewards!
        env._completed_dwell.index_fill_(0, success_ids, True)
        env._completed_time.index_fill_(0, success_ids, 0)  # Reset completion timer
        # print(f"   💰 Now earning +{staying_reward} per step for staying in target zone (max {max_staying_time:.1f}s)!")
    
    # Optional: Debug info for envs currently dwelling (DISABLE FOR CPU PERFORMANCE)
    # dwelling_envs = within_radius & (env._target_dwell_steps > 0) & ~sufficient_dwell
    # if dwelling_envs.any() and (getattr(env, '_debug_counter', 0) % 200 == 0):  # Print every 200 steps
    #     dwelling_ids = torch.nonzero(dwelling_envs).view(-1)
    #     dwelling_steps = env._target_dwell_steps.index_select(0, dwelling_ids)
    #     dwelling_times = dwelling_steps.float() * dt
    #     print(f"⏱️  {dwelling_envs.sum()} envs dwelling: progress {[f'{t:.2f}s' for t in dwelling_times.tolist()]} / {dwell_time_s:.1f}s required [EE-RELATIVE TRACKING]")
    
    # Increment debug counter
    if not hasattr(env, '_debug_counter'):
        env._debug_counter = 0
    env._debug_counter += 1
    
    return rewards

# ---------- reward: dwell time success with EE-relative TRACKING (but normal goal spawning) ----------
def dwell_time_success_reward_ee_relative_tracking(env, command_name="target_pose", radius=0.015, bonus=10.0, dwell_time_s=1.0, exit_penalty=-2.0):
    """
    Dwell time success reward using END-EFFECTOR relative TRACKING but normal goal spawning.
    Goals are spawned using CommandsCfg ranges (relative to initial base position).
    Only the distance calculations use EE-relative coordinates for easier real deployment.
    """
    # Initialize dwell time tracking if needed
    if not hasattr(env, "_target_dwell_steps"):
        env._target_dwell_steps = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    if not hasattr(env, "_was_dwelling"):
        env._was_dwelling = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.bool)
    
    # Calculate required steps for dwell time
    dt = getattr(env.cfg, 'sim_dt', 0.01) * getattr(env.cfg, 'decimation', 1)
    required_steps = int(dwell_time_s / dt)
    
    # Check which envs are within target radius using EE-relative coordinates
    dists = base_to_target_distance_xy_ee_relative(env, command_name).squeeze(-1)
    within_radius = dists <= radius
    
    # Initialize reward tensor
    rewards = torch.zeros_like(dists)
    
    # Update dwell step counters and detect exits
    for env_id in range(env.scene.num_envs):
        if within_radius[env_id]:
            # Robot is within radius - increment counter
            env._target_dwell_steps[env_id] += 1
            env._was_dwelling[env_id] = True
        else:
            # Robot left target zone
            if env._was_dwelling[env_id] and env._target_dwell_steps[env_id] > 0:
                # Was dwelling but fell out - penalty!
                rewards[env_id] = exit_penalty
                print(f"💥 Env {env_id} fell out of target after {env._target_dwell_steps[env_id]} steps ({env._target_dwell_steps[env_id] * dt:.2f}s)")
            
            # Reset counter and dwelling state
            env._target_dwell_steps[env_id] = 0
            env._was_dwelling[env_id] = False
    
    # Check for successful completions
    sufficient_dwell = env._target_dwell_steps >= required_steps
    just_completed = sufficient_dwell & (env._target_dwell_steps == required_steps)  # Only on the exact completion step
    
    if just_completed.any():
        success_ids = torch.nonzero(just_completed).view(-1)
        success_dists = dists.index_select(0, success_ids)
        dwell_times = env._target_dwell_steps.index_select(0, success_ids).float() * dt
        
        print(f"🎯 SUCCESS! {success_ids.numel()} envs completed target (radius={radius*1000:.1f}mm, dwell={dwell_time_s:.1f}s) [EE-RELATIVE TRACKING]")
        print(f"   Success envs: {success_ids.tolist()}")
        print(f"   Final distances: {[f'{d:.3f}' for d in success_dists.tolist()]}")
        print(f"   Dwell times: {[f'{t:.2f}s' for t in dwell_times.tolist()]}")
        
        # Give bonus for just completing
        rewards[success_ids] = bonus
        
        # Reset dwell tracking for successful envs
        env._target_dwell_steps.index_fill_(0, success_ids, 0)
        env._was_dwelling.index_fill_(0, success_ids, False)
        
        # DON'T respawn manually - let CommandsCfg handle it with proper ranges!
        print(f"   ✅ Targets will be respawned by CommandsCfg using base-relative ranges")
    
    # Optional: Debug info for envs currently dwelling (less frequent to avoid spam)
    dwelling_envs = within_radius & (env._target_dwell_steps > 0) & ~sufficient_dwell
    if dwelling_envs.any() and (getattr(env, '_debug_counter', 0) % 100 == 0):  # Print every 100 steps
        dwelling_ids = torch.nonzero(dwelling_envs).view(-1)
        dwelling_steps = env._target_dwell_steps.index_select(0, dwelling_ids)
        dwelling_times = dwelling_steps.float() * dt
        print(f"⏱️  {dwelling_envs.sum()} envs dwelling: progress {[f'{t:.2f}s' for t in dwelling_times.tolist()]} / {dwell_time_s:.1f}s required [EE-RELATIVE TRACKING]")
    
    # Increment debug counter
    if not hasattr(env, '_debug_counter'):
        env._debug_counter = 0
    env._debug_counter += 1
    
    return rewards

# ---------- reward: dwell time success with stability penalty (LEGACY) ----------
def dwell_time_success_reward(env, command_name="target_pose", radius=0.015, bonus=10.0, dwell_time_s=1.0, exit_penalty=-2.0):
    """
    LEGACY: Dwell time success reward using initial base position reference.
    RECOMMEND: Use dwell_time_success_reward_ee_relative() instead for better real-world deployment!
    """        
    # Initialize dwell time tracking if needed
    if not hasattr(env, "_target_dwell_steps"):
        env._target_dwell_steps = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    if not hasattr(env, "_was_dwelling"):
        env._was_dwelling = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.bool)
    
    # Calculate required steps for dwell time
    dt = getattr(env.cfg, 'sim_dt', 0.01) * getattr(env.cfg, 'decimation', 1)
    required_steps = int(dwell_time_s / dt)
    
    # Check which envs are within target radius
    dists = dist_xy_local(env, command_name).squeeze(-1)
    within_radius = dists <= radius
    
    # Initialize reward tensor
    rewards = torch.zeros_like(dists)
    
    # Update dwell step counters and detect exits
    for env_id in range(env.scene.num_envs):
        if within_radius[env_id]:
            # Robot is within radius - increment counter
            env._target_dwell_steps[env_id] += 1
            env._was_dwelling[env_id] = True
        else:
            # Robot left target zone
            if env._was_dwelling[env_id] and env._target_dwell_steps[env_id] > 0:
                # Was dwelling but fell out - penalty!
                rewards[env_id] = exit_penalty
                print(f"💥 Env {env_id} fell out of target after {env._target_dwell_steps[env_id]} steps ({env._target_dwell_steps[env_id] * dt:.2f}s)")
            
            # Reset counter and dwelling state
            env._target_dwell_steps[env_id] = 0
            env._was_dwelling[env_id] = False
    
    # Check for successful completions
    sufficient_dwell = env._target_dwell_steps >= required_steps
    just_completed = sufficient_dwell & (env._target_dwell_steps == required_steps)  # Only on the exact completion step
    
    if just_completed.any():
        success_ids = torch.nonzero(just_completed).view(-1)
        success_dists = dists.index_select(0, success_ids)
        dwell_times = env._target_dwell_steps.index_select(0, success_ids).float() * dt
        
        print(f"🎯 SUCCESS! {success_ids.numel()} envs completed target (radius={radius*1000:.1f}mm, dwell={dwell_time_s:.1f}s)")
        print(f"   Success envs: {success_ids.tolist()}")
        print(f"   Final distances: {[f'{d:.3f}' for d in success_dists.tolist()]}")
        print(f"   Dwell times: {[f'{t:.2f}s' for t in dwell_times.tolist()]}")
        
        # Give bonus for just completing
        rewards[success_ids] = bonus
        
        # Reset dwell tracking for successful envs and trigger respawn
        env._target_dwell_steps.index_fill_(0, success_ids, 0)
        env._was_dwelling.index_fill_(0, success_ids, False)
        
        # Respawn new targets
        respawn_target_at_initial_base(env, success_ids, command_name)
        print(f"   ✅ New targets spawned relative to initial base positions")
    
    # Optional: Debug info for envs currently dwelling (less frequent to avoid spam)
    dwelling_envs = within_radius & (env._target_dwell_steps > 0) & ~sufficient_dwell
    if dwelling_envs.any() and (getattr(env, '_debug_counter', 0) % 100 == 0):  # Print every 100 steps
        dwelling_ids = torch.nonzero(dwelling_envs).view(-1)
        dwelling_steps = env._target_dwell_steps.index_select(0, dwelling_ids)
        dwelling_times = dwelling_steps.float() * dt
        print(f"⏱️  {dwelling_envs.sum()} envs dwelling: progress {[f'{t:.2f}s' for t in dwelling_times.tolist()]} / {dwell_time_s:.1f}s required")
    
    # Increment debug counter
    if not hasattr(env, '_debug_counter'):
        env._debug_counter = 0
    env._debug_counter += 1
    
    return rewards

# Removed duplicate base_orientation_w function - using the one defined earlier

def base_linear_velocity_w_legacy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    i = _body_index(env, "link_base")
    return robot.data.body_lin_vel_w[:, i, :3]

def base_angular_velocity_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    i = _body_index(env, "link_base")
    return robot.data.body_ang_vel_w[:, i, :3]

# ---------- command accessor ----------
def _cmd_pos(env, command_name: str = "target_pose") -> torch.Tensor:
    """Command position in ENV-LOCAL frame (N,3)."""
    return env.command_manager.get_command(command_name)[:, :3]

# ---------- LOCAL-XY helpers ----------
def _origins_xy(env) -> torch.Tensor:
    return env.scene.env_origins[:, :2]  # (N,2)

def base_xy_local(env) -> torch.Tensor:
    """Base XY in ENV-LOCAL frame (translation by env_origins)."""
    return base_position_w(env)[:, :2] - _origins_xy(env)

def target_xy_local(env, command_name: str = "target_pose") -> torch.Tensor:
    """Target XY already in ENV-LOCAL frame."""
    return _cmd_pos(env, command_name)[:, :2]

def dist_xy_local(env, command_name: str = "target_pose") -> torch.Tensor:
    """Planar distance in LOCAL frame. Shape: (N,1)."""
    d = base_xy_local(env) - target_xy_local(env, command_name)
    return torch.norm(d, dim=1, keepdim=True)

# ---------- resets ----------
def set_start_pose(env, env_ids, *, joint_pos: dict):
    """
    At reset: set q and PD targets to the same angles so zero action doesn't move the arm.
    Signature matches EventTerm(func=..., mode="reset").
    """
    robot = env.scene["robot"]
    device = getattr(robot, "device", torch.device("cpu"))
    env_ids = torch.as_tensor(env_ids, device=device).view(-1).long()

    q_all = robot.data.joint_pos.clone()
    qd_all = torch.zeros_like(q_all)

    name_to_id = {n: i for i, n in enumerate(robot.joint_names)}
    for name, val in joint_pos.items():
        q_all[env_ids, name_to_id[name]] = float(val)

    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(position=q_all[env_ids], velocity=qd_all[env_ids], env_ids=env_ids)
    else:
        robot.set_joint_positions(q_all)
        robot.set_joint_velocities(qd_all)

    if hasattr(robot, "set_joint_position_targets"):
        robot.set_joint_position_targets(q_all)

def simple_robot_reset_event(
    env,
    env_ids,
    *,
    base_xy=(0.0, 0.0),
    base_z=0.501,  # Updated for 501mm base height
    base_rot=(1.0, 0.0, 0.0, 0.0),
    zero_joints: bool = False,
):
    """Place the base at the SAME local pose in each env (uses env_origins) and optionally zero joints."""
    import torch
    robot = env.scene["robot"]
    device = getattr(robot, "device", torch.device("cpu"))
    env_ids = torch.as_tensor(env_ids, device=device).view(-1).long()
    M = env_ids.numel()

    origins = env.scene.env_origins.index_select(0, env_ids)                   # (M,3)
    local   = torch.tensor([base_xy[0], base_xy[1], base_z],
                           device=device, dtype=torch.float32).repeat(M, 1)    # (M,3)
    pos_w   = origins + local                                                  # (M,3)
    quat_w  = torch.tensor(base_rot, device=device, dtype=torch.float32).repeat(M, 1)  # (M,4)
    pose7   = torch.cat([pos_w, quat_w], dim=-1)                               # (M,7)

    if hasattr(robot, "write_root_pose_to_sim"):
        robot.write_root_pose_to_sim(pose7, env_ids)
    else:
        z = torch.zeros_like(pos_w)
        root_state = torch.cat([pos_w, quat_w, z, z], dim=-1)                  # (M,13)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    if zero_joints:
        nj = robot.data.joint_pos.shape[-1]
        q  = torch.zeros((M, nj), device=device)
        qd = torch.zeros_like(q)
        if hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(position=q, velocity=qd, env_ids=env_ids)
        else:
            robot.set_joint_position_target(q, env_ids=env_ids)
            robot.set_joint_velocity_target(qd, env_ids=env_ids)

    env.scene.write_data_to_sim()

    # Print goal statistics before reset (DISABLE FOR CPU PERFORMANCE) 
    # if hasattr(env, '_goals_reached'):
    #     goals_reached = env._goals_reached.index_select(0, env_ids)
    #     if goals_reached.numel() > 0:
    #         print(f"📊 EPISODE RESET: Envs {env_ids.tolist()} completed goals: {goals_reached.tolist()}")
    #     # Reset goal counters for these environments
    #     env._goals_reached.index_fill_(0, env_ids, 0)

    # Reset goal counters for these environments (keep the functionality)
    if hasattr(env, '_goals_reached'):
        env._goals_reached.index_fill_(0, env_ids, 0)

    # Reset initial base pos buffer for multi-goal episodes
    reset_initial_base_pos(env)

# ---------- small quaternion helper ----------
def _quat_to_roll_pitch(qw, qx, qy, qz):
    # roll (x)
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr, cosr)
    # pitch (y)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return roll, pitch

# ---------- safety penalties ----------
def base_upright_penalty(env, roll_limit=math.radians(3.0), pitch_limit=math.radians(3.0)):
    quat = base_orientation_w(env)  # (N,4) [w,x,y,z]
    roll, pitch = _quat_to_roll_pitch(quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
    pen = torch.clamp(torch.abs(roll) - roll_limit, min=0) + torch.clamp(torch.abs(pitch) - pitch_limit, min=0)
    return pen

def base_lift_penalty(env, z_nom: float, deadband: float = 0.003):
    z = base_position_w(env)[:, 2]
    dz = torch.clamp(z - (z_nom + deadband), min=0.0)
    return dz * dz

def base_vertical_velocity_penalty(env):
    vz = base_linear_velocity_w(env)[:, 2]
    return vz * vz

# ---------- progress / success (LOCAL-XY) ----------
def position_progress_xy(env, command_name: str = "target_pose"):
    d = dist_xy_local(env, command_name).squeeze(-1)  # (N,)
    if not hasattr(env, "_prev_d_xy"):
        env._prev_d_xy = d.clone()
    if hasattr(env, "reset_buf"):
        env._prev_d_xy = torch.where(env.reset_buf.bool(), d, env._prev_d_xy)
    elif hasattr(env, "progress_buf"):
        env._prev_d_xy = torch.where((env.progress_buf == 0), d, env._prev_d_xy)
    prog = (env._prev_d_xy - d).clamp_min(0.0)
    env._prev_d_xy = d.detach().clone()
    return prog

def terminate_if_base_within_command_xy(env, command_name: str = "target_pose", radius: float = 0.005):
    """True when planar (x,y) LOCAL distance ≤ radius (m)."""
    return dist_xy_local(env, command_name).squeeze(-1) <= radius

def success_bonus_xy(env, command_name: str = "target_pose", radius: float = 0.005, bonus: float = 1.0):
    hit = terminate_if_base_within_command_xy(env, command_name, radius)
    return hit.float() * bonus
# --- put this near your other helpers (module-global) ---
_cmd_is_local_flag = None  # None = unknown, True = env-local, False = world

def _decide_cmd_frame_once(env, ids, cmd_pos_local3, base_w3):
    """Heuristic: pick the interpretation that puts target closer to base on average."""
    origins = env.scene.env_origins.index_select(0, ids)          # (M,3)
    candA_w = origins + cmd_pos_local3                             # env-local -> world
    candB_w = cmd_pos_local3                                       # already world
    dA = torch.norm(candA_w[:, :2] - base_w3[:, :2], dim=1).mean()
    dB = torch.norm(candB_w[:, :2] - base_w3[:, :2], dim=1).mean()
    return bool(dA <= dB)

def set_marker_to_command_at_base_height_event(
    env, env_ids, *, marker_name: str, command_name: str = "target_pose",
    base_body: str = "link_base", z_offset: float = 0.0,
):
    import torch
    marker = env.scene[marker_name]
    robot  = env.scene["robot"]

    cmd = env.command_manager.get_command(command_name)  # (N,7)
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        return

    ids = torch.as_tensor(env_ids, device=cmd.device).view(-1).long()
    if ids.numel() == 0:
        return

    # base world (for Z lock and heuristic)
    idxs = robot.find_bodies(base_body)
    base_idx = (idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0])
    base_w = robot.data.body_pos_w.index_select(0, ids)[:, base_idx, :3]  # (M,3)

    pos_cmd = cmd.index_select(0, ids)[:, :3].clone()     # what CommandMgr gives us
    quat    = cmd.index_select(0, ids)[:, 3:7].clone()

    global _cmd_is_local_flag
    if _cmd_is_local_flag is None:
        _cmd_is_local_flag = _decide_cmd_frame_once(env, ids, pos_cmd, base_w)

    if _cmd_is_local_flag:
        # env-local -> world
        origins = env.scene.env_origins.index_select(0, ids)
        pos_w = origins + pos_cmd
    else:
        # already world
        pos_w = pos_cmd

    # lock Z to base height (+ offset)
    pos_w[:, 2] = base_w[:, 2] + float(z_offset)

    # single world-space write; no manual USD xform editing
    try:
        marker.write_root_pose_to_sim(torch.cat([pos_w, quat], dim=-1), ids)
    except TypeError:
        marker.write_root_pose_to_sim(pos_w, quat, ids)
    env.scene.write_data_to_sim()

def fix_command_heights_to_base_event(
    env, env_ids, *, command_name: str = "target_pose", base_body: str = "link_base"
):
    """
    Fix command Z coordinates to match base height instead of being at ground level.
    This ensures targets spawn at the same height as the robot base.
    """
    import torch
    robot = env.scene["robot"]

    cmd = env.command_manager.get_command(command_name)  # (N,7)
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        return

    ids = torch.as_tensor(env_ids, device=cmd.device).view(-1).long()
    if ids.numel() == 0:
        return

    # Get base world position for Z reference
    idxs = robot.find_bodies(base_body)
    base_idx = (idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0])
    base_w = robot.data.body_pos_w.index_select(0, ids)[:, base_idx, :3]  # (M,3)

    # Update command Z to match base height
    cmd_updated = cmd.index_select(0, ids).clone()
    cmd_updated[:, 2] = base_w[:, 2]  # Set target Z to base Z
    
    # Write back to command manager
    cmd.index_copy_(0, ids, cmd_updated)
    if hasattr(env.command_manager, "set_command"):
        env.command_manager.set_command(command_name, cmd)

def print_target_xy_debug(
    env,
    env_ids,                           # provided by the manager
    *,
    command_name: str = "target_pose",
    body_name: str = "link_base",
    k: int = 6,                        # how many envs to print
    decimals: int = 3,
    success_radius: float = 0.005,     # 5 mm
):
    """Print target (local & world), base (world), distance, and success stats."""
    import torch

    robot = env.scene["robot"]
    cmd   = env.command_manager.get_command(command_name)  # (N,7), per-env LOCAL
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        print("[dbg] command buffer not ready")
        return

    # choose a small set of envs to print
    N = env.scene.num_envs
    k = max(1, min(k, N))
    ids = torch.arange(k, device=cmd.device, dtype=torch.long)

    # base pose (WORLD)
    idxs = robot.find_bodies(body_name)
    base_idx = (idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0])
    base_w = robot.data.body_pos_w.index_select(0, ids)[:, base_idx, :3]   # (k,3)

    # target (LOCAL → WORLD)
    tgt_local = cmd.index_select(0, ids)[:, :3]                            # (k,3) local
    origins   = env.scene.env_origins.index_select(0, ids)                 # (k,3) world
    tgt_w     = origins + tgt_local                                        # (k,3) world

    # planar distance (WORLD)
    d_xy = torch.norm((base_w[:, :2] - tgt_w[:, :2]), dim=1)               # (k,)

    # global stats over all envs (no printing of every env)
    all_d_xy = torch.norm(
        (robot.data.body_pos_w[:, base_idx, :2] - (env.scene.env_origins[:, :2] + cmd[:, :2])),
        dim=1
    )
    mean_d   = float(all_d_xy.mean())
    min_d    = float(all_d_xy.min())
    max_d    = float(all_d_xy.max())
    succ     = float((all_d_xy <= success_radius).float().mean())

    def _fmt3(v):
        return f"[{v[0]: .{decimals}f}, {v[1]: .{decimals}f}, {v[2]: .{decimals}f}]"

    print(f"[dbg] target debug (k={k}/{N})  mean_d={mean_d:.4f}  min={min_d:.4f}  max={max_d:.4f}  success@{success_radius*1000:.0f}mm={succ*100:.1f}%")
    for i, eid in enumerate(ids.tolist()):
        bp = base_w[i].tolist()
        tl = tgt_local[i].tolist()
        tw = tgt_w[i].tolist()
        print(f"  env {eid:4d}  base_w={_fmt3(bp)}  tgt_local={_fmt3(tl)}  tgt_world={_fmt3(tw)}  d_xy={d_xy[i]:.{decimals}f}")


# ---------- min-radius push (LOCAL-XY) ----------
def push_target_outside_radius_event(
    env,
    env_ids,
    *,
    command_name: str = "target_pose",
    base_body: str = "link_base",
    min_radius: float = 0.10,
    keep_z_at_base: bool = True,
):
    import torch
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)  # (N,7) LOCAL
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        return

    ids = torch.as_tensor(env_ids, device=cmd.device).view(-1).long()

    # base LOCAL XY = (base WORLD - origin WORLD)
    base_xy_loc = (
        robot.data.body_pos_w.index_select(0, ids)[:, _body_index(env, base_body), :2]
        - env.scene.env_origins.index_select(0, ids)[:, :2]
    )
    tgt_xy_loc  = cmd.index_select(0, ids)[:, :2].clone()

    dxy = tgt_xy_loc - base_xy_loc
    r = torch.linalg.norm(dxy, dim=1)
    too_close = r < min_radius
    if too_close.any():
        dir_xy = torch.where(
            r.unsqueeze(1) > 1e-9, dxy / r.clamp_min(1e-9).unsqueeze(1),
            torch.tensor([1.0, 0.0], device=cmd.device).repeat(dxy.shape[0], 1),
        )
        tgt_xy_loc = torch.where(
            too_close.unsqueeze(1),
            base_xy_loc + dir_xy * min_radius,
            tgt_xy_loc,
        )

    pos_local = cmd.index_select(0, ids)[:, :3].clone()
    pos_local[:, :2] = tgt_xy_loc

    if keep_z_at_base:
        base_z_loc = (
            robot.data.body_pos_w.index_select(0, ids)[:, _body_index(env, base_body), 2]
            - env.scene.env_origins.index_select(0, ids)[:, 2]
        )
        pos_local[:, 2] = base_z_loc

    new_pose7 = torch.cat([pos_local, cmd.index_select(0, ids)[:, 3:7]], dim=-1)
    cmd.index_copy_(0, ids, new_pose7)
    if hasattr(env.command_manager, "set_command"):
        try:
            env.command_manager.set_command(command_name, cmd)
        except Exception:
            pass

# ---------- WORLD (3D) helper & rewards kept for completeness ----------
def base_to_command_distance(env, command_name: str = "target_pose") -> torch.Tensor:
    """3D distance in WORLD (uses local->world translation)."""
    base_w = base_position_w(env)
    tgt_w  = _cmd_pos(env, command_name)
    origins = env.scene.env_origins
    return torch.norm((base_w - origins) - (tgt_w - origins), dim=1, keepdim=True)

def base_linear_motion_to_command(env, command_name: str = "target_pose",
                                  moving_threshold: float = 0.01, distance_threshold: float = 0.5) -> torch.Tensor:
    base_w = base_position_w(env)
    vel_w  = base_linear_velocity_w(env)
    tgt_w  = _cmd_pos(env, command_name)
    origins = env.scene.env_origins

    vec = (tgt_w - origins) - (base_w - origins)
    dist = torch.norm(vec, dim=1).clamp_min(1e-6)
    dir_unit = vec / dist.unsqueeze(-1)

    speed = torch.norm(vel_w, dim=1).clamp_min(1e-6)
    vel_unit = vel_w / speed.unsqueeze(-1)

    align = torch.sum(vel_unit * dir_unit, dim=1)          # [-1,1]
    is_moving = speed > moving_threshold
    is_close  = dist < distance_threshold
    return torch.where(is_moving & is_close, torch.clamp(align, 0.0, 1.0), torch.zeros_like(align))

def base_stability_to_command(env, command_name: str = "target_pose",
                              threshold: float = 0.08, max_velocity: float = 0.3, scale: float = 3.0) -> torch.Tensor:
    dist = base_to_command_distance(env, command_name).squeeze(-1)
    speed = torch.norm(base_linear_velocity_w(env), dim=1)
    norm = torch.clamp(speed / max_velocity, 0.0, 1.0)
    stable = (1.0 - norm) * scale
    return torch.where(dist <= threshold, stable, torch.zeros_like(stable))

def base_hovering_penalty_to_command(env, command_name: str = "target_pose",
                                     close_threshold: float = 0.12, reach_threshold: float = 0.08,
                                     penalty: float = 1.0) -> torch.Tensor:
    dist = base_to_command_distance(env, command_name).squeeze(-1)
    hovering = (dist < close_threshold) & (dist > reach_threshold)
    return torch.where(hovering, torch.full_like(dist, penalty), torch.zeros_like(dist))

def success_if_base_within_command(env, command_name: str = "target_pose", radius: float = 0.05):
    return (base_to_command_distance(env, command_name).squeeze(-1) <= radius)

# ---------- XY rewards (LOCAL-XY) ----------
def position_command_error_xy(env, command_name: str = "target_pose") -> torch.Tensor:
    """L2 XY error in LOCAL frame (N,). Use with a negative weight."""
    return dist_xy_local(env, command_name).squeeze(-1)

def position_command_error_tanh_xy(env, std: float = 0.1, command_name: str = "target_pose") -> torch.Tensor:
    """Smooth saturating XY term in LOCAL frame (N,)."""
    d = dist_xy_local(env, command_name).squeeze(-1)
    return torch.tanh((std - d) / (std + 1e-6))

def base_linear_motion_to_command_xy(
    env,
    command_name: str = "target_pose",
    moving_threshold: float = 0.01,
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """Cosine alignment of planar velocity with planar target direction (LOCAL-XY)."""
    b = base_xy_local(env)                         # (N,2)
    v = base_linear_velocity_w(env)[:, :2]         # same in local if no env rotation
    t = target_xy_local(env, command_name)         # (N,2)

    vec = (t - b)
    dist = torch.norm(vec, dim=1).clamp_min(1e-6)
    dir_u = vec / dist.unsqueeze(-1)

    speed = torch.norm(v, dim=1).clamp_min(1e-6)
    vel_u  = v / speed.unsqueeze(-1)

    align = (vel_u * dir_u).sum(dim=1)            # [-1,1]
    is_moving = speed > moving_threshold
    is_close  = dist < distance_threshold
    return torch.where(is_moving & is_close, torch.clamp(align, 0.0, 1.0), torch.zeros_like(align))

def base_stability_to_command_xy(
    env,
    command_name: str = "target_pose",
    threshold: float = 0.08,
    max_velocity: float = 0.3,
    scale: float = 3.0,
) -> torch.Tensor:
    """Reward small planar speed when within LOCAL-XY threshold."""
    d = dist_xy_local(env, command_name).squeeze(-1)
    speed_xy = torch.norm(base_linear_velocity_w(env)[:, :2], dim=1)
    norm = torch.clamp(speed_xy / max_velocity, 0.0, 1.0)
    stable = (1.0 - norm) * scale
    return torch.where(d <= threshold, stable, torch.zeros_like(stable))

def base_hovering_penalty_to_command_xy(
    env,
    command_name: str = "target_pose",
    close_threshold: float = 0.12,
    reach_threshold: float = 0.08,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Penalty for hovering in a ring (LOCAL-XY)."""
    d = dist_xy_local(env, command_name).squeeze(-1)
    hovering = (d < close_threshold) & (d > reach_threshold)
    return torch.where(hovering, torch.full_like(d, penalty), torch.zeros_like(d))

# ---------- optional marker reward (visual only) ----------
def follow_command_marker_rew(env, *, marker_name: str, command_name: str):
    """Writes marker pose to sim; returns zeros."""
    import torch
    marker = env.scene[marker_name]
    cmd = env.command_manager.get_command(command_name)  # (N,7)
    if cmd is None or cmd.shape[-1] != 7:
        return torch.zeros(env.num_envs, device=env.scene.env_origins.device)

    pos, quat = cmd[:, :3], cmd[:, 3:7]
    try:
        marker.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                      torch.arange(env.scene.num_envs, device=pos.device))
    except TypeError:
        try:
            marker.write_root_pose_to_sim(pos, quat,
                                          torch.arange(env.scene.num_envs, device=pos.device))
        except TypeError:
            zeros = torch.zeros_like(pos)
            root_state = torch.cat([pos, quat, zeros, zeros], dim=-1)
            marker.write_root_state_to_sim(root_state,
                                           env_ids=torch.arange(env.scene.num_envs, device=pos.device))
    env.scene.write_data_to_sim()
    return torch.zeros(env.scene.num_envs, device=pos.device)

def debug_print_goal_relative_to_initial_base(
    env,
    env_ids,  # provided by manager
    *,
    command_name: str = "target_pose",
    body_name: str = "link_base",
    first_n: int = 8,
    decimals: int = 3,
):
    """Print goal positions relative to INITIAL base position to verify spawning logic."""
    import torch
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)  # (N,7)
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        print(f"[dbg] command '{command_name}' not ready")
        return

    # Get initial base positions (stored during reset)
    initial_base_pos = get_initial_base_pos(env)  # (N,3) WORLD
    
    # Get current base positions
    idxs = robot.find_bodies(body_name)
    base_idx = (idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0])
    current_base_pos = robot.data.body_pos_w[:, base_idx, :3]  # (N,3) WORLD

    n = min(first_n, env.scene.num_envs)
    ids = torch.arange(n, device=cmd.device, dtype=torch.long)

    # Get target in LOCAL frame, convert to WORLD
    tgt_local = cmd.index_select(0, ids)[:, :3]                                 # (n,3) env-local
    origins = env.scene.env_origins.index_select(0, ids)                        # (n,3) world
    tgt_world = origins + tgt_local                                             # (n,3) world

    # Calculate distances
    initial_base_subset = initial_base_pos.index_select(0, ids)                 # (n,3)
    current_base_subset = current_base_pos.index_select(0, ids)                 # (n,3)
    
    # Goal relative to initial base position
    goal_rel_initial = tgt_world - initial_base_subset                          # (n,3)
    # Goal relative to current base position  
    goal_rel_current = tgt_world - current_base_subset                          # (n,3)
    
    # Distances
    dist_from_initial = torch.norm(goal_rel_initial, dim=1)                     # (n,)
    dist_from_current = torch.norm(goal_rel_current, dim=1)                     # (n,)

    def _fmt3(v):
        return f"[{v[0]: .{decimals}f}, {v[1]: .{decimals}f}, {v[2]: .{decimals}f}]"

    print(f"[dbg/goal_spawn] Goal positions relative to INITIAL base (n={n})")
    for i, eid in enumerate(ids.tolist()):
        goal_rel_init = goal_rel_initial[i].tolist()
        goal_rel_curr = goal_rel_current[i].tolist()
        d_init = dist_from_initial[i].item()
        d_curr = dist_from_current[i].item()
        print(f"  env {eid:4d}  goal_rel_initial={_fmt3(goal_rel_init)}  d={d_init:.3f}m  |  goal_rel_current={_fmt3(goal_rel_curr)}  d={d_curr:.3f}m")

# ---------- debug print (WORLD) ----------
def debug_print_bases_and_targets_world(
    env,
    env_ids,  # provided by manager
    *,
    command_name: str = "target_pose",
    body_name: str = "link_base",
    first_n: int = 8,
    decimals: int = 3,
):
    import torch
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)  # (N,7)
    if cmd is None or cmd.ndim != 2 or cmd.shape[1] != 7:
        print(f"[dbg] command '{command_name}' not ready")
        return

    idxs = robot.find_bodies(body_name)
    base_idx = (idxs[0][0] if isinstance(idxs[0], (list, tuple)) else idxs[0])

    n = min(first_n, env.scene.num_envs)
    ids = torch.arange(n, device=cmd.device, dtype=torch.long)

    base_w = robot.data.body_pos_w.index_select(0, ids)[:, base_idx, :3]       # (M,3) world
    tgt_local = cmd.index_select(0, ids)[:, :3]                                 # (M,3) env-local
    origins = env.scene.env_origins.index_select(0, ids)                        # (M,3) world
    tgt_w = origins + tgt_local                                                 # (M,3) world

    if not hasattr(env, "_dbg_prev_target_world"):
        env._dbg_prev_target_world = {}
    def _fmt3(v):
        return f"[{v[0]: .{decimals}f}, {v[1]: .{decimals}f}, {v[2]: .{decimals}f}]"

    print(f"[dbg/world] {n} envs — body='{body_name}' command='{command_name}'")
    for i, eid in enumerate(ids.tolist()):
        bp = base_w[i].tolist()
        tp = tgt_w[i].tolist()
        prev = env._dbg_prev_target_world.get(eid)
        drift = float(torch.norm(tgt_w[i] - prev)) if prev is not None else float('nan')
        env._dbg_prev_target_world[eid] = tgt_w[i].clone()
        print(f"  env {eid:4d}  base_w={_fmt3(bp)}  target_w={_fmt3(tp)}  ||Δtarget||={drift:.{decimals}f}")

# ---------- safety termination (optional) ----------
def terminate_if_tilted_or_lifted(env, z_nom: float, z_max_delta: float, roll_max: float, pitch_max: float):
    pos = base_position_w(env)
    quat = base_orientation_w(env)
    roll, pitch = _quat_to_roll_pitch(quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
    lifted = pos[:, 2] > (z_nom + z_max_delta)
    tilted = (torch.abs(roll) > roll_max) | (torch.abs(pitch) > pitch_max)
    return lifted | tilted

# --- yaw helpers ---
def _quat_to_yaw(qw, qx, qy, qz):
    # yaw (about +Z)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    return torch.atan2(siny, cosy)

def _yaw_error_signed(env, command_name: str = "target_pose") -> torch.Tensor:
    """Signed shortest-angle yaw error in radians, in [-pi, pi]."""
    q_b = base_orientation_w(env)  # (N,4) [w,x,y,z]
    yaw_b = _quat_to_yaw(q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3])

    cmd = env.command_manager.get_command(command_name)  # (N,7) local pose
    q_c = cmd[:, 3:7]
    yaw_c = _quat_to_yaw(q_c[:, 0], q_c[:, 1], q_c[:, 2], q_c[:, 3])

    diff = yaw_b - yaw_c
    # wrap to [-pi, pi] robustly
    return torch.atan2(torch.sin(diff), torch.cos(diff))

# --- orientation rewards ---
def orientation_error_yaw(env, command_name: str = "target_pose") -> torch.Tensor:
    """Absolute yaw error in radians (use with negative weight if you want a penalty)."""
    return torch.abs(_yaw_error_signed(env, command_name))

def orientation_tanh_yaw(env,
                         command_name: str = "target_pose",
                         std_deg: float = 8.0,
                         gate_with_distance: bool = True,
                         gate_sigma: float = 0.20) -> torch.Tensor:
    """
    Positive reward ~1 when yaw aligned, ~0 when far.
    Optionally gate by planar proximity so orientation matters more near the goal.
    """
    err = torch.abs(_yaw_error_signed(env, command_name))
    std = math.radians(std_deg)
    r = torch.tanh((std - err) / (std + 1e-6))  # -> 1 near 0 error, ~0 when err >> std

    if gate_with_distance:
        d = dist_xy_local(env, command_name).squeeze(-1)   # LOCAL-XY distance
        gate = torch.exp(-(d / gate_sigma) ** 2)
        r = r * gate
    return r

# --- optional orientation terminations/bonuses ---
def terminate_if_yaw_within_command(env, command_name: str = "target_pose", yaw_thresh_deg: float = 2.0):
    th = math.radians(yaw_thresh_deg)
    return torch.abs(_yaw_error_signed(env, command_name)) <= th

def terminate_if_reached_pose_xy_yaw(env,
                                     command_name: str = "target_pose",
                                     pos_radius: float = 0.005,
                                     yaw_thresh_deg: float = 2.0):
    pos_ok = dist_xy_local(env, command_name).squeeze(-1) <= pos_radius
    yaw_ok = torch.abs(_yaw_error_signed(env, command_name)) <= math.radians(yaw_thresh_deg)
    return pos_ok & yaw_ok

def success_bonus_xy_yaw(env, command_name: str = "target_pose",
                         pos_radius: float = 0.005, yaw_thresh_deg: float = 2.0, bonus: float = 1.0):
    hit = terminate_if_reached_pose_xy_yaw(env, command_name, pos_radius, yaw_thresh_deg)
    return hit.float() * bonus

# ---------- EE-RELATIVE REWARD FUNCTIONS ----------
def position_command_error_ee_relative(env, command_name: str = "target_pose") -> torch.Tensor:
    """Position error relative to end-effector (better for real deployment)."""
    return base_to_target_distance_ee_relative(env, command_name).squeeze(-1)

def position_command_error_xy_ee_relative(env, command_name: str = "target_pose") -> torch.Tensor:
    """XY position error relative to end-effector (better for real deployment)."""
    return base_to_target_distance_xy_ee_relative(env, command_name).squeeze(-1)

def position_command_error_tanh_xy_ee_relative(env, std: float = 0.1, command_name: str = "target_pose") -> torch.Tensor:
    """Smooth saturating XY reward relative to end-effector."""
    d = base_to_target_distance_xy_ee_relative(env, command_name).squeeze(-1)
    return torch.tanh((std - d) / (std + 1e-6))

def base_linear_motion_to_command_ee_relative(
    env,
    command_name: str = "target_pose",
    moving_threshold: float = 0.01,
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for moving toward target in EE-relative coordinates."""
    # Base position relative to EE
    base_rel_ee = base_position_relative_to_ee(env)[:, :2]  # (N,2) XY only
    
    # Base velocity in world frame (EE doesn't move much, so world ≈ EE frame for velocity)
    v = base_linear_velocity_w(env)[:, :2]  # (N,2)
    
    # Target position relative to EE
    target_local = _cmd_pos(env, command_name)  # (N,3) LOCAL
    env_origins = env.scene.env_origins  # (N,3)
    target_world = env_origins + target_local  # Convert to WORLD
    ee_pos = ee_position_w(env)  # (N,3) EE position in WORLD
    target_rel_ee = (target_world - ee_pos)[:, :2]  # (N,2) Target relative to EE, XY only

    # Direction and distance to target in EE frame
    vec = target_rel_ee - base_rel_ee
    dist = torch.norm(vec, dim=1).clamp_min(1e-6)
    dir_u = vec / dist.unsqueeze(-1)

    # Velocity alignment
    speed = torch.norm(v, dim=1).clamp_min(1e-6)
    vel_u = v / speed.unsqueeze(-1)

    align = (vel_u * dir_u).sum(dim=1)  # [-1,1]
    is_moving = speed > moving_threshold
    is_close = dist < distance_threshold
    return torch.where(is_moving & is_close, torch.clamp(align, 0.0, 1.0), torch.zeros_like(align))

def position_progress_xy_ee_relative(env, command_name: str = "target_pose"):
    """Position progress reward in EE-relative coordinates."""
    d = base_to_target_distance_xy_ee_relative(env, command_name).squeeze(-1)  # (N,)
    if not hasattr(env, "_prev_d_xy_ee"):
        env._prev_d_xy_ee = d.clone()
    if hasattr(env, "reset_buf"):
        env._prev_d_xy_ee = torch.where(env.reset_buf.bool(), d, env._prev_d_xy_ee)
    elif hasattr(env, "progress_buf"):
        env._prev_d_xy_ee = torch.where((env.progress_buf == 0), d, env._prev_d_xy_ee)
    prog = (env._prev_d_xy_ee - d).clamp_min(0.0)
    env._prev_d_xy_ee = d.detach().clone()
    return prog


def base_height_penalty(env, target_height: float = 0.501, penalty_scale: float = -10.0) -> torch.Tensor:
    """
    Penalty for lifting the base off the ground.
    
    Args:
        env: Environment instance
        target_height: Desired height for the base (typically around 0.3m)
        penalty_scale: Negative penalty multiplier for height deviations
        
    Returns:
        Penalty tensor (negative rewards for height deviations)
    """
    # Get base position in world coordinates
    base_pos = env.scene.sensors["robot"].data.body_pos_w[:, 1]  # Use second body (link_base)
    
    # Get current height (Z coordinate)
    current_height = base_pos[:, 2]  # (N,)
    
    # Calculate height deviation from target
    height_error = torch.abs(current_height - target_height)
    
    # Apply quadratic penalty for height deviations
    penalty = penalty_scale * height_error.pow(2)
    
    return penalty


def joint_position_action_with_warmup(env, action: torch.Tensor, warmup_time: float = 1.0) -> torch.Tensor:
    """
    Joint position action that disables movements during initial warmup period.
    This allows the robot to settle on the ground before accepting commands.
    
    Args:
        env: Environment instance
        action: Raw action commands from policy
        warmup_time: Time in seconds to disable actions after reset
        
    Returns:
        Modified action (zero during warmup, normal afterwards)
    """
    # Get time step info
    dt = getattr(env.cfg, 'sim_dt', 0.01) * getattr(env.cfg, 'decimation', 1)
    warmup_steps = int(warmup_time / dt)
    
    # Track episode steps since reset
    if not hasattr(env, '_episode_steps'):
        env._episode_steps = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    
    # Reset step counter on episode reset
    if hasattr(env, 'reset_buf'):
        reset_mask = env.reset_buf.bool()
        env._episode_steps = torch.where(reset_mask, torch.zeros_like(env._episode_steps), env._episode_steps)
    
    # Increment step counter
    env._episode_steps += 1
    
    # Mask actions during warmup period
    in_warmup = env._episode_steps <= warmup_steps
    warmup_mask = in_warmup.unsqueeze(-1).expand_as(action)  # (N,) -> (N, action_dim)
    
    # During warmup: use current joint positions (no movement)
    # After warmup: use policy actions
    robot = env.scene["robot"]
    current_positions = robot.data.joint_pos
    
    # Select warmup (current position) or normal action
    modified_action = torch.where(warmup_mask, current_positions, action)
    
    # Debug print for first few warmup steps
    if in_warmup.any() and (env._episode_steps[0] <= 5 or env._episode_steps[0] % 20 == 0):
        warmup_envs = in_warmup.sum().item()
        if warmup_envs > 0:
            remaining_time = (warmup_steps - env._episode_steps.max().item()) * dt
            # print(f"🛡️  {warmup_envs} envs in warmup: {remaining_time:.2f}s remaining")
    
    return modified_action


def robot_warmup_event(env, env_ids, warmup_time: float = 1.0):
    """
    Event that maintains robot in stable pose during warmup period.
    Overrides joint positions to prevent movement during settling.
    
    Args:
        env: Environment instance
        env_ids: Environment IDs to apply warmup to
        warmup_time: Duration of warmup period in seconds
    """
    # Get time step info
    dt = getattr(env.cfg, 'sim_dt', 0.01) * getattr(env.cfg, 'decimation', 1)
    warmup_steps = int(warmup_time / dt)
    
    # Track episode steps since reset for each environment
    if not hasattr(env, '_episode_steps'):
        env._episode_steps = torch.zeros(env.scene.num_envs, device=env.scene.env_origins.device, dtype=torch.long)
    
    # Reset step counter on episode reset
    if hasattr(env, 'reset_buf'):
        reset_mask = env.reset_buf.bool()
        env._episode_steps = torch.where(reset_mask, torch.zeros_like(env._episode_steps), env._episode_steps)
    
    # Increment step counter for active environments
    env._episode_steps += 1
    
    # Check which environments are still in warmup
    in_warmup_mask = env._episode_steps <= warmup_steps
    warmup_env_ids = torch.nonzero(in_warmup_mask).view(-1)
    
    if warmup_env_ids.numel() > 0:
        # Get robot asset
        robot = env.scene["robot"]
        
        # Get current joint positions
        current_joint_pos = robot.data.joint_pos
        
        # Set desired stable joint positions (from config)
        stable_joint_pos = current_joint_pos.clone()
        stable_joint_pos[warmup_env_ids, 0] = -0.01047198  # joint1
        stable_joint_pos[warmup_env_ids, 1] =  0.356047    # joint2
        stable_joint_pos[warmup_env_ids, 2] =  0.0523599   # joint3
        stable_joint_pos[warmup_env_ids, 3] =  0.764455    # joint4
        stable_joint_pos[warmup_env_ids, 4] =  0.0         # joint5
        stable_joint_pos[warmup_env_ids, 5] = -0.942478    # joint6
        stable_joint_pos[warmup_env_ids, 6] =  0.0         # joint7
        
        # Apply stable positions during warmup
        robot.set_joint_position_target(stable_joint_pos[warmup_env_ids], joint_ids=None, env_ids=warmup_env_ids)
        
        # Debug info (less frequent to avoid spam) - COMMENTED OUT FOR SPEED
        # if env._episode_steps[0] <= 5 or env._episode_steps[0] % 40 == 0:
        #     remaining_time = max(0, (warmup_steps - env._episode_steps.max().item()) * dt)
        #     print(f"🛡️  {warmup_env_ids.numel()} envs in warmup: {remaining_time:.2f}s remaining")

def debug_print_anchor_position(
    env,
    env_ids,  # provided by manager
    *,
    print_anchor: bool = True,
):
    """Debug function to print anchor position for height analysis."""
    try:
        # Only print for first environment to avoid spam
        if len(env_ids) == 0 or env_ids[0] != 0:
            return
            
        anchor = env.scene["anchor"]
        anchor_pos = anchor.data.root_pos_w[0, :3]  # Get position of first env
        
        # Also get robot base for comparison
        robot = env.scene["robot"]
        base_idx = robot.find_bodies("link_base")[0][0]
        base_pos = robot.data.body_pos_w[0, base_idx, :3]
        
        print(f"🔗 ANCHOR DEBUG:")
        print(f"   Anchor position (world): [{anchor_pos[0]:.3f}, {anchor_pos[1]:.3f}, {anchor_pos[2]:.3f}]")
        print(f"   Base position (world):   [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        print(f"   Height difference: {anchor_pos[2] - base_pos[2]:.3f}m")
        print(f"   Anchor height above ground: {anchor_pos[2]:.3f}m")
        
    except Exception as e:
        print(f"❌ Error getting anchor position: {e}")
