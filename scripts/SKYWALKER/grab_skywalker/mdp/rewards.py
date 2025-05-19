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
from isaacsim.core.utils.prims import get_prim_at_path
import isaacsim.core.utils.prims as prim_utils
import omni.usd
import omni.physx as physx
from isaacsim.core.utils.prims import get_prim_at_path
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Sdf, Usd



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




from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


def goal_potential(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    std: float = 0.75,
) -> torch.Tensor:
    """Smooth potential Φ(s) = exp(−‖x_base − x_goal‖ / std)."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # (N, 2)
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]  # (N, 2)
    dist = torch.norm(root_pos - goal_pos, dim=1)
    return torch.exp(-dist / std)


def mount_affinity(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    grip_term: str,
    std: float = 0.30,
) -> torch.Tensor:
    """Tiny lure toward *cube_cfg* if that cube is **not currently grasped**."""
    near = object_ee_distance(env, cube_cfg=cube_cfg, std=std)  # (N,)
    grasped = is_grasping_fixed_object(env, cube_cfg=cube_cfg, grip_term=grip_term)
    return near * (1.0 - grasped)



def drag_carry_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    grip_term: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg  = SceneEntityCfg("goal_marker"),
    std: float = 1.5,
) -> torch.Tensor:
    """
    Reward = 𝟙[holding *cube_cfg*] · exp(−‖base − goal‖ / std).

    • Only gives you credit for *moving your base* closer to the goal while you’re holding the cube.
    • Smooth shaping via exp(−d/std) so you always get a gradient.
    """
    # 1) Are you grasping the cube?
    hold = is_grasping_fixed_object(
        env,
        cube_cfg=cube_cfg,
        grip_term=grip_term,
    ).view(-1)  # (N,)

    # 2) How far is your base from the goal right now?
    base_xy = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # (N,2)
    goal_xy = env.scene[goal_cfg.name].data.root_pos_w[:, :2]   # (N,2)
    dist    = torch.norm(base_xy - goal_xy, dim=1)             # (N,)

    # 3) Shape it with an exp: closer → larger reward
    potential = torch.exp(-dist / std)                         # (N,)

    return hold * potential





def object_position_in_robot_root_frame(
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



def object_ee_distance(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg,
    std: float = 0.5,
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




def hold_far_cube_penalty(
    env: "ManagerBasedRLEnv",
    cube1_cfg: SceneEntityCfg = SceneEntityCfg("cube1"),
    cube2_cfg: SceneEntityCfg = SceneEntityCfg("cube2"),
    grip_term: str = "gripper_action",
    goal_cfg:  SceneEntityCfg = SceneEntityCfg("goal_marker"),
    reach_thresh: float = 0.6,     # outer radius   (m)
    falloff:      float = 0.25,    # width of the sigmoid (m)
    lam:          float = 1.0,     # scale of the penalty
    release_bonus: float = 2.0,    # reward for opening in the zone
    warmup_steps: int   = 150,     # ignore first ~1.5 s of every episode
) -> torch.Tensor:
    """
    • −λ · σ(zone)  if holding the cube that is farther from the goal  
    • +release_bonus·σ(zone)  if *no* cube is held (helps PPO notice the gradient)

    σ(zone) is a smooth step that reaches ~0.5 at reach_thresh and
    approaches 1.0 as the base sits exactly midway between both cubes.
    """
    # ------------------------------------------------------------------
    # 1) Grasp flags  →  (N,)    (helper returns (N,1) on older versions)
    # ------------------------------------------------------------------
    hold1 = is_grasping_fixed_object(env, cube_cfg=cube1_cfg,
                                     grip_term=grip_term).view(-1)
    hold2 = is_grasping_fixed_object(env, cube_cfg=cube2_cfg,
                                     grip_term=grip_term).view(-1)

    # ------------------------------------------------------------------
    # 2) Which cube is farther from the goal?
    # ------------------------------------------------------------------
    c1_xy   = env.scene[cube1_cfg.name].data.target_pos_w[..., 0, :2]   # (N,2)
    c2_xy   = env.scene[cube2_cfg.name].data.target_pos_w[..., 0, :2]
    goal_xy = env.scene[goal_cfg.name].data.root_pos_w[:, :2]

    farther1 = (torch.norm(c1_xy - goal_xy, dim=1) >
                torch.norm(c2_xy - goal_xy, dim=1)).float()             # (N,)
    farther2 = 1.0 - farther1

    hold_wrong = hold1 * farther1 + hold2 * farther2                   # (N,)

    # ------------------------------------------------------------------
    # 4) Optional bonus for releasing inside the zone
    # ------------------------------------------------------------------
    # 3)  Zone only cares about **Cube 2 distance** and only **after warm-up**
    base_xy   = env.scene["robot"].data.root_pos_w[:, :2]
    d2        = torch.norm(base_xy - c2_xy, dim=1)               # (N,)
    zone_raw  = torch.sigmoid((reach_thresh - d2) / falloff)     # (N,)
    reset_flags = getattr(env, "reset_buf", None)          # (N,)  1 → has just reset
    if not hasattr(hold_far_cube_penalty, "_step_timer"):
        hold_far_cube_penalty._step_timer = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )

    timer = hold_far_cube_penalty._step_timer
    timer += 1                                             # advance every sim step
    if reset_flags is not None:
        timer *= (1 - reset_flags.int())                   # zero where env just reset

    active = (timer > warmup_steps).float()                # (N,)
    zone_w = zone_raw * active                              # (N,)

    released  = 1.0 - (hold1 + hold2)
    bonus     = release_bonus * released * zone_w

    return -lam * hold_wrong * zone_w + bonus



def is_grasping_fixed_object(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube1"),
    tol: float = 0.15,
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
    cube_ent = env.scene[cube_cfg.name]
    # print(f"[DEBUG] cube '{cube_cfg.name}' → transformer prim = {cube_ent.cfg.prim_path}")
    # print(f"[DEBUG] cube '{cube_cfg.name}' world_pos = {cube_ent.data.target_pos_w[0,0].tolist()}")
    # d0 = torch.norm(ee_pos[0] - cube_pos[0]).item()
    # print(f"[DEBUG] grasp_dist={d0:.3f}  thresh={tol}")

    dist = torch.norm(ee_pos - cube_pos, dim=1)
    is_near = (dist < tol).float()

    
    #print(f"[DEBUG] is_closed: {is_closed[0].item()}, is_near: {is_near[0].item()}, reward: {reward[0].item()}")


    return is_closed * is_near  # Only reward if closed near the cube

def simultaneous_gripper_penalty(
    env: ManagerBasedRLEnv,
    grip_term_1: str = "gripper_action",
    grip_term_2: str = "gripper_action2",
    grace_steps: int = 30,          # ~2 s at 15 Hz
    penalty: float = -1.0,
) -> torch.Tensor:
    """
    −penalty  if **both** grippers are open for longer than <grace_steps>.
    No cost if they are both closed (or in opposite states).
    """
    try:
        term1 = env.action_manager.get_term(grip_term_1)
        term2 = env.action_manager.get_term(grip_term_2)
    except KeyError:                               # gripper terms missing
        return torch.zeros(env.num_envs, device=env.device)

    g1_closed = term1.is_closed().float()          # 1 = closed, 0 = open
    g2_closed = term2.is_closed().float()

    timers = term1._shared_gripper_state_timer     # shared int32 counter
    grace_mask = (timers >= grace_steps).float()   # 1 after grace period

    both_open = (1.0 - g1_closed) * (1.0 - g2_closed)   # 1 only if both open
    penalty_applied = penalty * both_open * grace_mask

    # ------------------------------------------------------------------
    # Debug print (first env only)
    # ------------------------------------------------------------------
   
    # print(
    #     f"[DBG] g1_closed={g1_closed[0].item():.0f}  "
    #     f"g2_closed={g2_closed[0].item():.0f}  "
    #     f"timer={timers[0]:3d}  "
    #     f"penalty={penalty_applied[0].item():+.1f}"
    # )

    return penalty_applied

def gripper2_docking_reward(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg,
    robot_cfg:  SceneEntityCfg = SceneEntityCfg("robot"),
    grip_term:  str = "gripper_action2",
    std: float  = 0.25,    # width of the smooth “approach” reward
    tol: float  = 0.12,    # close-enough radius for the bonus
    close_bonus: float = 1.0,
    far_penalty: float = -0.3,
) -> torch.Tensor:
    """
    Smooth reward for *closing* gripper 2 when the base is near <target_cfg>.

    • Always gives exp(−d / std) so the critic sees a gradient.
    • Adds +close_bonus only if the close happens inside tol.
      (If the gripper was already closed when it arrived, the bonus is NOT paid again;
       PPO will still learn to keep it closed because the penalty below vanishes.)
    • Adds −far_penalty if it closes while still outside tol.
    """
    robot  = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    grip   = env.action_manager.get_term(grip_term)
    # --- XY distance base ↔ target ------------------------------------
    robot_xy = robot.data.root_pos_w[:, :2]              # (N, 2)

    # target may be RigidObject (root_pos_w) or FrameTransformer (target_pos_w)
    if hasattr(target.data, "root_pos_w"):
        target_xy = target.data.root_pos_w[:, :2]        # (N, 2)
    elif hasattr(target.data, "target_pos_w"):
        target_xy = target.data.target_pos_w[..., 0, :2] # (N, 2)
    else:                                                # fallback: API get_world_poses
        target_xy, _ = target.get_world_poses()
        target_xy = target_xy[:, :2]

    d = torch.norm(robot_xy - target_xy, dim=1)          # (N,)


    base_reward = torch.exp(-d / std)          # [0,1]
    closed      = grip.is_closed().float()     # 0/1
    near        = (d < tol).float()            # 0/1

    return base_reward + close_bonus * closed * near + far_penalty * closed * (1 - near)




def ee_approach_alignment_in_base(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube1"),
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
    asset_cfg: SceneEntityCfg           = SceneEntityCfg("robot"),
    min_collisions: int                 = 2,      # ignore 0–1 minor grazes
    penalty_per_collision: float        = -1.0,
) -> torch.Tensor:
    # grab the robot articulation
    robot: Articulation = env.scene[asset_cfg.name]

    # if self‐collision wasn’t enabled, just return zeros
    if not hasattr(robot, "get_self_collisions"):
        return torch.zeros(env.num_envs, device=env.device)

    # that call now returns an (N,) int tensor
    coll_counts: torch.Tensor = robot.get_self_collisions()

    # only penalize when there are >= min_collisions simultaneous contacts
    mask = (coll_counts >= min_collisions).float()

    return penalty_per_collision * mask



def time_step_penalty(
    env,
    # no extra params needed
) -> torch.Tensor:
    # returns +1 for each env; the negative weight in RewardsCfg makes it a penalty
    return torch.ones(env.num_envs, device=env.device)



def cylinder_self_grasp_penalty(
    env,
    ee_cfg: SceneEntityCfg       = SceneEntityCfg("ee_frame"),
    cyl_cfg: SceneEntityCfg      = SceneEntityCfg("cylinder_frame"),
    grip_term: str               = "gripper_action2",
    reach_tol: float             = 0.02,    # meters
    penalty: float               = -1.0,
):
    # ----------------------------------------------------------------------------
    # 1) Gripper closed mask and fetch EE & cylinder center (all shape (N,3)/(N,))
    # ----------------------------------------------------------------------------
    closed = env.action_manager.get_term(grip_term).is_closed().float()      # (N,)
    ee_pos = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]            # (N,3)

    cyl_tf = env.scene[cyl_cfg.name]
    center = cyl_tf.data.target_pos_w[..., 0, :]                            # (N,3)

    # ---- DEBUG: print every env's EE pos & cylinder center ----
    # for i, (ee_i, cen_i, cl) in enumerate(zip(ee_pos.tolist(),
    #                                           center.tolist(),
    #                                           closed.tolist())):
        # print(f"[DEBUG] Cylinder Penalty – Env {i}: EE pos={ee_i}, "
        #       f"Center={cen_i}, Closed={bool(cl)}")
    # ----------------------------------------------------------------------------
    # 2) Static cylinder half-extents via USD template
    # ----------------------------------------------------------------------------
    stage     = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   includedPurposes={"default","render","proxy"},
                                   useExtentsHint=True)
    cyl_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "Cylinder" and str(prim.GetPath()).startswith("/World/envs/"):
            cyl_prim = prim
            break
    if cyl_prim is None or not cyl_prim.IsValid():
        raise RuntimeError("Couldn’t find any Cylinder prim under /World/envs/…")

    local_bound = bbox_cache.ComputeLocalBound(cyl_prim)
    rng         = local_bound.GetRange()
    min_pt, max_pt = rng.GetMin(), rng.GetMax()
    spans        = [max_pt[i] - min_pt[i] for i in range(3)]
    half_extents = torch.tensor([s * 0.5 for s in spans],
                                device=ee_pos.device)                         # (3,)
    cyl_radius      = torch.max(half_extents[0], half_extents[1])             # scalar
    cyl_half_height = half_extents[2]                                         # scalar

    # ----------------------------------------------------------------------------
    # 3) Distance‐to‐cylinder surface (vectorized)
    # ----------------------------------------------------------------------------
    delta       = ee_pos - center                                             # (N,3)
    dz          = torch.clamp(torch.abs(delta[..., 2]) - cyl_half_height, min=0.0)
    radial_dist = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2)
    dr          = torch.clamp(radial_dist - cyl_radius, min=0.0)
    dist_to_surf= torch.sqrt(dz**2 + dr**2)                                    # (N,)

    # ----------------------------------------------------------------------------
    # 4) Mask & penalty
    # ----------------------------------------------------------------------------
    too_close = (dist_to_surf < reach_tol).float()     # (N,)
    bad       = closed * too_close                     # (N,)
    return penalty * bad

import torch
import omni.usd
from pxr import Sdf
from grab_skywalker.mdp import SceneEntityCfg

import torch
import omni.usd
from pxr import UsdGeom, Sdf
from grab_skywalker.mdp import SceneEntityCfg


def wall_proximity_penalty(
    env,
    ee_cfg:   SceneEntityCfg = SceneEntityCfg("ee_frame"),
    obj_cfg:  SceneEntityCfg = SceneEntityCfg("object"),
    grip_term:str            = "gripper_action",
    cube_cfgs:list           = [
        SceneEntityCfg("cube1"),
        SceneEntityCfg("cube2"),
        SceneEntityCfg("cube3"),
    ],
    reach_tol:float          = 0.25,
    penalty:  float          = -1.0,
):
    # 1) Closed‐gripper mask & EE Y pos
    closed = env.action_manager.get_term(grip_term).is_closed().float()   # (N,)
    ee_y   = env.scene[ee_cfg.name].data.target_pos_w[..., 0, 1]         # (N,)

    # 2) World‐space center of your “object” (the wall assembly)
    obj_ent = env.scene[obj_cfg.name]
    if hasattr(obj_ent.data, "root_pos_w"):
        center_y = obj_ent.data.root_pos_w[..., 1]                       # (N,)
    else:
        center_y = obj_ent.data.target_pos_w[..., 0, 1]                  # (N,)

    # 3) Hard-coded half-thickness = 0.5 m / 2
    half_y = 0.5  

    # 4) Front face (–Y side) = center_y – half_y
    front_y = center_y + half_y                                          # (N,)

    # ── DEBUG ────────────────────────────────────────────────────────────
   
    print(f"[DEBUG][Env {0}] front_y={front_y[0].item():.3f}, "
              f"ee_y={ee_y[0].item():.3f}, closed={closed[0].item()}")
    # ─────────────────────────────────────────────────────────────────────

    # 5) Penetration into wall slab
    dy        = front_y - ee_y                                           # (N,)
    too_close = (dy > 0.0) & (dy < reach_tol)                             # (N,)

    # 6) Exempt any time you’re exactly over one of the mounting cubes
    ee_pos   = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]        # (N,3)
    near_any = torch.zeros_like(dy, dtype=torch.float32)                 # (N,)
    for cfg in cube_cfgs:
        ent = env.scene[cfg.name]
        # always use data.*:
        if hasattr(ent.data, "root_pos_w"):
            cube_pos = ent.data.root_pos_w                               # (N,3)
        else:
            cube_pos = ent.data.target_pos_w[..., 0, :]                  # (N,3)
        mask = (torch.abs(ee_pos[...,1] - cube_pos[...,1]) < reach_tol).float()
        near_any = torch.maximum(near_any, mask)

    # 7) Final mask: closed & too_close & not near_any
    bad = closed * too_close.float() * (1.0 - near_any)                   # (N,)

    return penalty * bad

    # # 2) EE position in world:
    # ee_frame = env.scene[ee_cfg.name]
    # ee_pos   = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)

    # # 3) Wall’s center in world:
    # # wall_prim = env.scene[wall_cfg.name]
    # # try:
    # #     center = wall_prim.data.root_pos_w           # (N, 3)
    # # except AttributeError:
    # #     center, _ = wall_prim.get_world_poses()      # (N, 3)



    # if hasattr(wall_cfg, "prim"):
    #     usd_prim = wall_cfg.prim
    # else:
    #     # Articulations carry their path under `prim_path`
    #     usd_prim = get_prim_at_path(wall_cfg.prim_path)

    # # now you can read the extent attribute:
    # extents = usd_prim.GetAttribute("extent").Get()   # [minX,minY,minZ, maxX,maxY,maxZ]
    # mins, maxs = torch.tensor(extents[:3], device=ee_pos.device), \
    #             torch.tensor(extents[3:], device=ee_pos.device)
    # half_extents = (maxs - mins) / 2

    # # 4) Half-extents of your cube (in meters).
    # #    You can either hard-code these if your wall is, say, 1×1×2 m:
    # #half_extents = torch.tensor([0.5, 0.5, 1.0], device=ee_pos.device)
    # #
    # #    Or, if you want to pull them at runtime from USD:
    # extents = wall_prim.prim.GetAttribute("extent").Get()  # [minX, minY, minZ, maxX, maxY, maxZ]
    # mins, maxs = torch.tensor(extents[:3], device=ee_pos.device), torch.tensor(extents[3:], device=ee_pos.device)
    # half_extents = (maxs - mins) / 2

    # # 5) Compute “outside‐the‐box” distances per axis:
    # delta = torch.abs(ee_pos - center) - half_extents    # (N,3)
    # delta_clamped = torch.clamp(delta, min=0.0)          # zero any negative (inside)
    # dist_to_surface = torch.norm(delta_clamped, dim=1)  # (N,)

    # # 6) Mask of “too close”
    # too_close = (dist_to_surface < reach_tol).float()    # (N,)

    # # 7) Combine masks (only penalize when closed *and* too close):
    # bad = closed * too_close

    # return penalty * bad


# def cylinder_self_grasp_penalty(
#     env,
#     ee_cfg: SceneEntityCfg       = SceneEntityCfg("ee_frame"),
#     cyl_cfg: SceneEntityCfg      = SceneEntityCfg("cylinder_frame"),
#     grip_term: str               = "gripper_action2",
#     reach_tol: float             = 0.02,    # meters
#     penalty: float               = -1.0,
# ):
#     # ----------------------------------------------------------------------------
#     # 1) Build masks and fetch EE & cylinder centers (all shape (N,3) or (N,))
#     # ----------------------------------------------------------------------------
#     term   = env.action_manager.get_term(grip_term)
#     closed = term.is_closed().float()                       # (N,)
#     ee_pos = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]  # (N,3)

#     cyl_tf = env.scene[cyl_cfg.name]
#     center = cyl_tf.data.target_pos_w[..., 0, :]           # (N,3)
#     # ----------------------------------------------------------------------------
#     # 2) On first call (or each call—cheap), scan USD for a Cylinder prim and get half-extents
#     # ----------------------------------------------------------------------------
#     stage     = omni.usd.get_context().get_stage()
#     bbox_cache = UsdGeom.BBoxCache(
#         Usd.TimeCode.Default(),
#         includedPurposes={"default", "render", "proxy"},
#         useExtentsHint=True
#     )
#     cyl_prim = None
#     for prim in stage.Traverse():
#         # find the first prim literally named "Cylinder" under any env
#         if prim.GetName() == "Cylinder" and str(prim.GetPath()).startswith("/World/envs/"):
#             cyl_prim = prim
#             break
#     if cyl_prim is None or not cyl_prim.IsValid():
#         raise RuntimeError("Couldn’t find any Cylinder prim under /World/envs/…")

#     # Compute its local (template) bounding-box
#     local_bound = bbox_cache.ComputeLocalBound(cyl_prim)   # GfBBox3d
#     rng         = local_bound.GetRange()                   # GfRange3d
#     min_pt, max_pt = rng.GetMin(), rng.GetMax()            # GfVec3d

#     # Turn into half-extents: X-span/2, Y-span/2, Z-span/2
#     spans        = [max_pt[i] - min_pt[i] for i in range(3)]
#     half_extents = torch.tensor([s * 0.5 for s in spans], device=ee_pos.device)  # (3,)
#     cyl_radius      = torch.max(half_extents[0], half_extents[1])  # scalar
#     cyl_half_height = half_extents[2]                             # scalar

#     # ----------------------------------------------------------------------------
#     # 3) Compute shortest distance from EE to *capped* cylinder surface (vectorized)
#     # ----------------------------------------------------------------------------
#     delta       = ee_pos - center                             # (N,3)
#     dz          = torch.clamp(torch.abs(delta[..., 2]) - cyl_half_height, min=0.0)  # (N,)
#     radial_dist = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2)                  # (N,)
#     dr          = torch.clamp(radial_dist - cyl_radius, min=0.0)                   # (N,)
#     dist_to_surf= torch.sqrt(dz**2 + dr**2)                                         # (N,)

#     # ----------------------------------------------------------------------------
#     # 4) Mask by reach_tol & closed, return vector of penalties
#     # ----------------------------------------------------------------------------
#     too_close = (dist_to_surf < reach_tol).float()         # (N,)
#     bad       = closed * too_close                         # (N,)
#     return penalty * bad