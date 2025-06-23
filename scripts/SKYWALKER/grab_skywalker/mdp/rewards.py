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
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


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



def forward_progress(env, w: float = 1.0):
    """Pay +w for each metre of forward ΔX (positive only)."""
    return w * mdp.progress_delta(env).squeeze(1)          # (N,)

# tunables – all ≤ 1 so positives never blow up
_EVT_BONUS  = 0.15         # paid once per phase-transition
_K_CUBE     = 0.8          # dense peak for EE→cube
_K_DOCK     = 0.8          # dense peak for base→dock/goal
_S_CUBE     = 0.35         # m  (≈ zero-cross for EE→cube)
_S_DOCK     = 0.50         # m  (≈ zero-cross for base→dock/goal)
_TOL_GRASP  = 0.20         # unchanged
_TOL_DOCK   = 0.10        # unchanged
# ──────────────────────────────────────────────────────────────


# --------------------------------------------------------------------
# Positive-only dense shaping
# --------------------------------------------------------------------
# ----------------------------------------------------------------------
#  strictly-positive bounded shaping
# ----------------------------------------------------------------------
def _bounded_positive(dist: torch.Tensor, k: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    r = k / (1 + dist / s)          ∈ (0, k]
        • r → k   as   dist → 0
        • r = k/2 when dist = s
        • r ↓ 0   asymptotically as distance grows
    """
    return k / (1.0 + dist / s)
@torch.no_grad()
def skywalker_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    def _bounded_positive(dist: torch.Tensor, k: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return k / (1.0 + dist / s)

    def drag_shaping_if_grasped(base, target, is_grasped, active_mask, k_val=0.4, s_val=0.5):
        dist = torch.norm(base - target, dim=1)
        shaping = _bounded_positive(
            dist,
            torch.full_like(dist, k_val),
            torch.full_like(dist, s_val)
        )
        return active_mask.float() * is_grasped.float() * shaping

    dev, N = env.device, env.num_envs
    idx = torch.arange(N, device=dev)

    # internal buffers
    if not hasattr(env, "_phase"):
        env._phase = torch.zeros(N, dtype=torch.int8, device=dev)
        env._flag = torch.zeros(N, dtype=torch.bool, device=dev)

    # reset tracking buffers if needed
    if hasattr(env, "reset_buf"):
        done = env.reset_buf.bool()
        env._phase[done] = 0
        env._flag[done] = False

    phase, flag = env._phase.long(), env._flag

    # --- state
    ee   = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    base = env.scene["robot"].data.root_pos_w
    cubes = torch.stack([
        env.scene["cube1"].data.target_pos_w[..., 0, :],
        env.scene["cube2"].data.target_pos_w[..., 0, :],
    ])
    dock = env.scene["dock_marker"].data.root_pos_w
    goal = env.scene["goal_marker"].data.root_pos_w

    # --- predicates
    grasped = lambda k: is_grasping_fixed_object(
        env,
        ee_cfg=SceneEntityCfg("ee_frame"),
        cube_cfg=SceneEntityCfg(f"cube{k+1}"),
        tol=_TOL_GRASP,
        grip_term="gripper_action"
    ).bool()

    clamp_closed = env.action_manager.get_term("gripper_action2").is_closed().bool()
    at_dock = torch.norm(base - dock, dim=1) < _TOL_DOCK
    at_goal = torch.norm(base - goal, dim=1) < _TOL_DOCK

    # --- transitions
    evt = torch.zeros_like(flag)

    m = (phase == 0) & grasped(0)
    env._phase[m] = 1; evt |= m

    m = (phase == 1) & grasped(0) & at_dock
    env._phase[m] = 2; evt |= m

    m = (phase == 2) & at_dock & (~grasped(0))
    env._phase[m] = 3; evt |= m

    m = (phase == 3) & grasped(1) & at_dock
    env._phase[m] = 4; evt |= m

    m = (phase == 4) & grasped(1) & (~at_dock)
    env._phase[m] = 5; evt |= m

    m = (phase == 5) & at_goal & grasped(1)
    env._phase[m] = 6; evt |= m

    new_evt = evt & ~flag
    flag[:] = evt

    # --- bonus per phase
    PHASE_BONUS = torch.tensor([0.05, 0.15, 0.10, 0.0, 0.25, 0.10, 0.30], device=dev)
    bonus = PHASE_BONUS[phase] * new_evt.float()

    # --- shaping targets
    src = torch.where((phase % 2 == 0).unsqueeze(1), ee, base)
    tgt = torch.zeros_like(src)

    tgt[phase == 0] = cubes[0, idx][phase == 0]     # EE → cube1
    tgt[phase == 1] = dock[phase == 1]              # base → dock (dragging cube1)
    tgt[phase == 2] = dock[phase == 2]              # base → dock (releasing cube1)
    tgt[phase == 3] = cubes[1, idx][phase == 3]     # EE → cube2
    tgt[phase == 4] = goal[phase == 4]              # base → goal (dragging cube2)
    tgt[phase == 5] = goal[phase == 5]              # base → goal (approach)
    tgt[phase == 6] = goal[phase == 6]              # final phase (clamped)


    dist = torch.norm(src - tgt, dim=1)

    PHASE_K = torch.tensor([0.3, 0.3, 0.5, 0.0, 0.6, 0.6, 0.8], device=dev)
    PHASE_S = torch.tensor([0.25, 0.25, 0.40, 0.0, 0.45, 0.45, 0.50], device=dev)

    k, s = PHASE_K[phase], PHASE_S[phase]
    dense = _bounded_positive(dist, k, s)

    # extra shaping
    dense += drag_shaping_if_grasped(base, dock, grasped(0), phase == 1, k_val=0.55)
    dense += drag_shaping_if_grasped(base, goal, grasped(1), phase == 4, k_val=0.55)

    # exploration incentive (try to grasp)
    grip_closed = env.action_manager.get_term("gripper_action").is_closed().float()
    near_cube = torch.norm(ee - cubes[0, idx], dim=1) < 0.2
    dense += (phase <= 1).float() * 0.05 * grip_closed * near_cube.float()

    # debug
    g0 = is_grasping_fixed_object(env, cube_cfg=SceneEntityCfg("cube1"), tol=_TOL_GRASP)
    g1 = is_grasping_fixed_object(env, cube_cfg=SceneEntityCfg("cube2"), tol=_TOL_GRASP)
    print(
        f"[debug] phase[0] = {phase[0]}, grasped(0)[0] = {g0[0].item():.2f}, "
        f"grasped(1)[0] = {g1[0].item():.2f}, at_dock[0] = {at_dock[0].item()}"
    )

    return bonus + dense

    #return torch.clamp(bonus + dense, 0.0, 1.0)





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

        # ── DEBUG: cube’s full 3D pose ───────────────────────────────────────
    #print(f"[DEBUG] is_grasping '{cube_cfg.name}' → cube_pos={cube_pos[0].tolist()}, ee_pos={ee_pos[0].tolist()}")
    # ─────────────────────────────────────────────────────────────────────

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
    grip_term: str               = "gripper_action",
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs     import ManagerBasedRLEnv

# ────────────────────────────────────────────────────────────────
#   Wall danger band  (1-D gap check along +Y direction only)
# ────────────────────────────────────────────────────────────────
def wall_proximity_penalty(
    env:          ManagerBasedRLEnv,
    ee_cfg:       SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wall_cfg:     SceneEntityCfg = SceneEntityCfg("object"),
    grip_term:    str            = "gripper_action",
    cube_cfgs:    list           = [SceneEntityCfg("cube1"),
                                    SceneEntityCfg("cube2")],
    reach_tol:    float          = 0.2,
    penalty:      float          = -5.0,
):
    """
    • Treat the wall as an infinite vertical plane **along the X-axis**.
    • Gap `dx = wall_front_x − ee_x`  (positive when the EE is in front of the wall).
      ⇒ `dx < reach_tol`  → danger.
    • Fires only if main gripper is CLOSED *and* the EE is not hovering over a mounting cube.
    """

    closed = env.action_manager.get_term(grip_term).is_closed().bool()        # (N,)

    # ── EE X-coordinate ───────────────────────────────────────────────
    ee_x = env.scene[ee_cfg.name].data.target_pos_w[..., 0, 0]                # (N,)

    # ── Wall front-face X (center + half-thickness) ───────────────────
    wall_ent = env.scene[wall_cfg.name]
    centre_x = (wall_ent.data.root_pos_w[..., 0]
                if hasattr(wall_ent.data, "root_pos_w")
                else wall_ent.data.target_pos_w[..., 0, 0])
    wall_front_x = centre_x                                             # push out 0.2m

    # ── Compute unsigned distance from wall front ─────────────────────
    gap = torch.abs(wall_front_x - ee_x)             # (N,)
    too_close = gap < reach_tol                      # (N,)

    # ── Exempt if EE is aligned with any cube in X ────────────────────
    ee_pos  = env.scene[ee_cfg.name].data.target_pos_w[..., 0, :]             # (N,3)
    exempt  = torch.zeros_like(too_close)                                     # bool (N,)
    for cfg in cube_cfgs:
        cube_ent = env.scene[cfg.name]
        cube_pos = (cube_ent.data.root_pos_w
                    if hasattr(cube_ent.data, "root_pos_w")
                    else cube_ent.data.target_pos_w[..., 0, :])
        exempt |= torch.abs(ee_pos[..., 0] - cube_pos[..., 0]) < reach_tol

    # ── Apply penalty only when: closed ∧ too_close ∧ ¬exempt ────────
    bad = closed & too_close #& (~exempt)

    # Optional debug print
   # print(f"[wall_penalty] wall_x={wall_front_x[0]:+.3f}, ee_x={ee_x[0]:+.3f}, dx={gap[0]:.3f}, closed={closed[0].item()}, too_close={too_close[0].item()}, exempt={exempt[0].item()}, bad={bad[0].item()}")

    return penalty * bad.float()
