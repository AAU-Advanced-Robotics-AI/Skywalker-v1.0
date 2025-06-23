# grab_skywalker/mdp/flags.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import grab_skywalker.mdp as mdp   # your reward helpers live here

# ---------- helper --------------------------------------------------
def _ensure_flags(env: ManagerBasedRLEnv):
    """Allocate flag tensors once and keep them on the env."""
    if not hasattr(env, "_gs_flags"):
        device = env.device
        env._gs_flags = {
            "c1_grasp":  torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "docked1":   torch.zeros_like(torch.zeros(env.num_envs, dtype=torch.bool, device=device)),
            "c2_grasp":  torch.zeros_like(torch.zeros(env.num_envs, dtype=torch.bool, device=device)),
            "docked2":   torch.zeros_like(torch.zeros(env.num_envs, dtype=torch.bool, device=device)),
        }
    return env._gs_flags
# --------------------------------------------------------------------

def update_flags(env: ManagerBasedRLEnv,
                 tol_cube: float = 0.12,
                 tol_dock: float = 0.15,
                 grip1: str = "gripper_action",
                 grip2: str = "gripper_action2"):
    """
    Runs every step.  Sets / latches four booleans per sub-env:

        c1_grasp   – cube 1 has been grasped at least once
        docked1    – robot base has closed clamp near dock_marker
        c2_grasp   – cube 2 has been grasped (after dock-1)
        docked2    – clamp closed near goal_marker   (→ episode done)

    Once a flag becomes True for an env it stays True until reset_idx().
    """
    f = _ensure_flags(env)          # allocate if missing
    g1 = env.action_manager.get_term(grip1)      # EE finger
    g2 = env.action_manager.get_term(grip2)      # cylinder clamp

    # == cube positions ============
    cube1 = env.scene["cube1"].data.target_pos_w[..., 0, :]
    cube2 = env.scene["cube2"].data.target_pos_w[..., 0, :]
    ee    = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    base  = env.scene["robot"].data.root_pos_w

    dock  = env.scene["dock_marker"].data.root_pos_w
    goal  = env.scene["goal_marker"].data.root_pos_w

    # == flags ==============================================================
    # 1) cube-1 grasp
    f["c1_grasp"] |= g1.is_closed() & (torch.norm(ee - cube1, dim=1) < tol_cube)

    # 2) dock-1 after cube-1 grasp
    mask1 = f["c1_grasp"] & ~f["docked1"]
    f["docked1"] |= mask1 & g2.is_closed() & (torch.norm(base - dock, dim=1) < tol_dock)

    # 3) cube-2 grasp only once we are docked-1
    mask2 = f["docked1"] & ~f["c2_grasp"]
    f["c2_grasp"] |= mask2 & g1.is_closed() & (torch.norm(ee - cube2, dim=1) < tol_cube)

    # 4) final docking near goal
    mask3 = f["c2_grasp"] & ~f["docked2"]
    f["docked2"] |= mask3 & g2.is_closed() & (torch.norm(base - goal, dim=1) < tol_dock)

    return torch.zeros(env.num_envs, device=env.device)   # event terms must return tensor
