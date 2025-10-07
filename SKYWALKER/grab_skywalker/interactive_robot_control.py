#!/usr/bin/env python3
"""
Interactive Point-to-Point Robot Control (BASE FRAME observations)
with robust quaternion/frames debugging and "Real Robot" guidance.

Key fixes & features:
- EE orientation in BASE: q_B_E = q_B_W^{-1} ⊗ q_W_E (correct order)
- EE position in BASE:   p_B_E = R(q_B_W^{-1}) * (p_W_E - p_W_B)
- Debug: WORLD & BASE angle errors, canonicalized quat prints
- Debug: consistency checks proving the algebra
- OffsetCfg.rot ordering toggle (WXYZ vs XYZW) to rule out hidden 180° flips
- Real robot notes: exactly what to send and how to compose frames

Quaternion convention everywhere here is [w, x, y, z] (Hamilton).
"""

import argparse
import os
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation  # only to build target quat from RPY

# Isaac Lab / Sim
from isaaclab.app import AppLauncher

# --------------------------- CLI ---------------------------
parser = argparse.ArgumentParser(description="Interactive robot control (policy handles normalization)")
parser.add_argument(
    "--model_path",
    type=str,
    default="scripts/SKYWALKER/grab_skywalker/exported_models/skywalker_ptp_baseframe_policy.pt",
    help="Path to exported TorchScript model",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --------------------------- Launch Isaac Sim ---------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# ====================== CONFIG TO VERIFY OFFSET QUAT ORDER ======================
# Some APIs want rot in WXYZ, others in XYZW. Toggle to test both quickly.
OFFSET_ROT_IS_WXYZ = False  # set True to try WXYZ; False uses XYZW

# Helper to build identity in whichever ordering:
def offset_rot_identity():
    if OFFSET_ROT_IS_WXYZ:
        # [w,x,y,z] identity
        return [1.0, 0.0, 0.0, 0.0]
    else:
        # [x,y,z,w] identity
        return [0.0, 0.0, 0.0, 1.0]

# --------------------------- Quaternion helpers (wxyz) ---------------------------
def quat_normalize_np(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def quat_shortest_angle_deg(q1_wxyz, q2_wxyz):
    q1 = quat_normalize_np(q1_wxyz)
    q2 = quat_normalize_np(q2_wxyz)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    ang = 2.0 * math.degrees(math.acos(dot))
    return ang, dot

def quat_mul_wxyz_torch(q1, q2):
    """Hamilton product in [w,x,y,z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ), dim=-1)

def rotate_vec_by_quat_wxyz_torch(q, v):
    """
    Rotate vector v by quaternion q (both torch tensors).
    q: (...,4) [w,x,y,z]; v: (...,3). Returns (...,3).
    v' = v + 2*q_vec × (q_vec × v + q_w * v)
    """
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)

def quat_to_rotmat_np_wxyz(q):
    """[w,x,y,z] -> 3x3 rotation matrix (numpy) for quick axis checks."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def quat_canonicalize_torch(q, ref):
    """Flip q so dot(q, ref) >= 0 for stable logging."""
    dot = torch.sum(q * ref, dim=-1, keepdim=True)
    sign = torch.sign(dot)
    sign[sign == 0] = 1.0
    return q * sign

def quat_angle_deg_torch(q1, q2):
    """Shortest angle between two [w,x,y,z] quats (torch)."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(-1.0, 1.0)
    return (2.0 * torch.acos(dot)) * (180.0 / math.pi)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Interactive scene - robot + ground + movable target."""

    num_envs: int = 1
    env_spacing: float = 2.0

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/bruno/IsaacLab/scripts/SKYWALKER/Collected_skywalker_robot/skywalker_robot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=True,  # fixed base
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # NOTE: We place the FrameTransformer at link_base and track link_eef with an offset.
    # The 'rot' ordering can differ across builds. We expose a toggle above to test both.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_base",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.copy().replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/xarm7/link_eef",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.06], rot=offset_rot_identity()),
            ),
        ],
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

# --------------------------- xArm7 limits (from your USD) ---------------------------
def get_xarm7_limits(device: torch.device):
    limits_low  = torch.tensor(
        [-6.283185307179586,
         -2.0589999996956756,
         -6.283185307179586,
         -0.19197993453406906,
         -6.283185307179586,
         -1.6929698980332752,
         -6.283185307179586],
        device=device, dtype=torch.float32
    )
    limits_high = torch.tensor(
        [ 6.283185307179586,
          2.0943998147821756,
          6.283185307179586,
          3.9269998926993517,
          6.283185307179586,
          3.141592653589793,
          6.283185307179586],
        device=device, dtype=torch.float32
    )
    return limits_low, limits_high

# --------------------------- Simple command controller ---------------------------
class InteractiveController:
    """Handles user input for new target positions via a text file."""

    def __init__(self):
        # BASE frame targets (same as training)
        self.target_position = [0.3, 0.1, 0.5]           # BASE
        # If you want +90° about BASE Y, use [0.7071, 0, 0.7071, 0]
        self.target_quaternion = [0.0, 0.707, 0.0, 0.707]  # [w,x,y,z] BASE
        self.use_quaternion = True

        self.new_target = False
        self.running = True
        self.command_file = "/tmp/robot_commands.txt"
        self.last_command_time = 0
        self.create_command_file()

    def create_command_file(self):
        with open(self.command_file, "w") as f:
            f.write("# Commands:\n")
            f.write("#   goto X Y Z           (e.g., goto 0.3 0.2 0.85)\n")
            f.write("#   quat W X Y Z         (e.g., quat 0.707 0 0.707 0)  # +90deg about Y\n")
            f.write("#   quit\n\n")
            f.write("# Targets are in BASE frame. Quaternion format: [w, x, y, z] (normalized).\n")
        self.last_command_time = os.path.getmtime(self.command_file)

    def check_for_commands(self):
        try:
            t = os.path.getmtime(self.command_file)
            if t > self.last_command_time:
                self.last_command_time = t
                with open(self.command_file, "r") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip().lower()
                    if not line or line.startswith("#"):
                        continue
                    self.process_command(line)
                    break
        except FileNotFoundError:
            self.create_command_file()
        except Exception as e:
            print(f"Error reading command file: {e}")

    def process_command(self, command: str):
        if command == "quit":
            self.running = False
            print("🛑 Quit command received!")
            return
        if command.startswith("goto"):
            parts = command.split()
            if len(parts) == 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.target_position = [x, y, z]
                    self.use_quaternion = True  # keep orientation unless user changes it
                    self.new_target = True
                    print(f"🎯 New target position (BASE): [{x:.3f}, {y:.3f}, {z:.3f}]")
                except ValueError:
                    print("❌ Invalid coordinates")
            else:
                print("❌ Usage: goto X Y Z")
        elif command.startswith("quat"):
            parts = command.split()
            if len(parts) == 5:
                try:
                    w, x, y, z = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Normalize quaternion
                    norm = math.sqrt(w*w + x*x + y*y + z*z)
                    if norm > 0:
                        self.target_quaternion = [w/norm, x/norm, y/norm, z/norm]
                        self.use_quaternion = True
                        self.new_target = True
                        print(f"🎯 New target quaternion (BASE, wxyz): [{w/norm:.3f}, {x/norm:.3f}, {y/norm:.3f}, {z/norm:.3f}]")
                    else:
                        print("❌ Invalid quaternion (zero magnitude)")
                except ValueError:
                    print("❌ Invalid quaternion values")
            else:
                print("❌ Usage: quat W X Y Z")
        else:
            print(f"❌ Unknown command: {command}")

# --------------------------- Base pose helper ---------------------------
def get_base_pose_world(robot, scene, base_idx, device):
    """
    Returns (base_pos_w [1,3], base_quat_wxyz [1,4]) as torch tensors.
    Tries several layouts; falls back to env origin + identity orientation.
    """
    if hasattr(robot.data, "link_pos_w") and hasattr(robot.data, "link_quat_w"):
        try:
            pos = robot.data.link_pos_w[:, base_idx, :3]
            quat = robot.data.link_quat_w[:, base_idx, :]
            return pos, quat
        except Exception:
            pass

    has_root_pos = hasattr(robot.data, "root_pos_w")
    has_root_quat = hasattr(robot.data, "root_quat_w")
    if has_root_pos and has_root_quat:
        try:
            pos = robot.data.root_pos_w[:, :3]
            quat = robot.data.root_quat_w[:, :4]
            return pos, quat
        except Exception:
            pass

    pos = scene.env_origins.to(device=device, dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # [w,x,y,z]
    return pos, quat

# --------------------------- Main ---------------------------
def main():
    print("🤖 INTERACTIVE Isaac Sim Robot Control (policy-embedded normalization)")
    print(f"📁 Model: {args_cli.model_path}")
    print("=" * 60)

    # ---- Load TorchScript model (contains obs normalization) ----
    try:
        model = torch.jit.load(args_cli.model_path, map_location="cpu")
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ---- Setup simulation ----
    print("🌍 Setting up Isaac Sim...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=2)
    sim = sim_utils.SimulationContext(sim_cfg)

    print("🤖 Spawning robot...")
    scene_cfg = RobotSceneCfg()
    scene = InteractiveScene(scene_cfg)

    print("🔄 Initializing...")
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    ee_frame = scene["ee_frame"]

    # Try to locate base body index
    try:
        base_indices = robot.find_bodies("xarm7/link_base")
        BASE_IDX = int(base_indices[0])
    except Exception:
        BASE_IDX = 0
    print(f"Using base body index for world composition: {BASE_IDX}")
    print(f"DEBUG - OFFSET_ROT_IS_WXYZ={OFFSET_ROT_IS_WXYZ} (flip this if BASE quats are 180° off around Z)")

    # Neutral starting pose (inside limits, not near boundaries)
    try:
        neutral = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                               device=robot.device, dtype=torch.float32)
        robot.write_joint_state(position=neutral)  # may not exist in some builds
        print("✅ Set neutral joint pose.")
    except Exception as e:
        print(f"⚠️ Could not set neutral pose: {e}")

    # Optional: print USD soft limits
    try:
        usd_limits = robot.data.soft_joint_pos_limits[0, :7].detach().cpu().numpy()
        print("USD soft limits (low, high) per joint:\n", usd_limits)
    except Exception:
        pass

    # Markers (target frame in WORLD)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_frame_markers = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/TargetFrameMarker")
    )

    # Controller + runtime buffers
    controller = InteractiveController()
    last_action_obs = torch.zeros((1, 7), device=robot.device, dtype=torch.float32)

    # ---- Helper to build obs EXACTLY like in training ----
    def build_obs_and_aux(step_count: int):
        """
        Return:
          obs[1,35]                - features like training (BASE frame observations)
          ee_pos_world[1,3]        - EE position (WORLD) for viz/metrics
          target_pos_world[1,3]    - target position (WORLD) for viz/metrics
          tgt_quat_wxyz_base[1,4]  - target quaternion (BASE), for obs
          tgt_quat_wxyz_world[1,4] - target quaternion (WORLD), for viz/metrics
        And visualize WORLD target.
        """
        # Joint state (relative to defaults)
        joint_pos = robot.data.joint_pos[:, :7] - robot.data.default_joint_pos[:, :7]
        joint_vel = robot.data.joint_vel[:, :7] - robot.data.default_joint_vel[:, :7]

        # Base pose in WORLD
        base_pos_w, base_quat_wxyz = get_base_pose_world(robot, scene, BASE_IDX, robot.device)

        # EE pose from sim (WORLD)
        ee_pos_world = ee_frame.data.target_pos_w[:, 0, :3]       # [1,3] WORLD
        ee_quat_wxyz_world = ee_frame.data.target_quat_w[:, 0, :] # [1,4] WORLD

        # Inverse base orientation
        base_quat_inv = torch.cat([base_quat_wxyz[:, :1], -base_quat_wxyz[:, 1:]], dim=1)  # [1,4]

        # --- EE pose in BASE frame (FIXED) ---
        ee_pos_base = rotate_vec_by_quat_wxyz_torch(base_quat_inv, ee_pos_world - base_pos_w)  # [1,3]
        ee_quat_base = quat_mul_wxyz_torch(base_quat_inv, ee_quat_wxyz_world)  # [1,4]

        # --- BASE-frame target pose command (policy expects this)
        target_pos_base = torch.tensor([controller.target_position], device=robot.device, dtype=torch.float32)  # [1,3]
        if controller.use_quaternion and controller.target_quaternion is not None:
            tgt_quat_wxyz_base = torch.tensor([controller.target_quaternion], device=robot.device, dtype=torch.float32)  # [1,4]
        else:
            # Convert RPY (XYZ intrinsic) to quaternion (BASE)
            r = Rotation.from_euler("xyz", controller.target_orientation, degrees=False)
            tq_xyzw_np = np.asarray(r.as_quat(), dtype=np.float32)     # [x,y,z,w]
            tq_xyzw = torch.from_numpy(tq_xyzw_np).unsqueeze(0).to(robot.device)
            tgt_quat_wxyz_base = torch.cat([tq_xyzw[:, 3:4], tq_xyzw[:, :3]], dim=1)  # [w,x,y,z]

        target_pose_command = torch.cat([target_pos_base, tgt_quat_wxyz_base], dim=1)  # [1,7] BASE

        # --- Compose BASE target to WORLD for viz & error (this was already correct) ---
        tgt_quat_wxyz_world = quat_mul_wxyz_torch(base_quat_wxyz, tgt_quat_wxyz_base)  # [1,4] WORLD
        target_pos_world = base_pos_w + rotate_vec_by_quat_wxyz_torch(base_quat_wxyz, target_pos_base)  # [1,3]

        # Visualize target in WORLD (markers expect xyzw)
        tgt_xyzw_world = torch.cat([tgt_quat_wxyz_world[:, 1:], tgt_quat_wxyz_world[:, :1]], dim=1)  # [1,4] xyzw
        target_frame_markers.visualize(
            translations=target_pos_world.repeat(robot.num_instances, 1),
            orientations=tgt_xyzw_world.repeat(robot.num_instances, 1)
        )

        # ------------- DEBUG: robust orientation checks -------------
        if step_count % 5 == 0:
            # WORLD angle error (matches visualization)
            world_ang = quat_angle_deg_torch(ee_quat_wxyz_world, tgt_quat_wxyz_world)[0].item()
            # BASE angle error (what the policy "sees")
            base_ang = quat_angle_deg_torch(ee_quat_base, tgt_quat_wxyz_base)[0].item()
            # Canonicalize for clean logging (avoid q vs -q confusion)
            ee_base_print = quat_canonicalize_torch(ee_quat_base, tgt_quat_wxyz_base)[0].detach().cpu().numpy()
            tgt_base_print = tgt_quat_wxyz_base[0].detach().cpu().numpy()

            # Algebra consistency: ee_base ?= base^{-1} ⊗ tgt_world
            ee_base_from_world_tgt = quat_mul_wxyz_torch(base_quat_inv, tgt_quat_wxyz_world)
            cmp_ang = quat_angle_deg_torch(ee_quat_base, ee_base_from_world_tgt)[0].item()

            # Base quaternion in WORLD, to understand non-identity base
            if step_count % 30 == 0:
                print("DEBUG - base_quat_wxyz (WORLD):", base_quat_wxyz[0].detach().cpu().numpy())

            print(f"DEBUG - ORIENT err WORLD: {world_ang:.2f} deg | BASE: {base_ang:.2f} deg | Consistency: {cmp_ang:.2f} deg")
            print(f"DEBUG - EE quat BASE (canon): {np.round(ee_base_print, 4)}")
            print(f"DEBUG - TGT quat BASE:        {np.round(tgt_base_print, 4)}")

        # ------------------------------------------------------------

        # Final 35-D observation (BASE frame observations like training)
        obs = torch.cat(
            [joint_pos, joint_vel, ee_pos_base, ee_quat_base, target_pose_command, last_action_obs],
            dim=1,
        ).to(device=robot.device, dtype=torch.float32)

        return obs, ee_pos_world, target_pos_world, tgt_quat_wxyz_base, tgt_quat_wxyz_world

    # Limits (USD)
    xarm_low, xarm_high = get_xarm7_limits(robot.device)

    print("🎮 ROBOT READY! Edit /tmp/robot_commands.txt (e.g., 'goto 0.5 0.25 0.75' or 'quat 0.707 0 0.707 0').")
    step_count = 0

    try:
        while simulation_app.is_running() and controller.running:
            controller.check_for_commands()
            if controller.new_target:
                controller.new_target = False
                if controller.use_quaternion:
                    print(f"🎯 Target set (BASE): pos={controller.target_position} | quat(wxyz)={controller.target_quaternion}")
                else:
                    print(f"🎯 Target set (BASE): pos={controller.target_position} | RPY={controller.target_orientation}")

            # ---- Build obs
            obs, ee_pos_world, target_pos_world, tgt_quat_wxyz_base, tgt_quat_wxyz_world = build_obs_and_aux(step_count)

            #-------------DEBUG: print obs slice every step---------------
            if step_count % 1 == 0:
                observation = obs[0].cpu().numpy()
                print(f"-----------OBS (BASE FRAME)--------------")
                print(f"DEBUG - Joint pos (0:7):         {np.round(observation[0:7], 3)}")
                print(f"DEBUG - Joint vel (7:14):        {np.round(observation[7:14], 3)}")
                print(f"DEBUG - EE pos BASE (14:17):     {np.round(observation[14:17], 3)}")
                print(f"DEBUG - EE quat BASE (17:21):    {np.round(observation[17:21], 4)}")
                print(f"DEBUG - Target pose BASE (21:28): {np.round(observation[21:28], 3)}")
                print(f"DEBUG - Last action (28:35):     {np.round(observation[28:35], 3)}")
                print(f"-----------END OF OBS--------------")
            #-------------------------------------------------------------

            # ---- Inference (model normalizes internally)
            with torch.no_grad():
                policy_out = model(obs)  # raw means (unbounded during training)
            raw_action = policy_out[:, :7]  # what training logged as "last_action"

            # ---- ABSOLUTE TARGET like training: q_target = default + 0.5 * action
            q_default = robot.data.default_joint_pos[:, :7]   # (1,7)
            target_joint_pos = q_default + 0.5 * raw_action

            # ---- Safety: clamp to USD soft limits
            target_joint_pos = torch.max(torch.min(target_joint_pos, xarm_high), xarm_low)

            # ---- Apply command, then step sim
            robot.set_joint_position_target(target_joint_pos)
            scene.write_data_to_sim()
            sim_utils.SimulationContext.instance().step(render=True)
            scene.update(0.01)
            robot.update(sim.get_physics_dt())

            # ---- Feed back RAW action (matches training "last_action")
            last_action_obs = raw_action.detach()

            # ---- Debug (every 5 steps): WORLD orientation error + distances
            if step_count % 5 == 0:
                jp = robot.data.joint_pos[0, :7].detach().cpu().numpy()
                tgt = target_joint_pos[0].detach().cpu().numpy()
                q_def = q_default[0].detach().cpu().numpy()
                raw_act = raw_action[0].detach().cpu().numpy()
                dist = float(torch.norm(ee_pos_world - target_pos_world))
                ra_min = raw_action.min().item()
                ra_max = raw_action.max().item()

                ee_quat_wxyz_np = ee_frame.data.target_quat_w[0, 0, :].detach().cpu().numpy()
                tgt_quat_wxyz_np = tgt_quat_wxyz_world[0].detach().cpu().numpy()
                ang_err_deg, qdot = quat_shortest_angle_deg(ee_quat_wxyz_np, tgt_quat_wxyz_np)

                print(f"🔍 raw_action range: [{ra_min:.3f}, {ra_max:.3f}]")
                print(f"🎯 Target distance: {dist: .3f} m")
                print(f"DEBUG - default pos {np.round(q_def, 3)}")
                print(f"DEBUG - Current joint pos: {np.round(jp, 3)}")
                print(f"DEBUG - raw action: {np.round(raw_act, 3)}")
                print(f"DEBUG - processed action (default + 0.5*action): {np.round(tgt, 3)}")
                print(f"DEBUG - EE position (WORLD): {ee_pos_world[0].cpu().numpy()}")
                print(f"DEBUG - Target position (WORLD): {target_pos_world[0].cpu().numpy()}")
                print(f"DEBUG - EE quat (wxyz, WORLD):  {np.round(ee_quat_wxyz_np, 4)}")
                print(f"DEBUG - TGT quat (wxyz, WORLD): {np.round(tgt_quat_wxyz_np, 4)}")
                print(f"DEBUG - |dot(q)|: {qdot:.4f}  -> ang_error_deg: {ang_err_deg:.2f}")

                # --- link_base component-wise distances ---
                base_pos_w_dbg, _ = get_base_pose_world(robot, scene, BASE_IDX, robot.device)
                base_xyz = base_pos_w_dbg[0].detach().cpu().numpy()
                target_xyz = target_pos_world[0].detach().cpu().numpy()

                dxdy_dz_world_to_base = base_xyz - np.array([0.0, 0.0, 0.0], dtype=np.float32)
                norm_world_to_base = float(np.linalg.norm(dxdy_dz_world_to_base))

                dxdy_dz_base_to_target = target_xyz - base_xyz
                norm_base_to_target = float(np.linalg.norm(dxdy_dz_base_to_target))

                print(f"DEBUG - link_base pos (WORLD):           {np.round(base_xyz, 3)}")
                print(f"DEBUG - (world→base)  dx,dy,dz:          {np.round(dxdy_dz_world_to_base, 3)}  | norm: {norm_world_to_base:.3f} m")
                print(f"DEBUG - (base→target) dx,dy,dz:          {np.round(dxdy_dz_base_to_target, 3)} | norm: {norm_base_to_target:.3f} m")

                # Optional: forward-axis check (assumes tool -Z is forward)
                R_ee = quat_to_rotmat_np_wxyz(ee_quat_wxyz_np)
                R_tg = quat_to_rotmat_np_wxyz(tgt_quat_wxyz_np)
                ee_forward = -R_ee[:, 2]
                tg_forward = -R_tg[:, 2]
                cos_dir = float(np.clip(np.dot(ee_forward, tg_forward), -1.0, 1.0))
                dir_err_deg = math.degrees(math.acos(cos_dir))
                print(f"DEBUG - forward-axis error (deg): {dir_err_deg:.2f}")

                if dist < 0.05 and ang_err_deg < 10.0:
                    print("🎉 TARGET REACHED (pos+orient)! Add new 'goto X Y Z' or 'quat ...' command.")

            step_count += 1

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")

    print("🏁 Interactive control session ended!")
    simulation_app.close()

if __name__ == "__main__":
    main()

# ========================= REAL ROBOT NOTES =========================
# 1) If your real robot API accepts BASE-frame targets (preferred):
#    - Send:
#         position_B = [x, y, z] in BASE
#         quaternion_B (w,x,y,z) in BASE
#    - Ensure the robot uses the SAME quaternion convention and frame axes as in training.
#    - If the robot expects RPY, convert quaternion_B -> RPY with the SAME Euler order (xyz intrinsic).
#
# 2) If your real robot API expects WORLD-frame targets:
#    - Obtain the robot base pose in WORLD: (p_W_B, q_W_B) at command time.
#    - Compose:
#         q_W_tgt = q_W_B ⊗ q_B_tgt
#         p_W_tgt = p_W_B + R(q_W_B) * p_B_tgt
#    - Send (p_W_tgt, q_W_tgt) to the robot.
#
# 3) Tool/EE offset:
#    - If your real flange->tool has an offset, multiply it the SAME way you do in sim.
#    - Keep conventions identical (handedness, axis directions). A hidden 180° flip often comes from an
#      XYZW vs WXYZ or flange-vs-tool mismatch.
#
# 4) Interpolation:
#    - For smooth motion on the real robot, slerp from current quat to target quat (shortest path).
#    - Quaternions q and -q are the same orientation; canonicalize sign before logging/thresholding.
#
# 5) Quick sanity for a +90° about BASE +Y:
#    - quat_B = [0.70710678, 0.0, 0.70710678, 0.0]  (wxyz)
#    - If the robot shows a very different number for "the same pose", check FRAME and ORDER first.
