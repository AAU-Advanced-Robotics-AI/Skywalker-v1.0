#!/usr/bin/env python3
"""
Interactive Base Move Robot Control

- Loads a TorchScript policy exported by base_move_export.py (it embeds obs normalization).
- Builds the 35-D observation exactly like NEW base_move training:
  [joint_pos(7), joint_vel(7), base_pos_rel_ee(3), base_quat_rel_ee(4), target_pose_command(7), actions(7)]
- Uses EE-relative coordinate system for better real robot deployment consistency.
- Uses ABSOLUTE joint targets like training: q_target = default_joint_pos + 0.5 * action
- Loads the welding scene USD and controls the robot base to move to commanded positions.
- Lets you type targets via /tmp/base_move_commands.txt.

This version controls the BASE position, not the end-effector like the PTP agent.
"""

import argparse
import os
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Isaac Lab / Sim
from isaaclab.app import AppLauncher

# --------------------------- CLI ---------------------------
parser = argparse.ArgumentParser(description="Interactive base move robot control")
parser.add_argument(
    "--model_path",
    type=str,
    default="scripts/SKYWALKER/grab_skywalker/exported_models/base_move_policy.pt",
    help="Path to exported TorchScript model",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --------------------------- Launch Isaac Sim ---------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import the actual base move functions from the training environment
import sys
import os
sys.path.append(os.path.dirname(__file__))
from mdp import base_move_functions as bm

# --------------------------- Scene Configuration ---------------------------
USD_SCENE = "/home/bruno/IsaacLab/scripts/SKYWALKER/welding_scene3.usd"

@configclass
class BaseMoveInteractiveSceneCfg(InteractiveSceneCfg):
    """Interactive scene exactly matching base_move training."""

    num_envs: int = 1
    env_spacing: float = 2.0

    # Load scene once per env at a shallow mount point
    welding_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene",
        spawn=UsdFileCfg(usd_path=USD_SCENE),
    )

    # Wrap existing robot (NO spawning). Do NOT target 'world_joint' — it doesn't exist.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/skywalker_robot/xarm7",
        spawn=None,  # robot already exists inside your scene.usd
        init_state=ArticulationCfg.InitialStateCfg(
            # Values converted from your screenshot (deg → rad):
            joint_pos={
                "joint1":  0.01221730,   # 0.7°
                "joint2": -0.04014257,   # -2.3°
                "joint3":  0.01221730,   # 0.7°
                "joint4":  0.53929221,   # 30.9°
                "joint5":  0.01221730,   # 0.7°
                "joint6": -1.00356287,   # -57.5°
                "joint7":  0.0,          # 0°
            }
        ),
        # IMPORTANT: For base movement, we need the robot to be mobile
        # This may require modifying the welding constraint in the USD
        actuators={
            # 🎯 CRITICAL FIX: Use EXACT PD gains from training to prevent oscillation!
            # big base joints
            "j1_j2": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2"],
                effort_limit_sim=179.4445,        # N·m (per joint cap)
                velocity_limit_sim=3.0,           # rad/s (set to your real speed if you have it)
                stiffness=600.0,                  # EXACT training values
                damping=40.0,
            ),
            # mid joints
            "j3": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit_sim=92.0,
                velocity_limit_sim=3.0,
                stiffness=600.0,
                damping=40.0,
            ),
            "j4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=92.0,
                velocity_limit_sim=3.0,
                stiffness=600.0,
                damping=40.0,
            ),
            "j5": ImplicitActuatorCfg(
                joint_names_expr=["joint5"],
                effort_limit_sim=81.6,
                velocity_limit_sim=3.0,
                stiffness=500.0,                  # Different from base joints!
                damping=35.0,                     # Different from base joints!
            ),
            # wrist joints
            "j6_j7": ImplicitActuatorCfg(
                joint_names_expr=["joint6", "joint7"],
                effort_limit_sim=30.6,
                velocity_limit_sim=4.0,
                stiffness=400.0,                  # Different from other joints!
                damping=30.0,                     # Different from other joints!
            ),
        },
        # Set soft_joint_pos_limits_sim to allow base movement
        soft_joint_pos_limit_factor=1.0,
    )

    # (optional) handles to existing rigid bodies if you need them
    wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/WallAssembly/wall",
        spawn=None,
    )
    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WeldingScene/WallAssembly/anchor",
        spawn=None,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/SkyDome",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            visible_in_primary_ray=False,   # hide HDRI in the background
        ),
    )

    target_pose_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_pose_marker",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # visual-only
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),  # Back to 0.0 - positions are relative
    )

# --------------------------- EE-relative observation helpers ---------------------------
# ✅ REMOVED: Manual implementations replaced with actual training functions!

def get_base_pose_world_for_marker(robot):
    """
    Helper to get absolute base position for marker visualization.
    """
    try:
        base_indices = robot.find_bodies("link_base")
        if base_indices:
            # Handle different return formats from find_bodies
            if isinstance(base_indices, tuple):
                base_idx = base_indices[0][0] if isinstance(base_indices[0], list) else base_indices[0]
            else:
                base_idx = base_indices[0] if isinstance(base_indices, list) else base_indices
            
            base_idx = int(base_idx)
            
            if hasattr(robot.data, "body_pos_w") and hasattr(robot.data, "body_quat_w"):
                pos = robot.data.body_pos_w[:, base_idx, :3]
                quat = robot.data.body_quat_w[:, base_idx, :]  # [w,x,y,z]
                return pos, quat
    except Exception as e:
        print(f"⚠️ Warning: Exception getting base pose for marker: {e}")
    
    # Fallback
    return torch.zeros((1, 3), device=robot.device), torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device)

# --------------------------- xArm7 limits ---------------------------
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

# --------------------------- Command controller ---------------------------
class BaseMoveController:
    """Handles user input for new base target positions via a text file."""

    def __init__(self):
        # 🎯 FIXED: Default target position matching training spawn ranges
        # Training uses pos_x=(-0.15, 0.00), pos_y=(-0.45, 0.15) relative to base
        self.target_position = [0.00, 0.0, 0.0]        # In front of robot (negative X in base frame)
        self.target_orientation = [0.0, 0.0, 0.0]       # RPY (rad) - small yaw within training range ±0.9
        self.new_target = False
        self.running = True
        self.command_file = "/tmp/base_move_commands.txt"
        self.last_command_time = 0
        self.create_command_file()

    def create_command_file(self):
        with open(self.command_file, "w") as f:
            f.write("# Base Move Commands:\n")
            f.write("#   goto X Y YAW  - Move base to position relative to INITIAL base spawn position\n")
            f.write("#   quit\n\n")
            f.write("# 🎯 TRAINING RANGES (stay within these for best results):\n")
            f.write("#   X: -0.15..0.00 (NEGATIVE = in front of robot, positive = behind)\n")
            f.write("#   Y: -0.45..0.15 (negative = robot's right, positive = robot's left)\n") 
            f.write("#   YAW: -0.9..0.9 rad (-51..51 deg)\n")
            f.write("#   Z is always 0 (planar movement)\n")
            f.write("# Example: goto -0.12 0.05 0.3\n")
            f.write("goto -0.10 0.0 0.2\n")
        self.last_command_time = os.path.getmtime(self.command_file)

    def check_for_commands(self):
        try:
            t = os.path.getmtime(self.command_file)
            if t > self.last_command_time:
                self.last_command_time = t
                with open(self.command_file, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
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
                    x, y, yaw = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # 🎯 VALIDATION: Check against training ranges for best results
                    if not (-0.15 <= x <= 0.00):
                        print(f"⚠️  X={x:.3f} outside training range [-0.15, 0.00]. Use negative X for reachable workspace!")
                    if not (-0.45 <= y <= 0.15):
                        print(f"⚠️  Y={y:.3f} outside training range [-0.45, 0.15]")
                    if not (-0.9 <= yaw <= 0.9):
                        print(f"⚠️  YAW={yaw:.3f} outside training range [-0.9, 0.9] rad")
                    
                    self.target_position = [x, y, 0.0]  # Z always 0 for base movement
                    self.target_orientation = [0.0, 0.0, yaw]  # Only yaw rotation
                    self.new_target = True
                    print(f"🎯 New base target: pos=[{x:.3f}, {y:.3f}, 0.0], yaw={yaw:.3f} rad ({yaw*57.3:.1f}°)")
                    print(f"🔍 This means: move base {abs(x):.1f}cm {'forward' if x<0 else 'backward'}, {abs(y):.1f}cm {'right' if y<0 else 'left'}")
                except ValueError:
                    print(f"❌ Invalid coordinates: {parts[1:]}")
            else:
                print(f"❌ Usage: goto X Y YAW (example: goto -0.12 0.05 0.3)")
        else:
            print(f"❌ Unknown command: {command}")

# --------------------------- Main ---------------------------
def main():
    print("🤖 INTERACTIVE Base Move Robot Control")
    print(f"📁 Model: {args_cli.model_path}")
    print("=" * 60)

    # ---- Load TorchScript model ----
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
    scene_cfg = BaseMoveInteractiveSceneCfg()
    scene = InteractiveScene(scene_cfg)

    print("🔄 Initializing...")
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    target_marker = scene["target_pose_marker"]

    # Create a mock environment object to use with base_move_functions
    class MockEnv:
        def __init__(self, scene):
            self.scene = scene

    mock_env = MockEnv(scene)

    # Try to locate base body index
    try:
        base_indices = robot.find_bodies("link_base")
        BASE_IDX = int(base_indices[0]) if base_indices else 0
    except Exception:
        BASE_IDX = 0
    print(f"Using base body index: {BASE_IDX}")

    # Set neutral starting pose
    try:
        neutral = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                               device=robot.device, dtype=torch.float32)
        robot.set_joint_position_target(neutral)
        print("✅ Set neutral joint pose.")
    except Exception as e:
        print(f"⚠️ Could not set neutral pose: {e}")

    # Controller + runtime buffers
    controller = BaseMoveController()
    last_action = torch.zeros((1, 7), device=robot.device, dtype=torch.float32)

    # Joint limits for clamping
    xarm_low, xarm_high = get_xarm7_limits(robot.device)
    
    # 🎯 CRITICAL FIX: Store initial base position as reference frame
    # The training uses the INITIAL spawn position as the reference frame for targets
    initial_base_pos_w, initial_base_quat_w = get_base_pose_world_for_marker(robot)
    print(f"📍 Initial base position (reference frame): {initial_base_pos_w.cpu().numpy().flatten()}")
    
    # 🛡️ ADDED: Warmup period like training (1 second to let robot settle)
    warmup_time = 0.0  # seconds
    warmup_steps = int(warmup_time / 0.01)  # assuming 0.01s timestep
    current_step = 0

    # ---- Helper to build observation like training ----
    def build_base_move_obs():
        """
        Build 35-D observation vector exactly like base_move training.
        
        🚨 CRITICAL: Your training config shows:
        - base_pos_rel_ee  = ObsTerm(func=bm.base_position_relative_to_ee)   # Base relative to EE
        - base_quat_rel_ee = ObsTerm(func=bm.base_orientation_relative_to_ee) # Base relative to EE
        
        But your export script comment mentions world coordinates. We use EE-relative to match training!
        [joint_pos(7), joint_vel(7), base_pos_rel_ee(3), base_quat_rel_ee(4), target_pose_command(7), actions(7)]
        """
        # Joint state (relative to defaults)
        joint_pos = robot.data.joint_pos[:, :7] - robot.data.default_joint_pos[:, :7]  # (1,7)
        joint_vel = robot.data.joint_vel[:, :7] - robot.data.default_joint_vel[:, :7]  # (1,7)

        # 🎯 EE-RELATIVE observations using ACTUAL training functions!
        base_pos_rel_ee = bm.base_position_relative_to_ee(mock_env)      # (1,3) - EXACT training function!
        base_quat_rel_ee = bm.base_orientation_relative_to_ee(mock_env)  # (1,4) - EXACT training function!

        # 🎯 CRITICAL FIX: Target pose command exactly like training
        # Training uses base-relative coordinates for target_pose_command
        # The agent learns: "Given base-EE relative position, move base toward target"
        target_pos = torch.tensor([controller.target_position], device=robot.device, dtype=torch.float32)  # (1,3) - relative to base
        
        # Convert RPY to quaternion [w,x,y,z]
        r = Rotation.from_euler("xyz", controller.target_orientation)
        tq_xyzw_np = np.asarray(r.as_quat(), dtype=np.float32)  # [x,y,z,w] scipy format
        tq_xyzw = torch.from_numpy(tq_xyzw_np).unsqueeze(0).to(robot.device)  # (1,4)
        target_quat = torch.cat([tq_xyzw[:, 3:4], tq_xyzw[:, :3]], dim=1)  # (1,4) [w,x,y,z]
        
        target_pose_command = torch.cat([target_pos, target_quat], dim=1)  # (1,7) - base-relative command

        # Update target marker visualization (convert base-relative to world coordinates)
        # 🎯 CRITICAL FIX: Use INITIAL base position as reference frame, not current position
        marker_pos = initial_base_pos_w + target_pos  # Convert base-relative target to world coordinates
        try:
            target_marker.write_root_pose_to_sim(
                torch.cat([marker_pos, target_quat], dim=1),
                env_ids=torch.tensor([0], device=robot.device)
            )
        except Exception:
            # Try alternative method
            try:
                target_marker.write_root_state_to_sim(
                    torch.cat([marker_pos, target_quat, torch.zeros_like(marker_pos), torch.zeros_like(marker_pos)], dim=1),
                    env_ids=torch.tensor([0], device=robot.device)
                )
            except Exception as e:
                pass  # Skip marker updates if both methods fail

        # Concatenate all observation components (35-D total)
        obs = torch.cat([
            joint_pos,           # 7
            joint_vel,           # 7  
            base_pos_rel_ee,     # 3 - 🎯 NEW: EE-relative base position
            base_quat_rel_ee,    # 4 - 🎯 NEW: EE-relative base orientation
            target_pose_command, # 7
            last_action,         # 7
        ], dim=1)  # (1, 35)

        return obs

    print("🎮 BASE MOVE ROBOT READY!")
    print("🛡️ Warmup period: 1.0 second (robot will stabilize)")
    print("Edit /tmp/base_move_commands.txt (use NEGATIVE X values for reachable workspace)")
    step_count = 0

    try:
        while simulation_app.is_running() and controller.running:
            current_step += 1
            
            # 🛡️ WARMUP: Hold robot in stable pose for first 1 second
            if current_step <= warmup_steps:
                if current_step % 20 == 1:  # Print occasionally
                    remaining = (warmup_steps - current_step) * 0.01
                    print(f"🛡️ Warmup: {remaining:.2f}s remaining...")
                
                # Hold neutral position during warmup
                neutral = torch.tensor([[0.01221730, -0.04014257, 0.01221730, 0.53929221, 0.01221730, -1.00356287, 0.0]],
                                       device=robot.device, dtype=torch.float32)
                robot.set_joint_position_target(neutral)
                scene.write_data_to_sim()
                sim_utils.SimulationContext.instance().step(render=True)
                scene.update(0.01)
                robot.update(sim.get_physics_dt())
                continue
            
            # Check for new commands
            controller.check_for_commands()

            # Build observation
            obs = build_base_move_obs()
            
            # 🔍 DEBUG: Print complete observation breakdown every 50 steps
            if current_step % 50 == 0:
                obs_np = obs.cpu().numpy().flatten()
                print(f"\n📊 Step {current_step} - COMPLETE OBSERVATION BREAKDOWN:")
                print(f"   🦾 Joint Positions [0:7]:   {obs_np[0:7]}")
                print(f"   🏃 Joint Velocities [7:14]:  {obs_np[7:14]}")
                print(f"   📍 Base-EE Relative Pos [14:17]: {obs_np[14:17]}")
                print(f"   🧭 Base-EE Relative Quat [17:21]: {obs_np[17:21]}")
                print(f"   🎯 Target Pose Command [21:28]: {obs_np[21:28]}")
                print(f"      └─ Target Pos: {obs_np[21:24]}")  
                print(f"      └─ Target Quat: {obs_np[24:28]}")
                print(f"   🎬 Last Action [28:35]: {obs_np[28:35]}")
                print(f"   📏 Total obs shape: {obs.shape} (should be [1, 35])")
                print(f"   🎯 Current Target from Controller: pos={controller.target_position}, orient={controller.target_orientation}")

            # Get action from policy 
            with torch.no_grad():
                policy_out = model(obs)  # Policy outputs RAW actions (no scaling applied)
            raw_action = policy_out[:, :7]  

            # Update last action for next observation
            last_action.copy_(raw_action)

            # ✅ CORRECT: Apply 0.5 scaling like the training environment does
            # The exported model gives raw policy outputs, we need to apply environment scaling
            # Training config: JointPositionActionCfg(scale=0.5) - this happens in environment, not policy
            q_default = robot.data.default_joint_pos[:, :7]   # (1,7)
            target_joint_pos = q_default + 0.5*raw_action  # Apply the 0.5 scaling from training!

            # ---- Safety: clamp to USD soft limits (same method as PTP controller)
            target_joint_pos = torch.max(torch.min(target_joint_pos, xarm_high), xarm_low)

            # ---- Apply command, then step sim (same as PTP controller)
            robot.set_joint_position_target(target_joint_pos)
            scene.write_data_to_sim()
            sim_utils.SimulationContext.instance().step(render=True)
            scene.update(0.01)
            robot.update(sim.get_physics_dt())

            # Periodic status
            step_count += 1
            if step_count % 50 == 0:  # More frequent debugging
                # Get current positions for analysis
                base_pos_w, _ = get_base_pose_world_for_marker(robot)
                base_rel_ee = bm.base_position_relative_to_ee(mock_env)
                
                # 🔍 DETAILED DEBUG INFO
                print(f"📊 Step {step_count}:")
                print(f"   🎯 Target (base-relative): [{controller.target_position[0]:.3f}, {controller.target_position[1]:.3f}, {controller.target_position[2]:.3f}]")
                print(f"   🤖 Base-EE relative: [{base_rel_ee[0,0]:.3f}, {base_rel_ee[0,1]:.3f}, {base_rel_ee[0,2]:.3f}]")
                print(f"   🎬 Raw action range: [{raw_action.min().item():.3f}, {raw_action.max().item():.3f}]")
                print(f"   🎛️  Joint pos (first 3): {robot.data.joint_pos[0, :3].cpu().numpy()}")
                
                # 🚨 CHECK: Are EE-relative observations changing when we give different targets?
                if step_count == 50:
                    print(f"   🚨 SANITY CHECK: Base-EE relative should change as robot moves toward targets!")
                    print(f"   🚨 If this stays near zero, the EE-relative calculation might be wrong.")
                
                # Show raw action details
                raw_act = raw_action[0].detach().cpu().numpy()
                q_def = q_default[0].detach().cpu().numpy()
                tgt = target_joint_pos[0].detach().cpu().numpy()
                jp = robot.data.joint_pos[0, :7].detach().cpu().numpy()
                
                print(f"   🔍 Raw action: {np.round(raw_act, 3)}")
                print(f"   🎛️  Current joints: {np.round(jp, 3)}")
                print(f"   🎯 Target joints: {np.round(tgt, 3)}")
                
                # 🚨 CRITICAL CHECK: Let's debug the actual positions
                # Get actual base position in world coordinates using robot's find_bodies method
                base_body_ids, _ = robot.find_bodies("link_base")
                ee_body_ids, _ = robot.find_bodies("link_eef")  # EE body name (capital C)
                base_body_idx = base_body_ids[0]  # First match
                ee_body_idx = ee_body_ids[0]  # First match
                base_pos_world = robot.data.body_pos_w[0, base_body_idx, :3].cpu().numpy()
                ee_pos_world = robot.data.body_pos_w[0, ee_body_idx, :3].cpu().numpy()
                
                print(f"   🌍 Base position (world): [{base_pos_world[0]:.3f}, {base_pos_world[1]:.3f}, {base_pos_world[2]:.3f}]")
                print(f"   🤖 EE position (world): [{ee_pos_world[0]:.3f}, {ee_pos_world[1]:.3f}, {ee_pos_world[2]:.3f}]")
                
                # 🎯 FIXED: Calculate target in world coordinates using INITIAL base position
                initial_base_xy = initial_base_pos_w[0, :2].cpu().numpy()
                target_world = initial_base_xy + np.array(controller.target_position[:2])
                current_base_xy = base_pos_world[:2]
                target_error = np.linalg.norm(target_world - current_base_xy)
                
                print(f"   📍 Initial base (XY): [{initial_base_xy[0]:.3f}, {initial_base_xy[1]:.3f}]")
                print(f"   🎯 Target (world XY): [{target_world[0]:.3f}, {target_world[1]:.3f}] - NOW CONSTANT!")
                print(f"   📏 Target error (XY): {target_error:.3f}m - This should be decreasing!")
                
                # Check if joint commands are reasonable
                joint_command_diff = np.abs(tgt - jp)
                print(f"   🎛️  Max joint command diff: {joint_command_diff.max():.3f} rad ({joint_command_diff.max()*57.3:.1f}°)")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")

    print("🏁 Interactive base move control session ended!")
    simulation_app.close()

if __name__ == "__main__":
    main()
