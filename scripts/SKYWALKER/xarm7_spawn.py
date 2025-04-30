# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg 
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import random
##
# Pre-defined configs
##
# isort: off
from xarm7 import XARM7_CFG

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create the origins and positions for the robots
    origins = define_origins(num_origins=20, spacing=6.0)

    # Dictionary to store robot instances by their origin names
    scene_entities = {}

    num_grappling_points = 10
    
    # Iterate over each origin and spawn an xArm7 robot at each location
    for i, origin in enumerate(origins):
        # Create the group for the origin (e.g., "Origin1", "Origin2", ...)
        origin_name = f"/World/Origin{i+1}"
        prim_utils.create_prim(origin_name, "Xform", translation=origin)

        # Set up the xArm7 robot configuration
        xarm7_arm_cfg = XARM7_CFG.copy()
        xarm7_arm_cfg.prim_path = f"{origin_name}/Robot"
        xarm7_arm_cfg.init_state.pos = (0.0, 0.0, 0.474)  # Position above the origin
        xarm7 = Articulation(cfg=xarm7_arm_cfg)

        # Add the robot to the scene dictionary
        scene_entities[f"xarm7_{i+1}"] = xarm7

        # Define the size of the wall (cuboid) and the wall position
        wall_size = [0.1, 4.0, 2.0]  # [width, depth, height]
        wall_position = f"{origin_name}/Wall"  # A unique name for the wall
        wall_translation = (1.6, 0.0, 0.0)

        # Create the cuboid wall using the CuboidCfg from sim_utils
        wall_cfg = sim_utils.CuboidCfg(
            size=wall_size,  # Set the wall size (width, depth, height)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )

        wall_cfg.func(wall_position, wall_cfg, translation=wall_translation) 
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 800 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            print(entities.values())
            for index, robot in enumerate(entities.values()):
                # root state
                print(robot)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                #set joint positions
                #joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                #joint_pos += torch.randn_like(joint_pos) * 0.1 
                #robot.write_joint_state_to_sim(joint_pos, joint_vel)
                #clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        # for robot in entities.values():
        #     # generate random joint positions
        #     joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        #     joint_pos_target = joint_pos_target.clamp_(
        #         robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        #     )
        #     # apply action to the robot
        #     robot.set_joint_position_target(joint_pos_target)
        #     # write data to sim
        #     robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
