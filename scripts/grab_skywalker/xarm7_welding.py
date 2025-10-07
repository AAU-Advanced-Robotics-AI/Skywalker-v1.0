# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the XARM7 robot with welding scene setup.

This configuration uses the welding_scene3.usd file which contains the robot
with wall and anchor already integrated with correct joint positions.

The following configurations are available:

* :obj:`XARM7_WELDING_CFG`: XARM7 from welding scene USD with integrated wall and anchor

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

XARM7_WELDING_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bruno/IsaacLab/scripts/SKYWALKER/welding_scene3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0,
            fix_root_link=False,  # Allow base to move
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Joint positions will be inherited from the USD file's default poses
        joint_pos={},  # Empty - use USD file defaults
        joint_vel={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,            
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of XARM7 robot from welding scene USD file with integrated wall and anchor."""
