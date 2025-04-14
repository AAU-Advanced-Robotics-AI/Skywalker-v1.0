# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`XARM7_CFG`: XARM7 ON BASE
* :obj:`XARM7_PANDA_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Joint-specific stiffness and damping values
##
joint_parameters = {
    1: {"stiffness": 34.6213, "damping": 0.01385},
    2: {"stiffness": 37.55463, "damping": 0.01052},
    3: {"stiffness": 52.92135, "damping": 0.02117},
    4: {"stiffness": 88.74968, "damping": 0.0355},
    5: {"stiffness": 69.78064, "damping": 0.02791},
    6: {"stiffness": 5.31709, "damping": 0.00213},
    7: {"stiffness": 3.70526, "damping": 0.00148},
}

##
# Configuration
##

XARM7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bruno/IsaacLab/scripts/SKYWALKER/Collected_skywalker_robot/skywalker_robot.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            #coming from the video, trying to fix wobbling
            #max_linear_velocity = 1000.0,
            #max_angular_velocity = 1000.0,
            #rigid_body_enabled=True,
            #enable_gyroscopic_forces=True,
            #end of the things from the video
            max_depenetration_velocity=5.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, #also from the video
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0,
            #also coming from the video
            #stabilization_threshold=0.001,
            #sleep_threshold=0.005
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0,
            "joint3": 0.0,
            "joint4": 0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        },
        joint_vel={
            "joint1": 0.0,
            "joint2": 0,
            "joint3": 0.0,
            "joint4": 0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,            
        }
    ),
    actuators={
        # Define actuators for each joint with individual stiffness and damping values
            "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # "xarm7_actuator_1": ImplicitActuatorCfg(
        #     joint_names_expr=["joint1"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[1]["stiffness"],
        #     damping=joint_parameters[1]["damping"],
        # ),
        # "xarm7_actuator_2": ImplicitActuatorCfg(
        #     joint_names_expr=["joint2"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[2]["stiffness"],
        #     damping=joint_parameters[2]["damping"],
        # ),
        # "xarm7_actuator_3": ImplicitActuatorCfg(
        #     joint_names_expr=["joint3"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[3]["stiffness"],
        #     damping=joint_parameters[3]["damping"],
        # ),
        # "xarm7_actuator_4": ImplicitActuatorCfg(
        #     joint_names_expr=["joint4"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[4]["stiffness"],
        #     damping=joint_parameters[4]["damping"],
        # ),
        # "xarm7_actuator_5": ImplicitActuatorCfg(
        #     joint_names_expr=["joint5"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[5]["stiffness"],
        #     damping=joint_parameters[5]["damping"],
        # ),
        # "xarm7_actuator_6": ImplicitActuatorCfg(
        #     joint_names_expr=["joint6"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[6]["stiffness"],
        #     damping=joint_parameters[6]["damping"],
        # ),
        # "xarm7_actuator_7": ImplicitActuatorCfg(
        #     joint_names_expr=["joint7"],
        #     effort_limit_sim=87.0,
        #     velocity_limit_sim=2.175,
        #     stiffness=joint_parameters[7]["stiffness"],
        #     damping=joint_parameters[7]["damping"],
        # ),
    },
    #soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


XARM7_HIGH_PD_CFG = XARM7_CFG.copy()
# XARM7_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_1"].stiffness = joint_parameters[1]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_2"].stiffness = joint_parameters[2]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_3"].stiffness = joint_parameters[3]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_4"].stiffness = joint_parameters[4]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_5"].stiffness = joint_parameters[5]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_6"].stiffness = joint_parameters[6]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_7"].stiffness = joint_parameters[7]["stiffness"] * 4
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_1"].damping = joint_parameters[1]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_2"].damping = joint_parameters[2]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_3"].damping = joint_parameters[3]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_4"].damping = joint_parameters[4]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_5"].damping = joint_parameters[5]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_6"].damping = joint_parameters[6]["damping"] * 500
# XARM7_HIGH_PD_CFG.actuators["xarm7_actuator_7"].damping = joint_parameters[7]["damping"] * 500

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
