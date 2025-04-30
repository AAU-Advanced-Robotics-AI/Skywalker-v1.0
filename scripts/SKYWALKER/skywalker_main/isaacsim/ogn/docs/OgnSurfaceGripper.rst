.. _isaacsim_robot_surface_gripper_SurfaceGripper_1:

.. _isaacsim_robot_surface_gripper_SurfaceGripper:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Surface Gripper
    :keywords: lang-en omnigraph node isaacSim WriteOnly compute-on-request surface_gripper surface-gripper


Surface Gripper
===============

.. <description>

Surface Gripper

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.robot.surface_gripper<ext_isaacsim_robot_surface_gripper>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Bend Angle (*inputs:BendAngle*)", "``float``", "maximum bend angle, degrees", "7.5"
    "Close (*inputs:Close*)", "``execution``", "call to close gripper", "None"
    "Damping (*inputs:Damping*)", "``float``", "Gripper damping", "1000.0"
    "Delta (*inputs:Delta*)", "``float``", "time since last step in seconds", "0.0"
    "Disable Gravity (*inputs:DisableGravity*)", "``bool``", "flag to disable gravity of picked object to compensate for object's mass on robotic controllers", "True"
    "Force Limit (*inputs:ForceLimit*)", "``float``", "Gripper breaking force", "1000000.0"
    "Grip Position (*inputs:GripPosition*)", "``target``", "The point at which objects will be gripped", "None"
    "Grip Threshold (*inputs:GripThreshold*)", "``float``", "How far from an object it allows the gripper to lock in. Object will be pulled in this distance when gripper is closed", "0.01"
    "Open (*inputs:Open*)", "``execution``", "call to Open gripper", "None"
    "Parent Rigid Body (*inputs:ParentRigidBody*)", "``target``", "The rigid body that is used as a surface Gripper", "None"
    "Retry Close (*inputs:RetryClose*)", "``bool``", "Flag to indicate if gripper should keep attempting to close until it grips some object", "False"
    "Stiffness (*inputs:Stiffness*)", "``float``", "Gripper stiffness", "10000.0"
    "Torque Limit (*inputs:TorqueLimit*)", "``float``", "Torque breaking limit", "1000000.0"
    "Enabled (*inputs:enabled*)", "``bool``", "node does not execute if disabled", "True"
    "On Step (*inputs:onStep*)", "``execution``", "step to animate textures", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Closed (*outputs:Closed*)", "``bool``", "Surface gripper is closed or not", "None"
    "Grip Broken (*outputs:GripBroken*)", "``execution``", "triggered when surface gripper unexpectedly breaks open", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.robot.surface_gripper.SurfaceGripper"
    "Version", "1"
    "Extension", "isaacsim.robot.surface_gripper"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Surface Gripper"
    "Categories", "isaacSim"
    "__categoryDescriptions", "isaacSim,Surface Gripper inside Isaac Sim"
    "Generated Class Name", "OgnSurfaceGripperDatabase"
    "Python Module", "isaacsim.robot.surface_gripper"

