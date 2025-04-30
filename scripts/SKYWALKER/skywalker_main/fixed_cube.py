# skywalker_main/fixed_cube.py   (new file)

from isaaclab.assets import RigidObject

class FixedCube(RigidObject):
    """RigidObject that never pushes linear / angular velocities to PhysX.

    Works with kinematic *or* static actors – perfect for immovable grasp targets.
    """
    def write_root_velocity_to_sim(self, *_, **__):
        # Skip the call that triggers
        #   PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!
        return
