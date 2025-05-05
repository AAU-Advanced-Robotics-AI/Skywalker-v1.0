from pxr import Usd, UsdGeom, UsdPhysics, Sdf

# 1) Create an empty stage
stage = Usd.Stage.CreateNew("wall_cube.usd")

# 2) Define the wall at /Wall
wall = UsdGeom.Xform.Define(stage, "/Wall")
# collision + kinematic rigid body
UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
rb_wall = UsdPhysics.RigidBodyAPI.Apply(wall.GetPrim())
rb_wall.CreateKinematicAttr().Set(True)
rb_wall.CreateMassAttr().Set(1e7)

# 3) Define the cube at /Object
cube = UsdGeom.Xform.Define(stage, "/Object")
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
rb_cube = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
rb_cube.CreateMassAttr().Set(1.0)  # dynamic by default (gravity on)

# 4) Fixed‐joint between them
joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/WallCubeJoint"))
joint.CreateBody0Rel().SetTargets([Sdf.Path("/Wall")])
joint.CreateBody1Rel().SetTargets([Sdf.Path("/Object")])

# 5) Save the USD file
stage.GetRootLayer().Save()
