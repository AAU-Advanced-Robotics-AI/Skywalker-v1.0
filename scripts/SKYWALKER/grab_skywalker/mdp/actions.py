# actions.py

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
import torch
from typing import Sequence, List

import omni.physics.tensors as physics

from isaaclab.envs import ManagerBasedEnv
import importlib

import carb

#from omni.isaac.core.articulations import Articulation 

# Access the filesystem interface
#fs = carb.filesystem

# Check where the logs are stored  # Adjust based on known directories

#print(f"Log path: {fs}")
# Set global log verbosity to 'info' to see info/warning/debug logs

#from isaacsim.robot.surface_gripper._surface_gripper import Surface_Gripper_Properties,  Surface_Gripper 

@configclass
class SurfaceGripperActionTerm(ActionTerm):
    """
    Action term to control the Surface Gripper's open/close state.

    In this case:
    - Open action: Positive float or True.
    - Close action: Negative float or False.
    """
    def is_grasping(self, object_name: str) -> torch.Tensor:
            result = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
            for i, gripper in enumerate(self._surface_grippers):
                grasped = gripper.get_grasped_object()
                if grasped and object_name in grasped.get_prim_path():
                    result[i] = True
            return result.int()
        

            # --- NEW ---
    def is_closed(self) -> torch.Tensor:
            """Bool per-env flag – True if gripper is closed."""
            closed = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
            for i, g in enumerate(self._surface_grippers):
                closed[i] = g.is_closed()
            return closed
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)


            # Just add this method inside the SurfaceGripperActionTerm class (outside __init__)
    

        # --- DYNAMIC / LAZY IMPORT (post-extension-enable) ---
        try:
            surface_mod = importlib.import_module("isaacsim.robot.surface_gripper._surface_gripper")
        except ImportError as e:
            raise RuntimeError(
                "isaacsim.robot.surface_gripper extension is not loaded. "
                "Make sure to enable it before creating the environment."
            ) from e
        
        Surface_Gripper = surface_mod.Surface_Gripper
        Surface_Gripper_Properties = surface_mod.Surface_Gripper_Properties

        self._surface_grippers: List[Surface_Gripper] = []
        self._gripper_prim_paths = []
        self._num_envs = self.num_envs

        for env_id in range(self._num_envs):
            prim_path = cfg.gripper_prim_path.format(ENV_REGEX_NS=f"/World/envs/env_{env_id}")

            sgp = Surface_Gripper_Properties()
            sgp.parentPath = prim_path
            print(prim_path)

            #print("am i a robot", robot.is_valid())
            sgp.d6JointPath = f"{prim_path}/d6FixedJoint"
            sgp.gripThreshold = 0.1

            sgp.bendAngle = 3.14*0.3
            sgp.offset = physics.Transform()
            sgp.offset.p.x = 0.0
            sgp.offset.p.y = 0.0
            sgp.offset.p.z = 0.01

            sgp.offset.r =  [0, -0.707,0 ,0.707]
            #sgp.offset.r = [0, 0.7171,0,0.7171]
            sgp.forceLimit = 50000
            sgp.torqueLimit = 50000
            sgp.stiffness = 100
            sgp.damping = 100
            sgp.disableGravity = True
            sgp.retryClose = False
            gripper = Surface_Gripper()
            gripper.initialize(sgp)

            self._surface_grippers.append(gripper)
            self._gripper_prim_paths.append(prim_path)



        self._raw_actions = torch.zeros(self._num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Converts raw actions into open/close binary flags."""
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            # True = close, False = open
            self._processed_actions[:] = actions.view(-1) == 0
        else:
            # Close if action < 0, else open
            self._processed_actions[:] = actions.view(-1) < 0

    def apply_actions(self):
        """Apply open/close commands to each gripper based on processed action."""
        to_close = torch.nonzero(self._processed_actions, as_tuple=False).squeeze(-1)
        to_open = torch.nonzero(~self._processed_actions, as_tuple=False).squeeze(-1)

        for i in to_close:
            gripper = self._surface_grippers[int(i.item())]
            if not gripper.is_closed():
                result = gripper.close()
            gripper.update()

        for i in to_open:
            gripper = self._surface_grippers[int(i.item())]
            result = gripper.open()
            gripper.update()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset the actions
        self._raw_actions[env_ids] = 0.0


@configclass
class SurfaceGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SurfaceGripperActionTerm
    gripper_prim_path: str = MISSING
    #surface_gripper: surface_gripper = S
