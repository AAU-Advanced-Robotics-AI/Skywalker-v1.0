# actions.py

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass



from dataclasses import MISSING
import torch
from typing import Sequence, List

import omni.physics.tensors as physics

from gymnasium.spaces import Box
import numpy as np     # add this import at top of file
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
                grasped = gripper.get_grasped_body()
                if grasped and object_name in grasped.get_prim_path():
                    result[i] = True
            return result.int()
        

    def get_grasping_mask(self) -> torch.Tensor:
        """Returns 1 for each env where the gripper is grasping something."""
        return torch.tensor(
            [g.is_closed() for g in self._surface_grippers],
            dtype=torch.float32,
            device=self.device,
    )


            # --- NEW ---
    def is_closed(self) -> torch.Tensor:
            """Bool per-env flag – True if gripper is closed."""
            closed = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
            for i, g in enumerate(self._surface_grippers):
                closed[i] = g.is_closed()
            return closed
    
    def is_grasping_any(self) -> torch.Tensor:
        """
        Return tensor of shape (num_envs,) indicating if any object is being grasped.
        """
        grasping = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        for i, gripper in enumerate(self._surface_grippers):
            grasping[i] = gripper.get_grasped_object() is not None
        return grasping
    
    def get_rl_action_space(self):
        """Binary gripper: 0 = open, 1 = close (float32 so SAC is happy)."""
        return Box(low=np.array([0.0],  dtype=np.float32),
                high=np.array([1.0],  dtype=np.float32),
                dtype=np.float32)
    
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
        self._offset = []
        self._rot_offset = []
        self._grip_threshold = []
        self._shared_gripper_state_timer = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        

        for env_id in range(self._num_envs):
            prim_path = cfg.gripper_prim_path.format(ENV_REGEX_NS=f"/World/envs/env_{env_id}")

            sgp = Surface_Gripper_Properties()
            sgp.parentPath = prim_path
            #print(prim_path)

            #print("am i a robot", robot.is_valid())
            sgp.d6JointPath = f"{prim_path}/d6FixedJoint"
            sgp.gripThreshold = cfg.grip_threshold

            sgp.bendAngle = 3.14*0.3
            sgp.offset = physics.Transform()
            sgp.offset.p.x = cfg.offset[0]
            sgp.offset.p.y = cfg.offset[1]
            sgp.offset.p.z = cfg.offset[2]

            sgp.offset.r =  cfg.rot_offset #[0, -0.707,0 ,0.707]
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


        self._gripper_timers = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
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


        """
        Convert the continuous action in the range [-1, 1] produced by the
        policy into a binary open / close command for the surface-gripper.

            (-1 …  0]  →  open   (False)
            ( 0  …  1]  →  close (True)

        If a Boolean tensor is provided (e.g. by a scripted reset routine),
        it is passed through unchanged.
        """
        # keep the incoming tensor (NaNs → 0) for logging / replay
        self._raw_actions[:] = torch.nan_to_num(actions)

        # scripted Boolean commands bypass thresholding
        if actions.dtype == torch.bool:
            self._processed_actions[:] = actions.view(-1)
            return

        # numeric input from SAC: threshold at 0
        self._processed_actions[:] = (actions.view(-1) > 0).to(torch.bool)

            # ✴︎ DEBUG
        #if torch.isnan(actions).any():
        #print("[GRIPPER]  input →", actions.cpu())

    def apply_actions(self):
        """Apply open/close commands to each gripper based on processed action."""
        # increment timers every step
        self._gripper_timers += 1
        self._shared_gripper_state_timer += 1

        to_close = torch.nonzero(self._processed_actions, as_tuple=False).squeeze(-1)
        to_open = torch.nonzero(~self._processed_actions, as_tuple=False).squeeze(-1)

        for i in to_close:
            env_id = int(i.item())
            if self._gripper_timers[env_id] < 50:  # wait first 30 steps (~2 seconds)
                continue
            gripper = self._surface_grippers[env_id]
            if not gripper.is_closed():
                gripper.close()
            gripper.update()

        for i in to_open:
            env_id = int(i.item())
            if self._gripper_timers[env_id] < 50:
                continue
            gripper = self._surface_grippers[env_id]
            gripper.open()
            gripper.update()


    # def apply_actions(self):
    #     """Force close all gripper2 for testing."""
    #     for i in range(self._num_envs):
    #         gripper = self._surface_grippers[i]
    #         gripper.close()
    #         gripper.update()


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset action buffers and timers
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = False
        self._gripper_timers[env_ids] = 0
        self._shared_gripper_state_timer[env_ids] = 0


        # Force-open each gripper to fully reset its physical state
        for env_id in env_ids:
            gripper = self._surface_grippers[int(env_id)]
            if gripper.is_closed():
                gripper.open()
            gripper.update()


@configclass
class SurfaceGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SurfaceGripperActionTerm
    gripper_prim_path: str = MISSING
    offset: list[float] = MISSING
    rot_offset: list[float] = MISSING
    grip_threshold: float = MISSING
    #surface_gripper: surface_gripper = S
