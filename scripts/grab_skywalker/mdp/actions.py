# actions.py

from dataclasses import MISSING
import torch
from typing import Sequence

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

@configclass
class SurfaceGripperActionTerm(ActionTerm):
    """
    Action term to control the Surface Gripper's open/close state.
    
    Surface gripper commands:
    - command < -0.3: Open gripper
    - -0.3 <= command <= 0.3: Idle (maintain current state)  
    - command > 0.3: Close gripper
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get the surface gripper from the scene
        self._gripper_name = cfg.gripper_name
        
        # Raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)

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
        """Process actions for surface gripper commands."""
        self._raw_actions[:] = actions
        
        # For the cylinder gripper (stability gripper), we want it to be engaged 
        # by default to provide stability unless explicitly commanded to open
        if self._gripper_name == "SurfaceGripper_Cylinder":
            # Default to closed (engaged) for stability
            # Only open if action is strongly negative (< -0.5)
            processed = torch.where(actions.squeeze() < -0.5, actions.squeeze(), torch.ones_like(actions.squeeze()))
            self._processed_actions[:] = processed.unsqueeze(-1)
        else:
            # Direct mapping for EE gripper: the actions are the gripper commands
            self._processed_actions[:] = actions

    def apply_actions(self):
        """Apply gripper commands to the surface gripper."""
        # Get the surface gripper from the scene
        if hasattr(self._env.scene, "surface_grippers") and self._gripper_name in self._env.scene.surface_grippers:
            gripper = self._env.scene.surface_grippers[self._gripper_name]
            # Set gripper commands
            gripper.set_grippers_command(self._processed_actions)
            
            # Enhanced debug output every 50 steps
            if hasattr(self._env, '_step_count') and self._env._step_count % 50 == 0:
                # Get gripper state (-1=Open, 0=Closing, 1=Closed)
                gripper_state = gripper.state[0].item() if hasattr(gripper, 'state') else "Unknown"
                gripper_command = gripper.command[0].item() if hasattr(gripper, 'command') else "Unknown"
                
                print(f"[{self._gripper_name}] Command: {self._processed_actions[0].item():.3f}, "
                      f"State: {gripper_state:.1f}, "
                      f"BufferCmd: {gripper_command:.3f}, "
                      f"Is_Closed: {self.is_closed()[0].item():.0f}, "
                      f"Raw_Action: {self._raw_actions[0].item():.3f}")
        else:
            print(f"Warning: Gripper '{self._gripper_name}' not found in scene.surface_grippers")

    def is_closed(self) -> torch.Tensor:
        """Returns whether each gripper is in the closed state.
        
        Based on the surface gripper command convention:
        - command > 0.3: Gripper is Closing/Closed
        - -0.3 <= command <= 0.3: Gripper is Idle
        - command < -0.3: Gripper is Opening/Open
        """
        return (self._processed_actions.squeeze() > 0.3).float()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the actions."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class SurfaceGripperActionCfg(ActionTermCfg):
    """Configuration for surface gripper action term."""
    class_type: type[ActionTerm] = SurfaceGripperActionTerm
    gripper_name: str = ""  # Name of gripper in scene.surface_grippers
