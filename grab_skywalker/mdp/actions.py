# actions.py

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
import torch
from typing import Sequence, List

from isaaclab.envs import ManagerBasedEnv
import importlib

@configclass
class SurfaceGripperActionTerm(ActionTerm):
    """
    Action term to control the Surface Gripper's open/close state.

    In this case:
    - Open action: Positive float or True.
    - Close action: Negative float or False.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # --- DYNAMIC / LAZY IMPORT (post-extension-enable) ---
        try:
            surface_mod = importlib.import_module("isaacsim.robot.surface_gripper")
        except ImportError as e:
            raise RuntimeError(
                "isaacsim.robot.surface_gripper extension is not loaded. "
                "Make sure to enable it before creating the environment."
            ) from e

        SurfaceGripper = surface_mod.SurfaceGripper

        self._surface_grippers: List[SurfaceGripper] = []
        self._gripper_prim_paths = []
        self._num_envs = self.num_envs
        device = self.device

        # Initialize grippers for each env
        for env_id in range(self._num_envs):
            prim_path = cfg.gripper_prim_path.format(ENV_REGEX_NS=f"/World/envs/env_{env_id}")
            gripper = SurfaceGripper(
                translate=0.047,
                direction="z",
                grip_threshold=5.5,
                force_limit=35,
                torque_limit=35,
                kp=100.0,
                kd=100.0,
                disable_gravity=False,
            )
            gripper.initialize(prim_path)
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
            if gripper._virtual_gripper is not None:
                try:
                    gripper.close()
                except Exception as e:
                    print(f"Error closing gripper {i.item()}: {e}")
            else:
                print("No gripper to close found")
            gripper.update()

        for i in to_open:
            gripper = self._surface_grippers[int(i.item())]
            if gripper._virtual_gripper is not None:
                try:
                    gripper.open()
                except Exception as e:
                    print(f"Error opening gripper {i.item()}: {e}")
            else:
                print("No gripper to open found")
            gripper.update()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset the actions
        self._raw_actions[env_ids] = 0.0


@configclass
class SurfaceGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SurfaceGripperActionTerm
    gripper_prim_path: str = MISSING
