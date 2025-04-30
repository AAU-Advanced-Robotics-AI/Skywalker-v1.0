# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_reached_goal(
    env: ManagerBasedRLEnv,
    goal_pos: list[float] = [0.8, 0.0],
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot base reaches a goal position."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]
    goal = torch.tensor(goal_pos, device=root_pos.device)
    return torch.norm(root_pos - goal, dim=1) < threshold

