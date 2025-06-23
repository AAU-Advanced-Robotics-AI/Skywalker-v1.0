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


def reached_goal(env) -> torch.Tensor:
    return (getattr(env, "stage") >= 4).float()   # (N,)

def robot_reached_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Terminate when the robot base reaches its per-env goal marker."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]
    return torch.norm(root_pos - goal_pos, dim=1) < threshold


