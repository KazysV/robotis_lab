# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_rel_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint positions for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]

def joint_vel_rel_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint velocities for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.03,
    gripper_close_threshold: torch.tensor = torch.tensor([0.6]),
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        robot.data.joint_pos[:, -1] >= gripper_close_threshold.to(env.device),
    )
    grasped = torch.logical_and(
        grasped, robot.data.joint_pos[:, -2] >= gripper_close_threshold.to(env.device)
    )
    return grasped

