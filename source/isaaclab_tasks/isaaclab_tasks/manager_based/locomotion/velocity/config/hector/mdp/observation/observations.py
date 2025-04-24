# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
    
"""
General
"""

def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_names: list[str]=[]) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_pos[:, 2] += torch.pi/4
    joint_pos[:, 3] -= torch.pi/2
    joint_pos[:, 4] += torch.pi/4
    joint_pos[:, 7] += torch.pi/4
    joint_pos[:, 8] -= torch.pi/2
    joint_pos[:, 9] += torch.pi/4
    return joint_pos

def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_names: list[str]=[]) -> torch.Tensor:
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return asset.data.joint_vel[:, joint_ids]

def joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_names: list[str]=[]) -> torch.Tensor:
    """The joint torques of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their torques returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return asset.data.applied_torque[:, joint_ids]


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. body frame

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ray_hits_w[..., 2]  - sensor.data.pos_w[:, 2].unsqueeze(1)


"""
mpc states
"""

def swing_phase(env: ManagerBasedEnv, action_name: str = "mpc_action")-> torch.Tensor:
    """Swing phase of the robot.

    The swing phase is defined as the phase where the foot is not in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    swing_phase = action_term.swing_phase
    return swing_phase

def foot_placement_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Foot placement of the robot.

    The foot placement is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    foot_placement_b = action_term.foot_placement_b
    return foot_placement_b

def foot_position_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Foot position of the robot.

    The foot position is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    foot_position_b = action_term.foot_pos_b
    return foot_position_b

def reference_foot_position_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Reference foot position of the robot.

    The reference foot position is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    reference_foot_position_b = action_term.ref_foot_pos_b
    return reference_foot_position_b