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
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
    
"""
Proprioceptive observations
"""

def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    return height.unsqueeze(-1)

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

def reference_command(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Reference foot position of the robot.

    The reference foot position is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    return action_term.original_command

def contact_forces(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor")
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return contact_forces.reshape(-1, contact_forces.shape[1] * contact_forces.shape[2])

"""
Exteroceptive observations
"""

def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, reshape_as_image: bool=False) -> torch.Tensor:
    """
    Height scan from the given sensor w.r.t. body frame
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_scan_b = (sensor.data.ray_hits_w[..., 2]  - sensor.data.pos_w[:, 2].unsqueeze(1)).clamp(-1.0, 0.0)
    if reshape_as_image:
        resolution = sensor.cfg.pattern_cfg.resolution
        grid_x, grid_y = sensor.cfg.pattern_cfg.size
        width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
        height_scan_b = height_scan_b.reshape(-1, 1, height, width).clamp(-1.0, 0.0)
    return height_scan_b

def depth_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth_image = sensor.data.output["distance_to_camera"]
    num_envs = depth_image.shape[0]
    depth_image_flat = depth_image.reshape(num_envs, -1)
    return depth_image_flat 

def foot_centric_height_scan(env: ManagerBasedEnv, action_name: str = "mpc_action")->torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return action_term.grid_point_height

def terrain_roughness(
        env: ManagerBasedEnv, 
        left_raycaster_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_L_foot"),
        right_raycaster_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_R_foot")
        ) -> torch.Tensor:
    left_foot_raycaster: RayCaster = env.scene.sensors[left_raycaster_cfg.name]
    right_foot_raycaster: RayCaster = env.scene.sensors[right_raycaster_cfg.name]
    left_foot_height_map = left_foot_raycaster.data.ray_hits_w[..., 2]
    right_foot_height_map = right_foot_raycaster.data.ray_hits_w[..., 2]
    roughness_left = left_foot_height_map.max(dim=1).values - left_foot_height_map.min(dim=1).values # (num_envs, )
    roughness_right = right_foot_height_map.max(dim=1).values - right_foot_height_map.min(dim=1).values # (num_envs, )
    roughness_left[roughness_left > 0.2] = 0.0  #  outlier rejection
    roughness_right[roughness_right > 0.2] = 0.0 # outlier rejection
    return torch.stack([roughness_left, roughness_right], dim=1)  # (num_envs, 2)

"""
MPC states
"""

def swing_phase(env: ManagerBasedEnv, action_name: str = "mpc_action")-> torch.Tensor:
    """Swing phase of the robot.

    The swing phase is defined as the phase where the foot is not in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    swing_phase = action_term.swing_phase
    return swing_phase

def foot_position_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Foot position of the robot.

    The foot position is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    # foot_position_b = action_term.foot_pos_b # from mpc controller
    foot_position_b = action_term.robot_api.foot_pos_b.reshape(-1, 6) # from simulation
    return foot_position_b

def foot_placement_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """
    Planned foothold calculated from heuristics planner. 
    Exlude z position. 
    
    Return: 
        foot_placement_b: torch.Tensor (num_envs, 4)
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    foot_placement = action_term.foot_placement[:, [0, 1, 3, 4]]
    return foot_placement

def reference_foot_position_b(env: ManagerBasedEnv, action_name: str = "mpc_action") -> torch.Tensor:
    """Reference foot position of the robot.

    The reference foot position is defined as the position of the foot in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    reference_foot_position_b = action_term.ref_foot_pos_b
    return reference_foot_position_b