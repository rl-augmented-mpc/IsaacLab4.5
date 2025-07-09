# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

from datetime import datetime
import numpy as np
import cv2
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

""" 
helper functions 
"""

def bilinear_interpolation(
    sensor_offset: tuple,
    foot_position_2d: torch.Tensor, 
    costmap_2d: torch.Tensor,
    resolution: float = 0.1,
):
    """
    foot position_2d: (num_envs, 2, N, 2)
    costmap_2d: (num_envs, height, width)
    
    x is 2D points in float and P1-P4 are the neighboring grid points.
    P1 ----- P2
    |    x   |
    P3 ----- P4
    """
    num_envs = foot_position_2d.shape[0]
    num_foot = foot_position_2d.shape[1]
    num_foot_edge = foot_position_2d.shape[2]
    height, width = costmap_2d.shape[1:3]
    body_center_in_image_space = (height//2 - int(sensor_offset[1]/resolution), width//2 - int(sensor_offset[0]/resolution))
    
    # in image space
    foot_position_2d_flat = foot_position_2d.clone().reshape(num_envs, -1, 2)
    foot_position_discrete = torch.zeros_like(foot_position_2d_flat)
    foot_position_discrete[:, :, 0] = (body_center_in_image_space[0] + (foot_position_2d_flat[:, :, 1]/resolution)).clamp(0, height-1) # row 
    foot_position_discrete[:, :, 1] = (body_center_in_image_space[1] + (foot_position_2d_flat[:, :, 0]/resolution)).clamp(0, width-1) # col
    
    # (num_envs, num_foot_edge*num_foot, 2)
    P1 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P2 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P3 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P4 = torch.zeros_like(foot_position_discrete).to(torch.long)
    
    P1[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P1[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P2[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P2[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    P3[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P3[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P4[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P4[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    
    # (num_envs, num_foot_edge*num_foot)
    env_ids = torch.arange(num_envs).view(-1, 1).repeat(1, num_foot_edge*num_foot).view(-1)
    P1_height = costmap_2d.reshape(num_envs, -1).repeat(num_foot_edge*num_foot, 1)[env_ids, (P1[:, :, 0] * width + P1[:, :, 1]).view(-1)].view(num_envs, num_foot_edge*num_foot)
    P2_height = costmap_2d.reshape(num_envs, -1).repeat(num_foot_edge*num_foot, 1)[env_ids, (P2[:, :, 0] * width + P2[:, :, 1]).view(-1)].view(num_envs, num_foot_edge*num_foot)
    P3_height = costmap_2d.reshape(num_envs, -1).repeat(num_foot_edge*num_foot, 1)[env_ids, (P3[:, :, 0] * width + P3[:, :, 1]).view(-1)].view(num_envs, num_foot_edge*num_foot)
    P4_height = costmap_2d.reshape(num_envs, -1).repeat(num_foot_edge*num_foot, 1)[env_ids, (P4[:, :, 0] * width + P4[:, :, 1]).view(-1)].view(num_envs, num_foot_edge*num_foot)
    
    # Calculate distances from the foot position to each corner
    dist_P1 = (foot_position_discrete - P1).float().abs()
    dist_P2 = (foot_position_discrete - P2).float().abs()
    dist_P3 = (foot_position_discrete - P3).float().abs()
    dist_P4 = (foot_position_discrete - P4).float().abs()

    # Compute weights based on inverse distance
    weights_P1 = (1.0 / (dist_P1[:, :, 0] * dist_P1[:, :, 1] + 1e-6))
    weights_P2 = (1.0 / (dist_P2[:, :, 0] * dist_P2[:, :, 1] + 1e-6))
    weights_P3 = (1.0 / (dist_P3[:, :, 0] * dist_P3[:, :, 1] + 1e-6))
    weights_P4 = (1.0 / (dist_P4[:, :, 0] * dist_P4[:, :, 1] + 1e-6))

    # Normalize weights
    total_weights = weights_P1 + weights_P2 + weights_P3 + weights_P4
    weights_P1 /= total_weights
    weights_P2 /= total_weights
    weights_P3 /= total_weights
    weights_P4 /= total_weights

    # Compute weighted average height
    height_at_foot = (
        weights_P1 * P1_height +
        weights_P2 * P2_height +
        weights_P3 * P3_height +
        weights_P4 * P4_height
    )
    
    height_at_foot = height_at_foot.reshape(-1, num_foot, num_foot_edge).transpose(1, 2) # (num_envs, num_foot_edge, num_foot)
    
    return height_at_foot

def get_ground_roughness_at_landing_point(
    num_envs: int,
    sensor_offset: tuple, 
    foot_position: torch.Tensor,
    costmap_2d: torch.Tensor,
    resolution: float = 0.1,
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
    ):
    """
    This function returns flatness value which is the difference between the max and min height in a 3x3 grid. 
    So, if robot is stepping on non-flat surface, flatness value results in terrain height value.
    """
    num_envs = costmap_2d.shape[0]
    device = costmap_2d.device
    
    # in cartesian space
    foot_position_2d = foot_position.clone()[:, :, :2] # foot position wrt body frame
    num_samples = 4
    X = torch.linspace(-l_heel, l_toe, num_samples, device=device)
    Y = torch.linspace(-l_width/2, l_width/2, num_samples, device=device)
    XX, YY = torch.meshgrid(X, Y, indexing='ij') # (num_samples, num_samples)
    foot_sample_points = torch.stack((XX, YY), dim=2).view(num_samples*num_samples, 2).unsqueeze(0).unsqueeze(1).repeat(num_envs, 2, 1, 1) # (num_envs, 2, N, 2)
    foot_sample_points[:, 0, :, :] += foot_position_2d[:, 0, None, :]
    foot_sample_points[:, 1, :, :] += foot_position_2d[:, 1, None, :]
    
    height_at_foot = bilinear_interpolation(
        sensor_offset=sensor_offset,
        foot_position_2d=foot_sample_points, 
        costmap_2d=costmap_2d, 
        resolution=resolution
    ) # (num_envs, num_samples, 2)
    
    # roughness_at_foot = torch.abs(height_at_foot.max(dim=1).values - height_at_foot.min(dim=1).values) # (num_envs, 2)
    # roughness_at_foot = (roughness_at_foot * foot_selection).max(dim=1).values # (num_envs,) get worst case roughness
    # roughness_at_foot = roughness_at_foot * (roughness_at_foot > 1e-3).float() # filter small value to be 0
    # std = (torch.std(height_at_foot, dim=1) * foot_selection).max(dim=1).values # (num_envs,)
    
    roughness_at_foot = torch.abs(height_at_foot.max(dim=1).values - height_at_foot.min(dim=1).values) # (num_envs, 2)
    roughness_at_foot = roughness_at_foot * (roughness_at_foot > 1e-3).float() # filter small value to be 0
    std = torch.std(height_at_foot, dim=1) # (num_envs, 2)
    
    return roughness_at_foot, std

def discrete_terrain_costmap(
    height_map_2d: torch.Tensor,
):
    costmap_2d = torch.zeros_like(height_map_2d)
    edgemap_2d = torch.zeros_like(height_map_2d)
    edgemap_2d[:, :, 1:] = torch.abs(height_map_2d[:, :, :-1] - height_map_2d[:, :, 1:])
    costmap_2d[:, :, 1:-1] = (torch.abs(edgemap_2d[:, :, 2:] - edgemap_2d[:, :, 1:-1]) + \
                            torch.abs(edgemap_2d[:, :, :-2] - edgemap_2d[:, :, 1:-1]))/2
    
    return costmap_2d

""" 
rewards
"""

def track_command_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    action_name: str,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - action_term.original_command[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_command_ang_vel_z_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    action_name: str, 
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_name)
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - action_term.original_command[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def track_torso_height_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor"), 
    reference_height: float=0.5, 
    std:float=0.5) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    
    reward = torch.exp(-torch.square(height - reference_height)/std**2) # exponential reward
    return reward

def active_action_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    action_name: str = "mpc_action",
):
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map = raycaster.data.ray_hits_w[..., 2]  - raycaster.data.pos_w[:, 2].unsqueeze(1)
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
    heightmap_2d = height_map.view(-1, height, width)
    
    window = 2
    heightmap_2d_patch = heightmap_2d[:, height//2-window:height//2+window+1, width//2-window:width//2+window+1].reshape(-1, (2*window+1)*(2*window+1))
    height_mask = heightmap_2d_patch.max(dim=1).values - heightmap_2d_patch.min(dim=1).values > 1e-3
    
    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions
    
    reward = height_mask * processed_actions[:, :2].norm(dim=1)

    return reward


def foot_placement_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    action_name: str = "mpc_action",
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
    std: float = 0.5, 
    offset: float = 0.55,
):
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
    sensor_offset = (raycaster.cfg.offset.pos[0], raycaster.cfg.offset.pos[1])
    heightmap_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    
    # if using foot placement 
    foot_selection = 1-action_term.gait_contact
    foot_position_b = action_term.foot_placement_b.reshape(-1, 2, 2)
    
    # retrieves ground flatness where stance foot position is projected onto height map.
    roughness_at_foot, _ = get_ground_roughness_at_landing_point(
        env.num_envs,
        sensor_offset,
        foot_position_b,
        heightmap_2d, # heightmap
        resolution,
        l_toe,
        l_heel,
        l_width,
    )
    
    # reward = (foot_selection * torch.exp(-torch.abs(ground_flatness_at_foot)/std)).sum(dim=1) # exponential reward
    reward = (foot_selection * torch.exp(-torch.square(roughness_at_foot)/(std**2 + 1e-6))).sum(dim=1) # gaussian reward
    # print(reward)
    
    return reward

def stance_foot_position_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    action_name: str = "mpc_action",
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
    std: float = 0.5, 
    offset: float = 0.55,
):
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
    sensor_offset = (raycaster.cfg.offset.pos[0], raycaster.cfg.offset.pos[1])
    heightmap_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    
    # if using actual landing location 
    contact = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=2) > 1.0
    # foot_selection = contact * (1 - action_term.gait_contact)
    foot_selection = contact
    foot_position_b = action_term.foot_pos_b.reshape(-1, 2, 3)
    
    # retrieves ground flatness where stance foot position is projected onto height map.
    roughness_at_foot, _ = get_ground_roughness_at_landing_point(
        env.num_envs,
        sensor_offset,
        foot_position_b,
        heightmap_2d, # heightmap
        resolution,
        l_toe,
        l_heel,
        l_width,
    )
    
    # reward = (foot_selection * torch.exp(-torch.abs(ground_flatness_at_foot)/std)).sum(dim=1) # exponential reward
    reward = (foot_selection * torch.exp(-torch.square(roughness_at_foot)/(std**2 + 1e-6))).sum(dim=1) # gaussian reward
    # print(reward)

    return reward

"""
penalties
"""

def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize y-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])

def foot_placement_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    action_name: str = "mpc_action",
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
    std: float = 0.5, 
    offset: float = 0.55,
):
    """
    Given robot-centric elevation map, project planned footholds onto the map, and get elevation of sampling points on the edge of foothold. 
    Finally, find roughness of foothold by computing the difference between the maximum and minimum height of the sampling points.
    """
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(round(grid_x/resolution, 4)) + 1, int(round(grid_y/resolution, 4)) + 1
    sensor_offset = (raycaster.cfg.offset.pos[0], raycaster.cfg.offset.pos[1])
    heightmap_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    
    # if using foot placement 
    foot_selection = 1-action_term.gait_contact
    foot_position_b = action_term.foot_placement_b.reshape(-1, 2, 2)
    
    # retrieves ground flatness where stance foot position is projected onto height map.
    roughness_at_foot, _ = get_ground_roughness_at_landing_point(
        env.num_envs,
        sensor_offset,
        foot_position_b,
        heightmap_2d, # heightmap
        resolution,
        l_toe,
        l_heel,
        l_width,
    )
    
    # penalty = (foot_selection * (1 - torch.exp(-torch.abs(roughness_at_foot)/std))).max(dim=1).values # exponential reward
    penalty = (foot_selection * (1 - torch.exp(-torch.square(roughness_at_foot)/(std**2 + 1e-6)))).max(dim=1).values # gaussian reward
    
    return penalty

def stance_foot_position_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    action_name: str = "mpc_action",
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
    std: float = 0.5, 
    offset: float = 0.55,
):
    """
    Given robot-centric elevation map, project actual touchdown position onto the map, and get elevation of sampling points on the edge of touchdown position. 
    Finally, find roughness by computing the difference between the maximum and minimum height of the sampling points.
    """
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(round(grid_x/resolution, 4)) + 1, int(round(grid_y/resolution, 4)) + 1
    sensor_offset = (raycaster.cfg.offset.pos[0], raycaster.cfg.offset.pos[1])
    heightmap_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    
    # contact = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=2) > 1.0
    first_contact = (contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=3) > 1.0).sum(dim=1).float() == 1.0 # (num_envs, num_body_ids)
    foot_position_b = action_term.foot_pos_b.reshape(-1, 2, 3)
    
    # retrieves ground flatness where stance foot position is projected onto height map.
    roughness_at_foot, _ = get_ground_roughness_at_landing_point(
        env.num_envs,
        sensor_offset,
        foot_position_b,
        heightmap_2d, # heightmap
        resolution,
        l_toe,
        l_heel,
        l_width,
    )
    
    # penalty = (first_contact * (1 - torch.exp(-torch.abs(roughness_at_foot)/std))).max(dim=1).values # exponential
    penalty = (first_contact * (1 - torch.exp(-torch.square(roughness_at_foot)/(std**2 + 1e-6)))).max(dim=1).values # gaussian

    return penalty

def swing_foot_landing_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    offset: float = 0.55,
    std: float = 0.5, 
):
    """
    Get foot-centric heightmap and compute roughness of neighboring terrain using max-min height difference.
    """
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    # contact = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=2) > 1.0
    # first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, contact_sensor_cfg.body_ids] # this method does not capture correct contact mode...
    first_contact = (contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=3) > 1.0).sum(dim=1).float() == 1.0 # (num_envs, num_body_ids)
    
    roughness_at_foot = height_map.max(dim=1).values - height_map.min(dim=1).values # (num_envs, )
    penalty = first_contact.squeeze(-1) * (1 - torch.exp(-torch.square(roughness_at_foot)/(std**2 + 1e-6))) # gaussian
    
    return penalty

def log_barrier_swing_foot_landing_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    l_toe: float = 0.091,
    l_heel: float = 0.054,
    l_width: float = 0.073,
):
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    
    boundary_grid_point_in_body = action_term.grid_point_boundary_in_body # (num_envs, n, 3)
    
    contact = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=2) > 1.0
    first_contact = (contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=3) > 1.0).sum(dim=1).float() == 1.0 # (num_envs, 2)
    foot_position_b = action_term.foot_pos_b.reshape(-1, 2, 3) # (num_envs, 2, 3)
    stance_position_b = (foot_position_b * contact[:, :, None]).sum(dim=1) # (num_envs, 3)
    
    # -l_heel <= dx <= l_toe and -l_width/2 <= dy <= l_width/2
    dx = boundary_grid_point_in_body[:, :, 0] - stance_position_b[:, None, 0]
    dy = boundary_grid_point_in_body[:, :, 1] - stance_position_b[:, None, 1]
    mask = (dx <= l_toe) & (-l_heel <= dx) & (dy.abs() <= l_width / 2) # (num_envs, n)
    
    eps = 1e-6  # small epsilon to avoid log(0)
    penalty_x_lb = -(torch.log(((-dx) * (dx<0) + eps) / (l_heel + eps))) * (dx < 0).float() # -l_heel >= dx > 0
    penalty_x_ub = -(torch.log((dx * (dx>=0) + eps) / (l_toe + eps))) * (dx >= 0).float() # 0 <= dx < l_toe
    penalty_x = penalty_x_lb + penalty_x_ub # (num_envs, n)
    penalty_y = -torch.log((dy.abs() + eps) / ((l_width / 2) + eps))
    penalty_x = torch.clamp(penalty_x, min=0.0)
    penalty_y = torch.clamp(penalty_y, min=0.0)
    
    # penalty_x_lb = (penalty_x_lb * mask.float()).sum(dim=1) * first_contact.sum(dim=1).float()
    # penalty_x_ub = (penalty_x_ub * mask.float()).sum(dim=1) * first_contact.sum(dim=1).float()
    
    penalty_x = (penalty_x * mask.float()).sum(dim=1) * first_contact.sum(dim=1).float()
    penalty_y = (penalty_y * mask.float()).sum(dim=1) * first_contact.sum(dim=1).float() * 0 # disable y penalty
    penalty = penalty_x + penalty_y
    
    return penalty
    

def negative_lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize negative forward velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    reward = torch.square(vel_x) * (vel_x < 0).float()  # penalize only if moving backwards
    return reward

def energy_penalty_l2(env: ManagerBasedRLEnv, assymetric_indices: int|list[int], action_name: str = "mpc_action") -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action_term = env.action_manager.get_term(action_name)
    actions = action_term.raw_actions.clone()
    actions[:, assymetric_indices] = 1 + actions[:, assymetric_indices]  # handle assymetry (enforcing 0 action means -1 raw action)
    return torch.sum(torch.square(actions), dim=1).view(-1)

def individual_action_l2(env: ManagerBasedRLEnv, action_idx:int|list[int], action_name: str = "mpc_action",) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions
    picked_action = processed_actions[:, action_idx]
    if len(picked_action.shape) == 2:
        value = torch.sum(torch.square(picked_action), dim=1)
    elif len(picked_action.shape) == 1:
        value = torch.square(picked_action)
    return value.view(-1)

def processed_action_l2(env: ManagerBasedRLEnv, action_name: str = "mpc_action",) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions
    return torch.sum(torch.square(processed_actions), dim=1).view(-1)

def rough_terrain_processed_action_l2(
    env: ManagerBasedRLEnv, 
    action_idx:int|list[int],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    action_name: str = "mpc_action",
    offset: float=0.55, 
    lookahead_distance: float=0.25, 
    lookback_distance: float = 0.1,
    patch_width: float = 0.15) -> torch.Tensor:
    
    """
    Penalize the actions using L2 squared kernel.
    This is different from standard L2 energy penalty as the penalty is non-zero only when terrain is flat. 
    This means you penalize energy only when the robot is stepping on flat terrain, but actively apply action otherwise.
    """
    
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    
    # process height map
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    num_envs = height_map.shape[0]
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(round(grid_x/resolution, 4)) + 1, int(round(grid_y/resolution, 4)) + 1
    
    # extract region-of-interest small patch
    window_front = int(lookahead_distance/resolution)
    window_back = int(lookback_distance/resolution)
    window_side = int(patch_width/resolution)
    height_map_patch = height_map.reshape(-1, height, width)[:, height//2-window_side: height//2+window_side+1, width//2-window_back:width//2+window_front+1].reshape(num_envs, -1)
    roughness = height_map_patch.max(dim=1).values - height_map_patch.min(dim=1).values # (num_envs, )
    
    # process energy penalty
    processed_actions = action_term.processed_actions
    picked_action = processed_actions[:, action_idx]
    if len(picked_action.shape) == 2:
        value = torch.sum(torch.square(picked_action), dim=1)
    elif len(picked_action.shape) == 1:
        value = torch.square(picked_action)
        
    energy_penalty = value.view(-1) * (roughness < 1e-2).float() # on flat terrain, penalize energy
    return energy_penalty

def leg_body_angle_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    sagittal_leg_body_angle = action_term.leg_angle[:, [0, 2]] # (num_envs, 2)
    reward = torch.sum(torch.square(sagittal_leg_body_angle), dim=1)
    return reward

def leg_distance_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    foot_pos_b = action_term.foot_pos_b # (num_envs, 6)
    foot_pos_b_lateral = foot_pos_b[:, [1, 4]]
    reward = torch.square(foot_pos_b_lateral[:, 0] - foot_pos_b_lateral[:, 1])
    return reward

def mpc_cost_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    mpc_cost = action_term.mpc_cost.clamp(max=1e3) # (num_envs, 1)
    return torch.abs(mpc_cost)


def feet_accel_l2(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet acceleration.

    This function penalizes the agent for accelerating its feet on the ground. The reward is computed as the
    norm of the linear acceleration of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet acceleration
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1).values > 1.0).float()
    asset = env.scene[asset_cfg.name]

    body_accel = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :] # foot acceleration
    reward = torch.sum(body_accel.norm(dim=-1) * (1-contacts), dim=1) # swing foot acceleration
    return reward

def depth_image_r(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    depth_image = sensor.data.output["distance_to_camera"]
    
    num_envs = depth_image.shape[0]
    timestamp = datetime.now().strftime("%H%M%S")
    gray = (depth_image[0, :, :, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(f"/home/jkamohara3/Downloads/depth/{timestamp}.png", color)
    
    return depth_image.reshape(num_envs, -1).sum(dim=1)